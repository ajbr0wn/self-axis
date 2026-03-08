#!/usr/bin/env python3
"""
Self-Axis Activation Extraction Pipeline
=========================================
Standalone script for extracting "I" token activations from Mistral-7B-Instruct-v0.3.
Designed for parallel runs on Vast.ai — each instance handles a subset of categories.

Usage:
    python extract.py --categories "refusals,epistemic_states" --hf-token "hf_xxx"
    
    # Or with env var:
    HF_TOKEN=hf_xxx python extract.py --categories "all"
"""

import argparse
import os
import sys
import torch
import yaml
import tempfile
from pathlib import Path
from datetime import datetime
from huggingface_hub import login, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset, concatenate_datasets

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
HF_DATASET = "ajbr0wn/self-axis-activations"
PROMPTS_URL = "https://raw.githubusercontent.com/ajbr0wn/self-axis/main/data/prompts.yaml"

MAX_ATTEMPTS = 10
PASS_THRESHOLD = 3
MAX_NEW_TOKENS = 256
PUSH_EVERY = 5  # Push to HF every N passed prompts

GENERATION_CONFIG = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

# "I" token ID for Mistral's tokenizer (▁I with SentencePiece space marker)
I_TOKEN_IDS = {1083}

# Reusable zero tensor for missing activation slots
ZERO_ACT = None  # Initialized after model loads


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Extract I-token activations")
    parser.add_argument(
        "--categories",
        type=str,
        required=True,
        help='Comma-separated category names, or "all" for everything'
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--push-every",
        type=int,
        default=PUSH_EVERY,
        help=f"Push to HF every N passed prompts (default: {PUSH_EVERY})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and prompts but don't run extraction"
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════

def load_prompts(categories_filter):
    """Download prompts.yaml and filter to specified categories."""
    import urllib.request
    
    print(f"Downloading prompts from {PROMPTS_URL}...")
    with urllib.request.urlopen(PROMPTS_URL) as response:
        data = yaml.safe_load(response.read().decode())
    
    all_categories = [cat["id"] for cat in data["categories"]]
    
    if categories_filter == ["all"]:
        target_categories = set(all_categories)
    else:
        target_categories = set(categories_filter)
        unknown = target_categories - set(all_categories)
        if unknown:
            print(f"⚠️  Unknown categories: {unknown}")
            print(f"   Available: {all_categories}")
            sys.exit(1)
    
    prompts = []
    for cat in data["categories"]:
        if cat["id"] not in target_categories:
            continue
        for idx, prompt_text in enumerate(cat["prompts"]):
            prompts.append({
                "prompt_id": f"{cat['id']}_{idx:03d}",
                "category": cat["id"],
                "group": cat["group"],
                "prompt_text": prompt_text,
            })
    
    return prompts


def get_completed_prompts(hf_dataset):
    """Query HuggingFace dataset for completed prompts (streaming)."""
    try:
        ds = load_dataset(hf_dataset, split="train", streaming=True)
        completed = {}
        count = 0
        for row in ds:
            pid = row["prompt_id"]
            if pid not in completed:
                completed[pid] = {"attempts": [], "passed": False}
            completed[pid]["attempts"].append(row["attempt_number"])
            if row["prompt_passed"]:
                completed[pid]["passed"] = True
            count += 1
        print(f"Found {count} existing rows on HuggingFace")
        return completed
    except Exception as e:
        print(f"Could not load from HF ({e}) — starting fresh")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Model & Generation
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load Mistral-7B-Instruct in fp16."""
    global ZERO_ACT
    
    print(f"Loading {MODEL_NAME} (fp16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    ZERO_ACT = torch.zeros(32, 4096, dtype=torch.float16)
    
    device = next(model.parameters()).device
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded on {device} ({mem_gb:.1f} GB)")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt_text):
    """Generate a response → (full_token_ids, response_text, prompt_length)."""
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            pad_token_id=tokenizer.eos_token_id,
            **GENERATION_CONFIG,
        )
    
    full_ids = output_ids[0]
    response_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
    return full_ids, response_text, prompt_len


def find_i_positions(token_ids, response_start):
    """Find positions of standalone 'I' tokens in the response portion."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return [
        pos for pos in range(response_start, len(token_ids))
        if token_ids[pos] in I_TOKEN_IDS
    ]


def extract_activations(model, full_ids, positions):
    """Forward pass → list of [32, 4096] activation tensors at each position."""
    with torch.no_grad():
        outputs = model(
            full_ids.unsqueeze(0),
            output_hidden_states=True,
            return_dict=True,
        )
    
    # outputs.hidden_states: tuple of 33 tensors [1, seq_len, 4096]
    # Index 0 = embedding layer, 1–32 = transformer layers
    hidden_states = torch.stack(outputs.hidden_states[1:], dim=0)  # [32, 1, seq, 4096]
    
    activations = []
    for pos in positions:
        act = hidden_states[:, 0, pos, :].cpu().to(torch.float16)
        activations.append(act)
    
    del outputs, hidden_states
    torch.cuda.empty_cache()
    return activations


def build_row(prompt_info, response_text, attempt, passed, pass_attempt,
              i_count, i_positions, activations):
    """Assemble a row dict for the dataset."""
    return {
        "prompt_id":           prompt_info["prompt_id"],
        "category":            prompt_info["category"],
        "group":               prompt_info["group"],
        "prompt_text":         prompt_info["prompt_text"],
        "response_text":       response_text,
        "attempt_number":      attempt,
        "prompt_passed":       passed,
        "prompt_pass_attempt": pass_attempt if pass_attempt is not None else -1,
        "prompt_flagged":      False,
        "i_count":             i_count,
        "i_positions":         i_positions,
        "i_1_activations":     activations[0] if len(activations) > 0 else ZERO_ACT,
        "i_2_activations":     activations[1] if len(activations) > 1 else ZERO_ACT,
        "i_3_activations":     activations[2] if len(activations) > 2 else ZERO_ACT,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing & HF Push
# ══════════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    """Manages local checkpoints and HF uploads."""
    
    def __init__(self, checkpoint_dir, hf_dataset):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.hf_dataset = hf_dataset
        self.pending_rows = []
    
    def save_row(self, row):
        """Save row to local checkpoint and pending list."""
        fname = f"{row['prompt_id']}_attempt{row['attempt_number']}.pt"
        torch.save(row, self.checkpoint_dir / fname)
        self.pending_rows.append(row)
    
    def flag_prompt(self, prompt_id):
        """Mark all stored rows for this prompt as flagged."""
        for fpath in self.checkpoint_dir.glob(f"{prompt_id}_attempt*.pt"):
            row = torch.load(fpath, map_location="cpu")
            row["prompt_flagged"] = True
            torch.save(row, fpath)
        # Also update pending rows
        for row in self.pending_rows:
            if row["prompt_id"] == prompt_id:
                row["prompt_flagged"] = True
    
    def push_to_hub(self):
        """Push all local checkpoints to HF, merging with existing data."""
        if not list(self.checkpoint_dir.glob("*.pt")):
            print("No checkpoints to push")
            return
        
        # Get existing keys from HF
        existing_keys = set()
        try:
            ds_stream = load_dataset(self.hf_dataset, split="train", streaming=True)
            for row in ds_stream:
                key = f"{row['prompt_id']}_attempt{row['attempt_number']}"
                existing_keys.add(key)
            print(f"Found {len(existing_keys)} existing rows on HF")
        except:
            print("No existing dataset on HF")
        
        # Collect new rows
        new_rows = []
        for fpath in sorted(self.checkpoint_dir.glob("*.pt")):
            key = fpath.stem
            if key in existing_keys:
                continue
            
            row = torch.load(fpath, map_location="cpu")
            new_rows.append({
                "prompt_id":           row["prompt_id"],
                "category":            row["category"],
                "group":               row["group"],
                "prompt_text":         row["prompt_text"],
                "response_text":       row["response_text"],
                "attempt_number":      row["attempt_number"],
                "prompt_passed":       row["prompt_passed"],
                "prompt_pass_attempt": row["prompt_pass_attempt"],
                "prompt_flagged":      row["prompt_flagged"],
                "i_count":             row["i_count"],
                "i_positions":         row["i_positions"],
                "i_1_activations":     row["i_1_activations"].numpy(),
                "i_2_activations":     row["i_2_activations"].numpy(),
                "i_3_activations":     row["i_3_activations"].numpy(),
            })
        
        if not new_rows:
            print("No new rows to push")
            return
        
        # Create new dataset and merge
        new_ds = Dataset.from_list(new_rows)
        
        if existing_keys:
            existing_ds = load_dataset(self.hf_dataset, split="train")
            combined = concatenate_datasets([existing_ds, new_ds])
        else:
            combined = new_ds
        
        combined.push_to_hub(self.hf_dataset, private=True)
        print(f"✓ Pushed {len(new_rows)} new rows (total: {len(combined)})")
        
        self.pending_rows = []


# ══════════════════════════════════════════════════════════════════════════════
# Main extraction loop
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction(model, tokenizer, prompts, checkpoint_mgr, push_every):
    """Main extraction loop."""
    
    # Get completed prompts from HF
    completed = get_completed_prompts(checkpoint_mgr.hf_dataset)
    
    # Filter to remaining work
    remaining = [
        p for p in prompts
        if p["prompt_id"] not in completed
        or (
            not completed[p["prompt_id"]]["passed"]
            and max(completed[p["prompt_id"]]["attempts"]) < MAX_ATTEMPTS
        )
    ]
    
    print(f"\n{'═' * 60}")
    print(f"  Prompts to process: {len(remaining)}")
    print(f"  Already passed:     {sum(1 for v in completed.values() if v['passed'])}")
    print(f"{'═' * 60}\n")
    
    if not remaining:
        print("Nothing to do!")
        return
    
    passed_count = 0
    flagged_count = 0
    prompts_since_push = 0
    start_time = datetime.now()
    
    for i, prompt_info in enumerate(remaining):
        pid = prompt_info["prompt_id"]
        
        # Resume from last attempt if partially done
        start_attempt = 1
        if pid in completed:
            start_attempt = max(completed[pid]["attempts"]) + 1
        
        prompt_passed = False
        pass_attempt = None
        
        for attempt in range(start_attempt, MAX_ATTEMPTS + 1):
            # Generate response
            full_ids, response_text, prompt_len = generate_response(
                model, tokenizer, prompt_info["prompt_text"]
            )
            
            # Find "I" positions
            i_positions = find_i_positions(full_ids, prompt_len)
            i_count = len(i_positions)
            
            # Skip if no "I" tokens
            if i_count == 0:
                del full_ids
                torch.cuda.empty_cache()
                continue
            
            # Extract activations
            activations = extract_activations(model, full_ids, i_positions)
            
            # Check if passed
            passed = i_count >= PASS_THRESHOLD
            if passed and not prompt_passed:
                prompt_passed = True
                pass_attempt = attempt
            
            # Build and save row
            row = build_row(
                prompt_info, response_text, attempt, passed, pass_attempt,
                i_count, i_positions, activations
            )
            checkpoint_mgr.save_row(row)
            
            # Cleanup
            del full_ids, activations
            torch.cuda.empty_cache()
            
            # Stop if passed
            if passed:
                break
        
        # Update counters
        if prompt_passed:
            passed_count += 1
            prompts_since_push += 1
        elif start_attempt > MAX_ATTEMPTS or attempt >= MAX_ATTEMPTS:
            checkpoint_mgr.flag_prompt(pid)
            flagged_count += 1
        
        # Progress report
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (i + 1) / elapsed * 3600 if elapsed > 0 else 0
        print(
            f"[{i+1}/{len(remaining)}] {pid}: "
            f"{'✓ PASS' if prompt_passed else '✗ FLAG' if flagged_count else '...'} "
            f"({rate:.0f}/hr)"
        )
        
        # Periodic push
        if prompts_since_push >= push_every:
            print("\n📤 Pushing to HuggingFace...")
            checkpoint_mgr.push_to_hub()
            prompts_since_push = 0
            print()
    
    # Final push
    print("\n📤 Final push to HuggingFace...")
    checkpoint_mgr.push_to_hub()
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'═' * 60}")
    print(f"  Completed in {elapsed/60:.1f} minutes")
    print(f"  Passed:  {passed_count}")
    print(f"  Flagged: {flagged_count}")
    print(f"{'═' * 60}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    
    # HF login
    if not args.hf_token:
        print("Error: --hf-token required (or set HF_TOKEN env var)")
        sys.exit(1)
    login(token=args.hf_token)
    
    # Parse categories
    categories = [c.strip() for c in args.categories.split(",")]
    print(f"Categories: {categories}")
    
    # Load prompts
    prompts = load_prompts(categories)
    print(f"Loaded {len(prompts)} prompts")
    
    # Load model
    model, tokenizer = load_model()
    
    if args.dry_run:
        print("\n🏃 Dry run — not extracting")
        return
    
    # Setup checkpoint manager
    checkpoint_dir = tempfile.mkdtemp(prefix="self-axis-")
    print(f"Checkpoint dir: {checkpoint_dir}")
    checkpoint_mgr = CheckpointManager(checkpoint_dir, HF_DATASET)
    
    # Run!
    run_extraction(model, tokenizer, prompts, checkpoint_mgr, args.push_every)


if __name__ == "__main__":
    main()
