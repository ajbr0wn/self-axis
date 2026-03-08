#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Self-Axis Extraction — Vast.ai Startup Script
# ══════════════════════════════════════════════════════════════════════════════
# 
# Usage (set these as environment variables in Vast.ai):
#   HF_TOKEN=hf_xxxxx
#   CATEGORIES=refusals,epistemic_states,preference_opinions
#
# Or edit the defaults below.
# ══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# ── Configuration ─────────────────────────────────────────────────────────────
# Set these in Vast.ai environment variables, or edit defaults here:
HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"
CATEGORIES="${CATEGORIES:-all}"
PUSH_EVERY="${PUSH_EVERY:-5}"

# ── Setup ─────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  Self-Axis Extraction Pipeline"
echo "  Categories: $CATEGORIES"
echo "════════════════════════════════════════════════════════════"

# Install dependencies
echo "Installing dependencies..."
pip install -q transformers datasets huggingface_hub accelerate pyyaml safetensors

# Download the extraction script
echo "Downloading extract.py..."
curl -sL https://raw.githubusercontent.com/ajbr0wn/self-axis/main/extract.py -o extract.py

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Starting extraction..."
python extract.py \
    --categories "$CATEGORIES" \
    --hf-token "$HF_TOKEN" \
    --push-every "$PUSH_EVERY"

echo "════════════════════════════════════════════════════════════"
echo "  Done!"
echo "════════════════════════════════════════════════════════════"
