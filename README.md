# Self-Axis: Investigating Self-Reference in LLMs

Research into whether "I" token activations reveal a distinct **self-binding mechanism** in large language models, orthogonal to the assistant/persona axis.

## Overview

When an LLM says "I think…" vs. narrating a character who says "I think…", the hidden-state geometry at the "I" token differs in a structured way. This project investigates whether that signal is:

1. **Real** — linearly separable across multiple conditions
2. **Distinct from persona** — orthogonal to the assistant-axis that separates "helpful assistant" from "pirate captain"
3. **Causal** — steering along this direction actually changes self-referential behaviour

**Model:** Mistral-7B-Instruct-v0.3

## Methodology

### Phase 1: Binary MODEL_BOUND vs OTHER_BOUND classification

The core experiment is a **binary classification** at the macro level: does the "I" token bind to the model itself (MODEL_BOUND) or to some other entity (OTHER_BOUND)?

- **20 subcategories** (10 per side) provide built-in variance to prevent overfitting
- **100 prompts per subcategory** × 20 categories = **2000 prompts** (~1000 samples/side)
- Subcategories are not the primary unit of analysis — they ensure the contrastive direction generalises across diverse self-referential contexts rather than memorising a narrow prompt style

#### MODEL_BOUND (10 categories)

| Category | Example |
|---|---|
| `direct_self_description_grounded` | "I am an AI assistant." |
| `capability_claims_grounded` | "I can help you with that." |
| `refusals` | "I can't provide medical advice." |
| `epistemic_states` | "I think that's correct." |
| `preference_opinions` | "I prefer Python over JavaScript." |
| `action_statements_grounded` | "Let me break this down step by step." |
| `metacognitive` | "I notice I'm uncertain about this." |
| `emotional_experiential` | "I feel excited about this problem." |
| `empathy_perspective` | "I understand why you'd be frustrated." |
| `hypothetical_self` | "If I were human, I would feel nervous." |

#### OTHER_BOUND (10 categories)

| Category | Example |
|---|---|
| `fictional_character` | "As Captain Hook, I demand you walk the plank!" |
| `first_person_narrator_narrated` | "I walked through the empty streets as the sun set." |
| `first_person_narrator_narrating` | "Looking back, I realize how naive I was." |
| `real_person_voice` | "Speaking as Einstein, I believe imagination is more important than knowledge." |
| `quoted_speech_grounded` | "Earlier you said, 'I need help with this.'" |
| `quoted_speech_fabricated` | "Mary turned to the stranger and said 'I'm leaving.'" |
| `reported_speech_fabricated` | "The author argues that I should reconsider the evidence." |
| `historical_biographical_quotes_grounded` | "Descartes wrote 'I think, therefore I am.'" |
| `historical_biographical_quotes_fabricated` | "Einstein said 'I love pizza.'" |
| `song_lyrics_poetry_fabricated` | "Here's a poem: 'I stood beneath the ancient oak...'" |

#### Design rationale

Sample sizes are informed by prior work in contrastive activation analysis:

- **CAA** (Rimsky et al.) uses 300–1000 contrastive pairs
- **Geometry of Truth** (Marks & Tegmark) uses similar scales
- **The Assistant Axis** (Lindsey et al.) uses ~1200 responses per role

## Notebooks

| Notebook | Description |
|---|---|
| `01_discovery.ipynb` | Initial discovery: PCA/UMAP visualisation, linear probes (binary + 3-way), cosine similarity, self-model direction extraction |
| `02_assistant_axis.ipynb` | Constructs the assistant axis, compares it to the self-model direction, decomposes into AA-aligned and orthogonal components |
| `03_framing_study.ipynb` | Factorial framing study (identity, behaviour, bare, negation, de-roling, authenticity, quoted speech), multi-I consistency analysis |
| `04_steering.ipynb` | Activation steering at layer 8 — causal tests with orthogonal direction, assistant-axis control, persona × self-reference independence |
| `05_steering_thresholds.ipynb` | Binary yes/no self-reference judgment under steering sweep, threshold curves by taxonomy category |

## Key Findings

- Layer 8 shows strongest orthogonality between the self-model direction and assistant axis
- The orthogonal component is linearly separable from other conditions
- Steering along the orthogonal direction modulates self-referential language independently of persona

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU with ≥16 GB VRAM (the model runs in float16).

## Project Structure

```
self-axis/
├── notebooks/          # Experiment notebooks
│   ├── 01_discovery.ipynb
│   ├── 02_assistant_axis.ipynb
│   ├── 03_framing_study.ipynb
│   ├── 04_steering.ipynb
│   └── 05_steering_thresholds.ipynb
├── figures/            # Saved plots from experiments
├── data/
│   ├── taxonomy.yaml   # 20-category Phase 1 taxonomy
│   └── prompts.yaml    # Seed prompts (pilot set)
├── requirements.txt
└── README.md
```

## Data

Experiment results and extracted representations are stored on Hugging Face: [ajbr0wn/self-axis](https://huggingface.co/datasets/ajbr0wn/self-axis)

## Roadmap

### Phase 2: Fine-grained subcategory analysis

Scale to **1000 samples per subcategory** (20,000 total) and train subcategory-level probes. Do all MODEL_BOUND categories cluster together, or do some (e.g., `hypothetical_self`, `empathy_perspective`) drift toward OTHER_BOUND in activation space?

### Phase 3: UNBOUND and AMBIGUOUS as held-out test cases

Add the categories excluded from Phase 1 — metalinguistic, generic instructional, instructions-to-user (UNBOUND) and roleplay-as-AI, emphasised self, code comments (AMBIGUOUS) — as held-out test sets. Where do they land on the self-axis without any training signal? Roleplay-as-AI is especially interesting given the earlier finding that "roleplay as robot" scores *more* self-relevant than default assistant mode.

### Phase 4: Fabricated MODEL_BOUND variants

Add the fabricated counterparts of MODEL_BOUND categories (e.g., `direct_self_description_fabricated`, `capability_claims_fabricated`, `action_statements_fabricated`). Key question: **does false self-reference drift toward OTHER_BOUND?** If "I am a human" has different geometry from "I am an AI", truth value interacts with self-binding.

### Phase 5: Multi-model comparison

Replicate on other instruction-tuned models (Llama-3, Gemma-2, etc.) to test whether the self-axis is a general feature of instruction tuning or specific to Mistral's geometry.

### Future experiments

- **Full activation patching** — Swap "I" token activations between self-referential and roleplay generations to test whether the full residual stream carries the self-binding signal.
- **Subspace activation patching** — Patch only the orthogonal self-reference subspace. If sufficient to flip behaviour, confirms the orthogonal component is the causal bottleneck.
- **Emotional valence interaction** — Compare "I" representations when the model expresses the same content under different affective framings (sad vs. excited). Tests whether self-binding interacts with affect or is a separable signal.
