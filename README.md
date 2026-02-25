# Self-Axis: Investigating Self-Reference in LLMs

Research into whether "I" token activations reveal a distinct **self-binding mechanism** in large language models, orthogonal to the assistant/persona axis.

## Overview

When an LLM says "I think…" vs. narrating a character who says "I think…", the hidden-state geometry at the "I" token differs in a structured way. This project investigates whether that signal is:

1. **Real** — linearly separable across multiple conditions
2. **Distinct from persona** — orthogonal to the assistant-axis that separates "helpful assistant" from "pirate captain"
3. **Causal** — steering along this direction actually changes self-referential behaviour

**Model:** Mistral-7B-Instruct-v0.3

## Notebooks

| Notebook | Description |
|---|---|
| `01_discovery.ipynb` | Initial discovery: PCA/UMAP visualisation, linear probes (binary + 3-way), cosine similarity, self-model direction extraction |
| `02_assistant_axis.ipynb` | Constructs the assistant axis, compares it to the self-model direction, decomposes into AA-aligned and orthogonal components |
| `03_framing_study.ipynb` | Factorial framing study (identity, behaviour, bare, negation, de-roling, authenticity, quoted speech), multi-I consistency analysis |
| `04_steering.ipynb` | Activation steering at layer 8 — causal tests with orthogonal direction, assistant-axis control, persona × self-reference independence |

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU with ≥16 GB VRAM (the model runs in float16).

## Key Findings

- Layer 8 shows strongest orthogonality between the self-model direction and assistant axis
- The orthogonal component is linearly separable from other conditions
- Steering along the orthogonal direction modulates self-referential language independently of persona

## Project Structure

```
self-axis/
├── notebooks/          # Experiment notebooks
│   ├── 01_discovery.ipynb
│   ├── 02_assistant_axis.ipynb
│   ├── 03_framing_study.ipynb
│   └── 04_steering.ipynb
├── figures/            # Saved plots from experiments
├── data/               # Experiment data and results
├── requirements.txt
└── README.md
```
