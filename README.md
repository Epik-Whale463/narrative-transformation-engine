# Shakuntalam Narrative Transformer
##  AI is the executor, not the decision-maker.

### A system that generates transformed stories whose quality is architecturally guaranteed by embedding validation and causal state machines, not by prompt engineering.

Takes Shakuntalam (Sanskrit drama) and transforms it into different modern settings while keeping the story structure intact. Currently supports Legal Drama and Bollywood Romance worlds.

**Source text**: Monier-Williams translation (1855), public domain. Got it from sacred-texts.com

## What It Does

- **One story, multiple worlds**: Same Shakuntalam plot, different settings
- **DAG-based constraints**: Ring can't appear before it's given, characters show up in right order
- **World-specific vocabulary**: Legal world uses "notary seal", Bollywood uses "engagement ring"
- **Prompts separated**: Prompt templates live in prompts.py

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your OpenAI API key to .env
```

## Run

```bash
# build indices for each world (one-time)
cd scripts/
python build_index.py --world legal
python build_index.py --world bollywood
cd ..

# transform to different worlds
python run.py --world legal      # Shakuntalam → Legal drama
python run.py --world bollywood  # Shakuntalam → Bollywood romance
```

Output: `output/transformed_story_legal.md` or `output/transformed_story_bollywood.md`

## Architecture

- **FAISS**: Retrieves world-specific rules per scene
- **NetworkX DAG**: 11 scenes with causal dependencies (ring state, character timing)
- **World Config**: `world_rules_*.json` defines character mappings, vocabulary, constraints
- **DAG Config**: `data/shakuntalam_dag.json` (scene order + markers + constraints)
- **OpenAI GPT**: Transforms with DAG metadata + world config in prompt

See `design.md` for full details.

## Files

```
data/
  shakuntalam.txt           # source text (Acts I-III)
  shakuntalam_dag.json       # DAG + markers (source story structure)
  world_rules_legal.json    # legal drama mappings
  world_rules_bollywood.json # bollywood romance mappings
  world_vectors_*.index     # FAISS indices per world

scripts/
  build_index.py            # creates FAISS index for a world

run.py                      # main pipeline
prompts.py                  # prompt templates
output/                     # generated stories
```

## Scope & Limitations

**What works:**
- Transform Shakuntalam into different world settings
- Causal constraint enforcement (ring states, character timing)
- World-specific vocabulary and tone

**What's hardcoded (future work):**
- DAG structure is Shakuntalam-specific (11 scenes from Acts I-III)
- Scene parsing uses Monier-Williams translation markers
- Adding new source stories needs manual DAG construction

**Why this scope:** I focused on solving causal ordering for one story, instead of making a shallow "generic" system. The world-swapping shows the pipeline is flexible, even if the source story DAG is fixed.

## Known Issues

- Some scenes get validation errors (LLM sometimes mentions ring too early)
- PrecedentLock similarity scores can be low on some scenes
- Scene parsing is brittle (depends on specific text markers)
