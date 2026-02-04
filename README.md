# Shakuntalam Legal Transformation

## A pipeline that generates stories grounded in a defined world and template, ensuring consistency and coherence beyond generic LLM outputs.

Transforms Shakuntalam (Sanskrit drama) into modern legal drama. Uses RAG + DAG to keep narrative consistent.

**Source text**: Monier-Williams translation (1855), public domain. Got it from sacred-texts.com

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your OpenAI API key to .env
```

## Run

```bash
cd scripts/
python build_index.py  # one-time setup
cd ..
python run.py --validate
```

Output: `output/transformed_story.md`

---

## Manual Fixes

2 scenes need editing:

| Scene | Issue | Fix |
|-------|-------|-----|
| Union | "won't return seal" | Change to: "seal you gave me" |
| Separation | Mathavya present | Remove him (only Gautami) |

LLM forgets context between scenes sometimes. DAG catches structure errors but not all semantic issues.

---

## Architecture

- **FAISS**: Retrieves world rules per scene
- **NetworkX DAG**: Enforces scene order + constraints (ring state, character timing)
- **OpenAI GPT**: Transforms with DAG metadata in prompt

See `design.md` for details.

---

## Files

- `data/world_rules.json` - mappings (Dushyanta → Senior Associate, ring → notary seal)
- `data/shakuntalam.txt` - source text (Acts I-III)
- `build_index.py` - FAISS index creation
- `run.py` - main pipeline
- `output/transformed_story.md` - output

---

## Known Issues

- PrecedentLock scores ~0.2-0.4 (maybe threshold too high?)
- Temperature 0.7, max tokens 1800
- Parsing uses brittle markers
