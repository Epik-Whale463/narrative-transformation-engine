# Shakuntalam Legal Transformation

## I built a pipeline that generates stories grounded in a defined world and template, ensuring consistency and coherence beyond generic LLM outputs.

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

Output: `output/transformed_story.md` (prose, ~2-3 pages)

**Changing format**: Edit `data/world_rules.json` → `style.format`:
- `"prose_narrative"` - pipeline generates normal story format (current setting)
- `"screenplay"` - pipeline generates script format like the original play

Ran it both ways - prose version for submission, screenplay in `transformed_story_full.md` to show all scenes.

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

- PrecedentLock threshold set at 0.85 (still tuning - scores vary by scene)
- Scene parsing uses brittle string markers 
- Temperature 0.7, max tokens 1800

## Output

- Output: `output/transformed_story.md` (prose, ~2-3 pages)
