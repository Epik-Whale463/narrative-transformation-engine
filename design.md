# How I Built This

## Starting Point

Asked GPT-4 to rewrite Shakuntalam as legal drama. Failed hard 😂.

First output had Shakuntala refusing to return ring before Dushyanta gave it to her. Characters appeared in wrong acts. Realized preserving narrative logic is harder than swapping words.

Inspiration: Been watching Suits. Mike Ross has no law degree (no "official ID") but Harvey recognizes him anyway. Same as Shakuntala and the ring. Wanted that dynamic.

## Approach Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Shakuntalam.txt                            │
│                   (Source Epic Text)                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Scene Parser                                  │
│          Marker-based extraction (string markers)               │
│          markers live in shakuntalam_dag.json                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DAG Constructor                              │
│     Loads DAG from JSON and topologically sorts scenes          │
│     (ring order + character timing)                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FAISS Retriever                                │
│     Fetch relevant world rules from world_rules_*.json           │
│     using scene embedding similarity                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GPT-4o-mini Generator                         │
│     Transform scene with DAG + world constraints injected        │
│     Maintain emotional arc & narrative coherence                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PrecedentLock Validator                        │
│     Cosine similarity check (threshold: 0.85)                   │
│     Ensures emotional fidelity to source                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 transformed_story.md                            │
│          Final output with preserved narrative logic            │
└─────────────────────────────────────────────────────────────────┘
```

## Why DAG?

Tried 3 times with simple prompts:

1. Single prompt for Act I - lost ring exchange scene
2. Scene by scene - Shakuntala appeared too early
3. Added scene list - ring direction reversed ("refused to return" before he gave it)

Problem wasn't prompts. It was causal dependencies. Can't refuse to return something before receiving it.

Used NetworkX (used it before for AI4Bharat project). Built DAG with 11 nodes, edges for order: meeting → lineage → ring_exchange → mathavya_complaint. Topological sort fixes processing order. The DAG is now in `data/shakuntalam_dag.json` so it's not hardcoded in Python.

Each node has metadata:

- `ring_state`: not_present → dushyanta_gives → shakuntala_has
- `chars`: who appears, who doesn't
- `scene_tone`: comic_frustration, private_consensual, etc

DAG wasn't planned from start. Came from fixing 3 failed attempts.

## Pipeline

shakuntalam.txt → parse by markers → DAG sort → FAISS retrieve → GPT-4o-mini → validate → output

**Parsing**: Use marker strings like "Enter King DUSHYANTA" or "Offers a ring" from `shakuntalam_dag.json`. Crude but works.

**Retrieval**: Embed world_rules.json with all-MiniLM-L6-v2 into FAISS. For each scene, get top-3 relevant rules. Keeps prompt focused - Dushyanta stays Senior Associate, doesn't become judge.

**Generation**: Temperature 0.7. Inject constraints from DAG node + world config. Max tokens 1800 (bumped up after scenes cut off). Prompt templates are in `prompts.py`.

**Validation**: Called it PrecedentLock. Cosine similarity between original and transformed (first 500 chars). Threshold 0.85 - tried 0.7 first (too many false positives, flagged everything), tried 0.9 (missed actual drift). 0.85 felt right but honestly still tuning.

Why not LLM-as-judge? Learned at AI4Bharat - metrics beat opinions. Embedding similarity is reproducible, fast, cheap.

## House Style (data‑driven)

- world_rules_*.json defines style: tone_mode, format, banned/preferred terms, max_monologue_words, character_voices.
- `constraint_templates` connects DAG states (ring_state / ring_intent / scene_tone) to world text like "ring_give_action".
- `scene_constraints` stores per-scene motivations (world-specific).
- After generation, a quick check enforces style (no banned terms, cap monologues, ensure INT./EXT. if screenplay). If violated, do one minimal rewrite.
- Net effect: system dictates voice and format; model just writes within those rails.

**Output format**: `style.format` lets me switch between prose and screenplay. I kept it as prose to fit the 2-3 page target.

## What I Fixed

| Problem                     | How Found                                 | Fix                                         |
|----------------------------|-------------------------------------------|---------------------------------------------|
| Ring direction reversed     | Read output, saw "refuses to return" too early | Added `ring_intent` in DAG + constraint templates |
| Gautami in Act I            | Validation flagged "too early"            | Added `forbidden_chars` in DAG              |
| Union became courtroom fight| Scene tone drifted adversarial            | `scene_tone` + world tone constraints       |
| Mathavya in separation      | Only Gautami should interrupt             | `only_chars` + DAG constraints              |

DAG validation caught most errors before I read output. Rest fixed manually.

## Why Legal? I'm a FANN of SUITS 😊(its a TV Show)

Could've done AI labs (like Romeo & Juliet example). I felt its TOO generic.

Could've done cyberpunk. Cool but "recognition token" is just another MacGuffin.

Legal system has built-in "recognition without registration": common law marriage, equitable estoppel, precedent. Ring as notary seal maps perfectly - physical token bridging informal agreement and formal validity.

## TODO / Future Work

- If I had time: per-world DAG overrides so Legal/Bollywood can diverge more
- Scene parsing uses string markers which is brittle
- Maybe add --debug flag to dump retrieved rules and validation scores to JSON
- Test with Acts IV-VII (only did I-III for now)

## Alternatives Considered

| Approach                          | Why It Failed                                      |
|-----------------------------------|----------------------------------------------------|
| Single prompt for full text       | Lost scene-level detail, no validation hooks        |
| No DAG, just list of scenes       | Chronology broke repeatedly                         |
| LLM-as-judge for validation       | Slow, expensive, subjective                         |
| Few-shot with 3 examples          | Worked per-scene, failed across transitions         |
| AI labs world (like assignment example) | Too generic, matches their Romeo & Juliet demo too closely |

## One Clever Idea: PrecedentLock

```
Most narrative transformation uses LLM-as-judge: "Does this capture the original?" 
Slow, expensive, subjective.

I repurposed **embedding similarity** as objective validation. First 500 chars of 
scene establish emotional valence. If transformed embeds similarly, "ratio 
decidendi" (reasoning) is preserved. Fast, cheap, reproducible.

Meta-layer: Legal metaphor extends to validation itself — "stare decisis" 
(consistency with precedent).
```