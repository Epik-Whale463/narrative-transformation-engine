# How I Built This

## Starting Point

Tried just asking GPT-4 to rewrite Shakuntalam as legal drama. Failed hard.

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
│          Marker-based extraction (###SCENE###)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DAG Constructor                              │
│     Topological sort ensures narrative dependencies            │
│     (e.g., ring_given → ring_kept → ring_lost)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FAISS Retriever                                │
│     Fetch relevant world rules from world_rules.json           │
│     using scene embedding similarity                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GPT-4o-mini Generator                         │
│     Transform scene with DAG constraints injected               │
│     Maintain emotional arc & narrative coherence                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PrecedentLock Validator                        │
│     Cosine similarity check (threshold: 0.75)                   │
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

Used NetworkX (used it before for AI4Bharat project). Built DAG with 11 nodes, edges for order: meeting → lineage → ring_exchange → mathavya_complaint. Topological sort fixes processing order.

Each node has metadata:

- `ring_state`: not_present → dushyanta_gives → shakuntala_has
- `chars`: who appears, who doesn't
- `scene_tone`: comic_frustration, private_consensual, etc

DAG wasn't planned from start. Came from fixing 3 failed attempts.

## Pipeline

shakuntalam.txt → parse by markers → DAG sort → FAISS retrieve → GPT-4o-mini → validate → output

**Parsing**: Use marker strings like "Enter King DUSHYANTA" or "Offers a ring". Crude but works.

**Retrieval**: Embed world_rules.json with all-MiniLM-L6-v2 into FAISS. For each scene, get top-3 relevant rules. Keeps prompt focused - Dushyanta stays Senior Associate, doesn't become judge.

**Generation**: Temperature 0.7. Inject constraints from DAG node: "This is a GIFT, not a loan" for ring_exchange, "Gautami does NOT appear" for early scenes. Max tokens 1800 (bumped up after scenes cut off).

**Validation**: Called it PrecedentLock. Cosine similarity between original and transformed (first 500 chars). Threshold 0.85. Catches drift without subjective "does this feel right" evaluation.

Why not LLM-as-judge? Learned at AI4Bharat - metrics beat opinions. Embedding similarity is reproducible, fast, cheap.

## What I Fixed

| Problem                     | How Found                                 | Fix                                         |
|----------------------------|-------------------------------------------|---------------------------------------------|
| Ring direction reversed     | Read output, saw "refuses to return" in scene 6 | Added `ring_intent` to DAG: `"gift_not_loan"` |
| Gautami in Act I            | Validation flagged "too early"            | Enforced character lists per node           |
| Union became courtroom fight| Scene tone drifted adversarial            | Added `scene_tone`: `"private_consensual"`  |
| Mathavya in separation      | Only Gautami should interrupt             | Added check for `scene_id`                  |


DAG validation caught most errors before I read output. Rest fixed manually.

## Why Legal? I'm a FANN of SUITS 😊(its a TV Show)

Could've done AI labs (like Romeo & Juliet example). I felt its TOO generic.

Could've done cyberpunk. Cool but "recognition token" is just another MacGuffin.

Legal system has built-in "recognition without registration": common law marriage, equitable estoppel, precedent. Ring as notary seal maps perfectly - physical token bridging informal agreement and formal validity.

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

