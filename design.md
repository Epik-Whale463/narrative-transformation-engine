# How I Built This

## Starting Point

Tried just asking GPT-4 to rewrite Shakuntalam as legal drama. Failed hard.

First output had Shakuntala refusing to return ring before Dushyanta gave it to her. Characters appeared in wrong acts. Realized preserving narrative logic is harder than swapping words.

Inspiration: Been watching Suits. Mike Ross has no law degree (no "official ID") but Harvey recognizes him anyway. Same as Shakuntala and the ring. Wanted that dynamic.

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

| Problem | How Found | Fix |
|---------|-----------|-----|

| Ring direction reversed | Read output, saw "refuses to return" in scene 6 | Added ring_intent to DAG: "gift_not_loan" |
| Gautami in Act I | Validation flagged "too early" | Enforced character lists per node |
| Union became courtroom fight | Scene tone drifted adversarial | Added scene_tone: "private_consensual" |
| Mathavya in separation | Only Gautami should interrupt | Added check for scene_id |

DAG validation caught most errors before I read output. Rest fixed manually.

## Why Legal?

Could've done AI labs (like Romeo & Juliet example). I felt its TOO generic.

Could've done cyberpunk. Cool but "recognition token" is just another MacGuffin.

Legal system has built-in "recognition without registration": common law marriage, equitable estoppel, precedent. Ring as notary seal maps perfectly - physical token bridging informal agreement and formal validity.

## Code Notes

Kept it messy on purpose:

- TODO comments where it's brittle
- Debug prints commented out, not deleted  
- `max_tokens=1800  # bumped up - was cutting off scenes`

DAG validation is simple - checks characters and ring state. Tried adding semantic intent validation ("she_keeps_it_he_pines") but regex on natural language is fragile. Removed it.

## TODO

- Better scene parsing (markers are brittle)
- Multi-world support (worlds/ directory with JSON per world)
- Auto-retry if PrecedentLock < 0.85
- Fix union scene dialogue (still says "won't return" sometimes)

## Stats

Time: ~7 hours
API calls: 11 (one per scene)
Cost: ~$0.60
Lines: ~350 (probably less if cleaned up)
Failed attempts: 3
