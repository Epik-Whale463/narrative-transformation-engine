import argparse
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

# testing - uncomment if needed
# print("DEBUG: API key loaded")

def load_world_rules():
    with open('data/world_rules.json', encoding='utf-8') as f:
        return json.load(f)

def load_index():
    index = faiss.read_index("data/world_vectors.index")
    chunks = np.load("data/world_chunks.npy", allow_pickle=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, chunks, model

def build_narrative_dag():
    # DAG for scene ordering and validation
    G = nx.DiGraph()
    
    # scene nodes with metadata
    scenes = [
        ("hunt", {
            "chars": ["Dushyanta"], 
            "objects": [], 
            "beat": "inciting_incident", 
            "ring_state": "not_present",
            "scene_tone": "energetic_pursuit"
        }),
        ("enter_grove", {
            "chars": ["Dushyanta"], 
            "objects": ["bark_dress"], 
            "beat": "world_discovery", 
            "ring_state": "not_present",
            "scene_tone": "discovery"
        }),
        ("meeting", {
            "chars": ["Dushyanta", "Shakuntala"], 
            "objects": ["water_jar"], 
            "beat": "meet_cute", 
            "ring_state": "not_present",
            "scene_tone": "romantic_first_encounter"
        }),
        ("lineage", {
            "chars": ["Anasuya", "Dushyanta", "Shakuntala"], 
            "objects": [], 
            "beat": "eligibility_confirmed", 
            "ring_state": "not_present",
            "scene_tone": "investigative_hopeful"
        }),
        ("ring_exchange", {
            "chars": ["Dushyanta", "Shakuntala"], 
            "objects": ["ring"], 
            "beat": "token_given", 
            "ring_state": "dushyanta_gives_to_shakuntala",
            "ring_intent": "gift_not_loan",
            "scene_tone": "romantic_trust",
            "char_goals": {
                "Dushyanta": "give seal as proof of commitment, enable her future claim",
                "Shakuntala": "accept token as promise of recognition"
            }
        }),
        ("mathavya_complaint", {
            "chars": ["Mathavya", "Dushyanta"], 
            "objects": [], 
            "beat": "comic_relief", 
            "ring_state": "shakuntala_has_ring",
            "ring_intent": "she_keeps_it_he_pines",
            "scene_tone": "comic_frustration",
            "char_goals": {
                "Mathavya": "complain about Dushyanta's distraction and workload",
                "Dushyanta": "distracted, thinking of Shakuntala, not wanting seal back"
            }
        }),
        ("demon_defense", {
            "chars": ["Dushyanta", "Hermits"], 
            "objects": [], 
            "beat": "excuse_to_stay", 
            "ring_state": "shakuntala_has_ring",
            "scene_tone": "duty_vs_desire"
        }),
        ("lovesick", {
            "chars": ["Dushyanta"], 
            "objects": ["lotus_leaf"], 
            "beat": "yearning", 
            "ring_state": "shakuntala_has_ring",
            "ring_intent": "she_has_it_he_waits_for_her",
            "scene_tone": "melancholic_longing",
            "char_goals": {
                "Dushyanta": "yearns for Shakuntala to arrive/contact, NOT for seal back"
            }
        }),
        ("letter", {
            "chars": ["Shakuntala", "Dushyanta"], 
            "objects": ["lotus_leaf"], 
            "beat": "communication", 
            "ring_state": "shakuntala_has_ring",
            "scene_tone": "tender_communication"
        }),
        ("union", {
            "chars": ["Shakuntala", "Dushyanta"], 
            "objects": ["ring"], 
            "beat": "climax_union", 
            "ring_state": "shakuntala_has_ring",
            "ring_intent": "seal_validates_mutual_agreement",
            "scene_tone": "private_consensual",
            "char_goals": {
                "Dushyanta": "establish commitment despite bureaucratic system",
                "Shakuntala": "accept union as valid even without formal registration"
            }
        }),
        ("separation", {
            "chars": ["Shakuntala", "Dushyanta", "Gautami"], 
            "objects": ["ring"], 
            "beat": "interrupted_departure", 
            "ring_state": "shakuntala_keeps_ring",
            "scene_tone": "bittersweet_interruption"
        })
    ]
    
    for node_id, attrs in scenes:
        G.add_node(node_id, **attrs)
    
    # Edges: strict causal dependencies
    dependencies = [
        ("hunt", "enter_grove"),
        ("enter_grove", "meeting"),
        ("meeting", "lineage"),
        ("lineage", "ring_exchange"),  # CRITICAL: Must meet before ring
        ("ring_exchange", "mathavya_complaint"),  # Ring given, now complaints
        ("mathavya_complaint", "demon_defense"),
        ("demon_defense", "lovesick"),
        ("lovesick", "letter"),
        ("letter", "union"),  # Letter leads to union
        ("union", "separation")  # Union interrupted by departure
    ]
    
    G.add_edges_from(dependencies)
    
    # Validate DAG
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Narrative graph has cycles! Check dependencies.")
    
    return G

def check_scene(scene_id, text, dag):
    # just checking structural stuff
    errors = []
    node = dag.nodes[scene_id]
    text_lower = text.lower()
    
    # check characters are there
    required_chars = node['chars']
    for char in required_chars:
        # Handle name variations
        if char == "Shakuntala":
            if "shakuntala" not in text_lower and "shakoontala" not in text_lower:
                errors.append(f"Missing character: {char}")
        elif char.lower() not in text_lower:
            errors.append(f"Missing character: {char}")
    
    # ring/seal checking
    ring_state = node['ring_state']
    has_ring_mention = "ring" in text_lower or "seal" in text_lower or "signature" in text_lower
    
    if ring_state == "not_present" and has_ring_mention:
        errors.append("Ring/seal too early")
    elif ring_state == "dushyanta_gives_to_shakuntala":
        if not has_ring_mention:
            errors.append("Ring exchange missing")
    
    # gautami should only be in last scene
    if scene_id != "separation" and "gautami" in text_lower:
        errors.append("Gautami too early")
    
    # mathavya scene is solo
    if scene_id == "mathavya_complaint" and ("shakuntala" in text_lower or "shakoontala" in text_lower):
        errors.append("Shakuntala shouldn't appear in Mathavya complaint scene")
    
    return errors

def retrieve_rules(query, index, chunks, model, k=3):
    query_vec = model.encode([query])
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]], scores[0]

def parse_scenes_with_dag(text, dag):
    # map scene IDs to where they appear in source text
    scene_markers = {
        "hunt": ("Enter King DUSHYANTA", "Hunt scene"),
        "meeting": ("Enter [S']AKOONTALÁ", "Meeting/watering plants"),
        "lineage": ("ANASÚYÁ. You shall hear it", "Lineage reveal"),
        "ring_exchange": ("Offers a ring", "Ring exchange - CRITICAL"),
        "mathavya_complaint": ("Enter the Jester", "Mathavya complains"),
        "demon_defense": ("evil demons", "Demon defense excuse"),
        "lovesick": ("Enter KING DUSHYANTA, with the air of one in love", "Lovesickness"),
        "letter": ("lotus-leaf", "Love letter writing"),
        "union": ("Gandharva", "Union scene"),
        "separation": ("GAUTAMÍ", "Separation")
    }
    
    scenes = []
    
    # TODO: handle missing scenes better
    for node_id in nx.topological_sort(dag):
        if node_id in scene_markers:
            marker, description = scene_markers[node_id]
            idx = text.find(marker)
            if idx != -1:
                content = text[idx:idx+2500]
                scenes.append({
                    'id': node_id,
                    'act': f"Scene: {description}",
                    'description': description,
                    'content': content,
                    'dag_data': dag.nodes[node_id]
                })
            else:
                # Scene not found in text, create placeholder
                scenes.append({
                    'id': node_id,
                    'act': f"Scene: {description}",
                    'description': description,
                    'content': f"[Scene {node_id} not found in source text]",
                    'dag_data': dag.nodes[node_id]
                })
    
    return scenes

def transform_scene(scene, rules, world, client):
    # build prompt with constraints from DAG
    dag_data = scene['dag_data']
    ring_state = dag_data['ring_state']
    required_chars = dag_data['chars']
    beat = dag_data['beat']
    scene_tone = dag_data.get('scene_tone', '')
    ring_intent = dag_data.get('ring_intent', '')
    char_goals = dag_data.get('char_goals', {})
    style = world.get('style', {})
    house = style.get('house_rules', {})
    banned = house.get('banned_terms', [])
    preferred = house.get('preferred_terms', [])
    tone_mode = style.get('tone_mode', '')
    fmt = style.get('format', '')
    voices = style.get('character_voices', {})
    
    # constraints list
    constraints = []
    if ring_state == "dushyanta_gives_to_shakuntala":
        constraints.append("- Dushyanta MUST give the notary seal/digital signature to Shakuntala")
        constraints.append("- This is a GIFT, not a loan - he doesn't expect it back")  # important!
    elif ring_state == "shakuntala_has_ring":
        constraints.append("- Shakuntala HAS the notary seal (from previous scene)")
        constraints.append("- Shakuntala does NOT give the seal back to Dushyanta")
        if ring_intent == "she_keeps_it_he_pines":
            constraints.append("- Dushyanta is distracted/pining for SHAKUNTALA HERSELF, not for the seal back")
            constraints.append("- NO dialogue about 'refusing to return' - he never asked for it")
        elif ring_intent == "she_has_it_he_waits_for_her":
            constraints.append("- Dushyanta yearns for SHAKUNTALA TO ARRIVE/CONTACT, not for seal return")
            constraints.append("- The seal is WITH HER as proof she will come")
        elif ring_intent == "seal_validates_mutual_agreement":
            constraints.append("- The seal validates their MUTUAL AGREEMENT (common law marriage)")
            constraints.append("- NO conflict about returning seal - this is consensual union")
    elif ring_state == "not_present":
        constraints.append("- NO mention of ring/seal/signature (too early in story)")
    
    # tone stuff
    if scene_tone == "private_consensual":
        constraints.append("- PRIVATE office setting, NOT courtroom")
        constraints.append("- CONSENSUAL agreement, both parties willing, no adversarial conflict")
    elif scene_tone == "comic_frustration":
        constraints.append("- Comic tone: Mathavya complains about workload")
    
    # character motivations if needed
    if char_goals:
        constraints.append(f"- Character motivations:")
        for char, goal in char_goals.items():
            constraints.append(f"  * {char}: {goal}")
    
    if scene['id'] == "mathavya_complaint":
        constraints.append("- ONLY Mathavya and Dushyanta appear (Shakuntala not present)")
    elif scene['id'] == "separation":
        constraints.append("- Gautami INTERRUPTS the scene (enters suddenly)")
    elif "gautami" not in [c.lower() for c in required_chars]:
        constraints.append("- Gautami does NOT appear (too early)")
    
    # Build prompt
    prompt = f"""Transform this scene from Shakuntalam into legal setting (2024).

WORLD CONTEXT:
{chr(10).join(f"- {r}" for r in rules)}

CHARACTERS:
- Dushyanta = {world['characters']['DUSHYANTA']['mapping']}
- Shakuntala = {world['characters']['SHAKOONTALA']['mapping']}
- Mathavya = {world['characters']['MATHAVYA']['mapping']}

HOUSE STYLE (GLOBAL):
- Tone mode: {tone_mode}
- Format: {fmt} (use INT./EXT. headers if screenplay)
- Do NOT use archaic terms: {', '.join(banned) if banned else '—'}
- Prefer modern legal vocabulary: {', '.join(preferred) if preferred else '—'}
- Cap monologues to ~{house.get('max_monologue_words', 'N/A')} words; alternate dialogue lines

CHARACTER VOICES:
- Dushyanta: {voices.get('DUSHYANTA', world['characters']['DUSHYANTA'].get('voice',''))}
- Shakuntala: {voices.get('SHAKOONTALA', world['characters']['SHAKOONTALA'].get('voice',''))}
- Mathavya: {voices.get('MATHAVYA', world['characters']['MATHAVYA'].get('voice',''))}

SCENE: {scene['description']} (Beat: {beat}, Tone: {scene_tone})
REQUIRED CHARACTERS: {', '.join(required_chars)}

NARRATIVE CONSTRAINTS (CRITICAL - FOLLOW EXACTLY):
{chr(10).join(constraints)}

ORIGINAL SCENE:
{scene['content'][:2000]}...

INSTRUCTIONS:
- Follow narrative constraints exactly
- Keep emotional beat: {beat}
- Use legal terminology from world context
- Obey HOUSE STYLE strictly
- Dramatic script format (consistent with {fmt})
- Preserve dialogue structure

TRANSFORMED SCENE:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a legal drama adapter with strict narrative logic. Follow the constraints precisely to maintain story coherence. The ring/seal is a GIFT given once, not repeatedly exchanged."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,  # TODO: try 0.6 if still getting weird outputs
        max_tokens=1800  # bumped up - was cutting off scenes
    )
    draft = response.choices[0].message.content

    # simple style enforcement pass
    violations = []
    text_lower = draft.lower()
    for term in banned:
        if term and term.lower() in text_lower:
            violations.append(f"banned term: {term}")
    max_words = house.get('max_monologue_words')
    if max_words:
        # rough check: any paragraph over max_words
        for para in draft.split("\n\n"):
            if len(para.split()) > max_words:
                violations.append("monologue too long")
                break
    # format nudge: ensure screenplay INT./EXT. if chosen
    if fmt == 'screenplay' and ('INT.' not in draft and 'EXT.' not in draft):
        violations.append("missing screenplay headers")

    if violations:
        fix_prompt = f"""You produced a draft that violates HOUSE STYLE. Fix minimally (keep content), just adjust diction/format.

HOUSE STYLE:
- Tone: {tone_mode}
- Format: {fmt}
- Do NOT use: {', '.join(banned)}
- Cap monologues to ~{max_words} words

Violations:
- {chr(10).join(violations)}

Draft:
{draft}

Rewrite the scene to comply. Keep dialogue beats and constraints intact."""
        resp2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an editor enforcing house style and narrative constraints with minimal changes."},
                {"role": "user", "content": fix_prompt}
            ],
            temperature=0.3,
            max_tokens=1800
        )
        return resp2.choices[0].message.content
    
    return draft

def validate_fidelity(original, transformed, model):
    # PrecedentLock - cosine similarity check
    emb1 = model.encode([original[:500]])
    emb2 = model.encode([transformed[:500]])
    
    # Cosine similarity
    similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
    return float(similarity)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="shakuntalam", help="Source text")
    parser.add_argument("--target", default="legal", help="Target world")
    parser.add_argument("--validate", action="store_true", help="Run PrecedentLock validation")
    args = parser.parse_args()
    
    # load stuff
    print("Loading world rules and index...")
    world = load_world_rules()
    index, chunks, model = load_index()
    
    # build DAG
    print("Building narrative DAG...")
    dag = build_narrative_dag()
    print(f"DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} dependencies")
    # print("DEBUG:", list(nx.topological_sort(dag)))  # uncomment to see order
    
    # Read source
    with open('data/shakuntalam.txt', 'r', encoding='utf-8') as f:
        source_text = f.read()
    
    # parse scenes
    print("Parsing scenes...")
    scenes = parse_scenes_with_dag(source_text, dag)
    print(f"Found {len(scenes)} scenes")
    # for s in scenes: print(s['id'])  # debug
    
    # Initialize OpenAI client
    client = OpenAI()  # assumes OPENAI_API_KEY in env
    
    # process scenes
    results = []
    for i, s in enumerate(scenes):
        print(f"\nProcessing {s['act']}...")
        
        # get relevant rules from index
        query = f"{s['description']} {s['content'][:300]}"
        rules, scores = retrieve_rules(query, index, chunks, model)
        print(f"  Retrieved: {[r[:50] + '...' for r in rules]}")
        
        # transform
        transformed = transform_scene(s, rules, world, client)
        
        # validate
        dag_errors = check_scene(s['id'], transformed, dag)
        if dag_errors:
            print(f"  ERRORS: {dag_errors}")
        else:
            print(f"  ✓ OK")
        
        results.append({
            'original_act': s['act'],
            'scene_id': s['id'],
            'transformed': transformed,
            'dag_errors': dag_errors
        })
        
        # similarity check
        if args.validate:
            score = validate_fidelity(s['content'], transformed, model)
            print(f"  Similarity: {score:.3f}")
            if score < 0.85:
                print(f"  (low)")
    
    # output
    print("\n" + "="*60)
    print("TRANSFORMED NARRATIVE")
    print("="*60)
    for ts in results:
        print(f"\n\n{ts['original_act']}\n")
        print(ts['transformed'])
    
    # save
    with open('output/transformed_story.md', 'w', encoding='utf-8') as f:
        for ts in results:
            f.write(f"\n\n{ts['original_act']}\n")
            f.write(ts['transformed'])
    
    print(f"\nSaved to output/transformed_story.md")

if __name__ == "__main__":
    main()