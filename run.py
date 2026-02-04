import argparse
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import networkx as nx
from dotenv import load_dotenv
from prompts import build_scene_prompt, build_fix_prompt

load_dotenv()

def load_world_rules(world_name):
    path = f'data/world_rules_{world_name}.json'
    if not os.path.exists(path):
        print(f"World '{world_name}' not found. Available: legal, bollywood")
        raise FileNotFoundError(path)
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_index(world_name):
    index_path = f"data/world_vectors_{world_name}.index"
    chunks_path = f"data/world_chunks_{world_name}.npy"
    
    if not os.path.exists(index_path):
        print(f"Index for '{world_name}' not found. Run: python scripts/build_index.py --world {world_name}")
        raise FileNotFoundError(index_path)
    
    index = faiss.read_index(index_path)
    chunks = np.load(chunks_path, allow_pickle=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, chunks, model

def load_dag_config(path="data/shakuntalam_dag.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def build_narrative_dag(dag_config):
    G = nx.DiGraph()
    
    for scene in dag_config.get("scenes", []):
        node_id = scene["id"]
        attrs = scene.get("attrs", {})
        G.add_node(node_id, **attrs)
    
    dependencies = dag_config.get("dependencies", [])
    G.add_edges_from(dependencies)
    
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Narrative graph has cycles! Check dependencies.")
    
    return G

def check_scene(scene_id, text, dag, world):
    errors = []
    node = dag.nodes[scene_id]
    text_lower = text.lower()
    aliases_map = world.get('validation', {}).get('character_aliases', {})
    
    required_chars = node['chars']
    for char in required_chars:
        aliases = aliases_map.get(char, [char.lower()])
        if not any(a.lower() in text_lower for a in aliases):
            errors.append(f"Missing character: {char}")

    forbidden_chars = node.get('forbidden_chars', [])
    for char in forbidden_chars:
        aliases = aliases_map.get(char, [char.lower()])
        if any(a.lower() in text_lower for a in aliases):
            errors.append(f"Forbidden character appeared: {char}")

    only_chars = node.get('only_chars', [])
    if only_chars:
        for char, aliases in aliases_map.items():
            if char in only_chars:
                continue
            if any(a.lower() in text_lower for a in aliases):
                errors.append(f"Unexpected character in solo scene: {char}")
    
    ring_state = node['ring_state']
    ring_words = world.get('validation', {}).get('ring_words', 
                 ["ring", "seal", "signature", "token", "engagement"])
    has_ring_mention = any(w.lower() in text_lower for w in ring_words)
    
    if ring_state == "not_present" and has_ring_mention:
        errors.append("Ring/seal too early")
    elif ring_state == "dushyanta_gives_to_shakuntala":
        if not has_ring_mention:
            errors.append("Ring exchange missing")
    
    return errors

def retrieve_rules(query, index, chunks, model, k=3):
    query_vec = model.encode([query])
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]], scores[0]

def parse_scenes_with_dag(text, dag, scene_markers):
    scenes = []
    
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
                scenes.append({
                    'id': node_id,
                    'act': f"Scene: {description}",
                    'description': description,
                    'content': f"[Scene {node_id} not found in source text]",
                    'dag_data': dag.nodes[node_id]
                })
    
    return scenes

def transform_scene(scene, rules, world, client):
    dag_data = scene['dag_data']
    ring_state = dag_data['ring_state']
    required_chars = dag_data['chars']
    beat = dag_data['beat']
    scene_tone = dag_data.get('scene_tone', '')
    ring_intent = dag_data.get('ring_intent', '')
    scene_constraints = world.get('scene_constraints', {}).get(scene['id'], {})
    char_goals = scene_constraints.get('char_goals', {})
    style = world.get('style', {})
    house = style.get('house_rules', {})
    banned = house.get('banned_terms', [])
    preferred = house.get('preferred_terms', [])
    tone_mode = style.get('tone_mode', '')
    fmt = style.get('format', '')
    voices = style.get('character_voices', {})
    
    nc = world.get('narrative_constraints', {})
    setting = world.get('setting', 'modern setting')
    
    constraints = []
    templates = world.get('constraint_templates', {})
    for key in templates.get('ring_state', {}).get(ring_state, []):
        template = nc.get(key)
        if template:
            constraints.append(f"- {template}")
    for key in templates.get('ring_intent', {}).get(ring_intent, []):
        template = nc.get(key)
        if template:
            constraints.append(f"- {template}")
    for key in templates.get('scene_tone', {}).get(scene_tone, []):
        template = nc.get(key)
        if template:
            constraints.append(f"- {template}")
    
    if char_goals:
        constraints.append(f"- Character motivations:")
        for char, goal in char_goals.items():
            constraints.append(f"  * {char}: {goal}")
    
    for extra in dag_data.get('constraints', []):
        constraints.append(f"- {extra}")
    
    if fmt.lower() in ['prose', 'prose_narrative', 'narrative']:
        format_instruction = "flowing prose narrative with embedded dialogue (no screenplay format, no stage directions)"
        format_note = "Write as continuous narrative prose. Embed key dialogue naturally with quotes."
    else:
        format_instruction = "screenplay format with INT./EXT. headers and stage directions"
        format_note = "Use dramatic script format with scene headers and character names."
    
    char_lines = []
    for char_key, char_info in world['characters'].items():
        char_lines.append(f"- {char_key} = {char_info['mapping']}")
    
    voice_lines = []
    for char_key, char_info in world['characters'].items():
        voice = voices.get(char_key, char_info.get('voice', ''))
        if voice:
            voice_lines.append(f"- {char_key}: {voice}")
    
    prompt = build_scene_prompt(
        setting=setting,
        rules=rules,
        char_lines=char_lines,
        tone_mode=tone_mode,
        format_instruction=format_instruction,
        banned=banned,
        preferred=preferred,
        max_monologue_words=house.get('max_monologue_words'),
        voice_lines=voice_lines,
        description=scene['description'],
        beat=beat,
        scene_tone=scene_tone,
        required_chars=required_chars,
        constraints=constraints,
        original_content=scene['content'],
        format_note=format_note
    )

    sys_prompt = nc.get('system_prompt', 'You are a narrative adapter. Follow constraints precisely.')
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1800
    )
    draft = response.choices[0].message.content

    violations = []
    text_lower = draft.lower()
    for term in banned:
        if term and term.lower() in text_lower:
            violations.append(f"banned term: {term}")
    max_words = house.get('max_monologue_words')
    if max_words:
        for para in draft.split("\n\n"):
            if len(para.split()) > max_words:
                violations.append("monologue too long")
                break
    if fmt == 'screenplay' and ('INT.' not in draft and 'EXT.' not in draft):
        violations.append("missing screenplay headers")

    if violations:
        fix_prompt = build_fix_prompt(
            tone_mode=tone_mode,
            fmt=fmt,
            banned=banned,
            max_words=max_words,
            violations=violations,
            draft=draft
        )
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
    emb1 = model.encode([original[:500]])
    emb2 = model.encode([transformed[:500]])
    
    similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
    return float(similarity)

def main():
    parser = argparse.ArgumentParser(description="Transform Shakuntalam into different worlds")
    parser.add_argument("--world", default="legal", choices=["legal", "bollywood"],
                        help="Target world (legal or bollywood)")
    parser.add_argument("--validate", action="store_true", help="Run PrecedentLock validation")
    args = parser.parse_args()
    
    print(f"Loading world: {args.world}")
    world = load_world_rules(args.world)
    index, chunks, model = load_index(args.world)
    
    print("Building narrative DAG...")
    dag_config = load_dag_config()
    dag = build_narrative_dag(dag_config)
    print(f"DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} dependencies")
    
    with open('data/shakuntalam.txt', 'r', encoding='utf-8') as f:
        source_text = f.read()
    
    print("Parsing scenes...")
    scene_markers = dag_config.get("scene_markers", {})
    scenes = parse_scenes_with_dag(source_text, dag, scene_markers)
    print(f"Found {len(scenes)} scenes")
    
    client = OpenAI()
    
    results = []
    for i, s in enumerate(scenes):
        print(f"\nProcessing {s['act']}...")
        
        query = f"{s['description']} {s['content'][:300]}"
        rules, scores = retrieve_rules(query, index, chunks, model)
        print(f"  Retrieved: {[r[:50] + '...' for r in rules]}")
        
        transformed = transform_scene(s, rules, world, client)
        
        dag_errors = check_scene(s['id'], transformed, dag, world)
        if dag_errors:
            print(f"  ERRORS: {dag_errors}")
        else:
            print(f"  OK")
        
        results.append({
            'original_act': s['act'],
            'scene_id': s['id'],
            'transformed': transformed,
            'dag_errors': dag_errors
        })
        
        if args.validate:
            score = validate_fidelity(s['content'], transformed, model)
            print(f"  Similarity: {score:.3f}")
            if score < 0.85:
                print(f"  (low)")
    
    print("\n" + "="*60)
    print(f"TRANSFORMED NARRATIVE ({args.world} world)")
    print("="*60)
    for ts in results:
        print(f"\n\n{ts['original_act']}\n")
        print(f"[Scene generated - see output file]")
    
    output_file = f'output/transformed_story_{args.world}.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Shakuntalam -> {args.world.title()} World\n\n")
        for ts in results:
            f.write(f"\n\n{ts['original_act']}\n")
            f.write(ts['transformed'])
    
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    main()