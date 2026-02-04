def build_scene_prompt(setting, rules, char_lines, tone_mode, format_instruction, banned, preferred, max_monologue_words, voice_lines, description, beat, scene_tone, required_chars, constraints, original_content, format_note):
    return f"""Transform this scene from Shakuntalam into {setting}.

WORLD CONTEXT:
{chr(10).join(f"- {r}" for r in rules)}

CHARACTERS:
{chr(10).join(char_lines)}

HOUSE STYLE (GLOBAL):
- Tone mode: {tone_mode}
- Format: {format_instruction}
- Do NOT use archaic terms: {', '.join(banned) if banned else '—'}
- Prefer vocabulary: {', '.join(preferred) if preferred else '—'}
- Cap monologues to ~{max_monologue_words if max_monologue_words else 'N/A'} words; keep dialogue concise

CHARACTER VOICES:
{chr(10).join(voice_lines) if voice_lines else '(see character definitions above)'}

SCENE: {description} (Beat: {beat}, Tone: {scene_tone})
REQUIRED CHARACTERS: {', '.join(required_chars)}

NARRATIVE CONSTRAINTS (CRITICAL - FOLLOW EXACTLY):
{chr(10).join(constraints)}

ORIGINAL SCENE:
{original_content[:2000]}...

INSTRUCTIONS:
- Follow narrative constraints exactly
- Keep emotional beat: {beat}
- Use vocabulary from WORLD CONTEXT
- Obey HOUSE STYLE strictly
- {format_note}
- Keep output concise (max 200-250 words per scene for prose)

TRANSFORMED SCENE:"""


def build_fix_prompt(tone_mode, fmt, banned, max_words, violations, draft):
    return f"""You produced a draft that violates HOUSE STYLE. Fix minimally (keep content), just adjust diction/format.

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
