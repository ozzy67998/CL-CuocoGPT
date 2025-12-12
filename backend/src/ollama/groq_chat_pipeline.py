import os
import sys
import json
import textwrap
from typing import Any, Dict
from contextlib import redirect_stdout, redirect_stderr
import importlib

# -------------------------
# Load groq.env optionally (same folder)
# -------------------------
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), "groq.env")
    if os.path.exists(dotenv_path):
        print(f"Loading environment variables from {dotenv_path}")
        load_dotenv(dotenv_path)
except Exception:
    pass

# -------------------------
# Pipeline import (lazy)
# -------------------------
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

mod = importlib.import_module("NLP.pipeline")

_handle_query = None
_extract_slots_only = None

def _load_pipeline():
    global _handle_query, _extract_slots_only
    if _handle_query is None or _extract_slots_only is None:
        print("Loading pipeline components...")
        try:
            mod = importlib.import_module("NLP.pipeline")
            _handle_query = getattr(mod, "handle_query")
            _extract_slots_only = getattr(mod, "extract_slots_only")
            print("Pipeline components loaded successfully")
        except Exception as e:
            print(f"Failed to import pipeline: {e}")
            sys.exit(1)
    return _handle_query, _extract_slots_only

from RAG.description_index import similarity_search


# -------------------------
# Config
# -------------------------
_raw_debug = os.environ.get("GROQ_DEBUG", "0").strip().strip('"').strip("'")
try:
    DEBUG = bool(int(_raw_debug))
except ValueError:
    DEBUG = _raw_debug.lower() in {"true", "yes", "on", "1"}

GROQ_API_KEY = (os.environ.get("groq_key") or os.environ.get("GROQ_KEY") or "").strip().strip('"').strip("'")
DEFAULT_MODEL = (os.environ.get("groq_model") or os.environ.get("GROQ_MODEL") or "openai/gpt-oss-120b").strip().strip('"').strip("'")
_TEMP = os.environ.get("temperature") or os.environ.get("GROQ_TEMPERATURE")
_MAX_TOK = os.environ.get("max_tokens") or os.environ.get("MAX_TOKENS")
_TOP_P = os.environ.get("top_p") or os.environ.get("GROQ_TOP_P")
_REASONING = os.environ.get("reasoning_effort") or os.environ.get("GROQ_REASONING_EFFORT")
_STREAM_FLAG = os.environ.get("stream") or os.environ.get("GROQ_STREAM")
_STOP = os.environ.get("stop") or os.environ.get("GROQ_STOP")

try:
    _TEMP = float(_TEMP) if _TEMP is not None else None
except Exception:
    _TEMP = None
try:
    _MAX_TOK = int(_MAX_TOK) if _MAX_TOK is not None else None
except Exception:
    _MAX_TOK = None
try:
    _TOP_P = float(_TOP_P) if _TOP_P is not None else None
except Exception:
    _TOP_P = None

_STREAM = False
if isinstance(_STREAM_FLAG, str):
    _STREAM = _STREAM_FLAG.lower() in {"true", "1", "yes", "on"}

# -------------------------
# Groq client (lazy)
# -------------------------
_groq_client = None
_current_model = None

# NEW: store full conversation (turns) for context injection
_conversation_history: list[dict[str, str]] = []  # {"query":..., "structured":..., "response":..., "prompt":...}
CONTEXT_WINDOW = 3

def _extract_query_metadata(block: str) -> str:
    """
    Keep only the high-level structured metadata lines from a structured block:
    original_query=, detected_language=, translated_query=, intent=, slot_*
    Stop before any recipe group listings or steps.
    """
    lines = []
    for line in (block or "").splitlines():
        if line.startswith(("original_query=",
                            "detected_language=",
                            "translated_query=",
                            "intent=",
                            "slot_")):
            lines.append(line)
            continue
        # stop when detailed recipe grouping starts
        if line.startswith(("recipes with ", "Recipe:", "Steps:", "Answer:")):
            break
    return "\n".join(lines)

def _build_context_annotation(prev_turns: list[dict[str, str]]) -> str:
    """
    Previous messages (up to CONTEXT_WINDOW):
    Message 1 = most recent, increasing numbers go further back.
    Each shows: User query, metadata-only Structured recipe data, Assistant Response.
    """
    if not prev_turns:
        return ""
    subset = prev_turns[-CONTEXT_WINDOW:][::-1]  # reverse so newest first
    out = ["Conversation context (previous messages):"]
    for i, t in enumerate(subset, start=1):
        user_q = (t.get("query") or "").strip()
        structured_full = (t.get("structured") or "")
        meta_only = _extract_query_metadata(structured_full).strip()
        resp = (t.get("response") or "").strip()
        out.append(f"Message {i} - User query:\n{user_q}")
        out.append(f"Message {i} - Structured recipe data:\n{meta_only}")
        out.append(f"\nMessage {i} - Assistant Response:\n{resp}")
    return "\n\n".join(out)

def _write_context_file(current_full_prompt: str, structured_block: str):
    """
    Write current full prompt FIRST, then previous messages.
    Order:
    current message:
    <full current prompt>

    Conversation context (previous messages):
    ...
    """
    try:
        prev = _build_context_annotation(_conversation_history[:-1])
        # Remove any duplicated annotation prefix from current_full_prompt
        if prev and current_full_prompt.startswith(prev):
            current_full_prompt = current_full_prompt[len(prev):].lstrip("\n")
        context_path = os.path.join(os.path.dirname(__file__), "context.txt")
        with open(context_path, "w", encoding="utf-8") as f:
            f.write("current message:\n")
            f.write(current_full_prompt.rstrip() + "\n\n")
            if prev:
                f.write(prev.rstrip() + "\n")
    except Exception as e:
        if DEBUG:
            print(f"[context write error] {e}")

def _ensure_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            print("Missing GROQ_API_KEY (groq_key) in environment.")
            return None
        try:
            from groq import Groq
            _groq_client = Groq(api_key=GROQ_API_KEY)
        except ModuleNotFoundError:
            print("Groq package not installed. Run: pip install groq")
            return None
        except Exception as e:
            print(f"Failed to init Groq client: {e}")
            return None
    return _groq_client

def _list_models():
    client = _ensure_client()
    if not client:
        return []
    try:
        data = client.models.list()
        return [m.id for m in getattr(data, "data", [])]
    except Exception as e:
        print(f"[Model list error] {e}")
        return []

# -------------------------
# LLM API wrapper
# -------------------------
def api_chat(prompt: str, model_name: str) -> str:
    client = _ensure_client()
    if client is None:
        return ""
    params: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if _TEMP is not None:
        params["temperature"] = _TEMP
    if _MAX_TOK is not None:
        # map to correct field name
        params["max_tokens"] = _MAX_TOK
    if _TOP_P is not None:
        params["top_p"] = _TOP_P
    if _STOP and _STOP.lower() != "none":
        params["stop"] = _STOP

    try:
        completion = client.chat.completions.create(**params)
        choices = getattr(completion, "choices", [])
        if not choices:
            return ""
        return (choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[LLM EXCEPTION] {e}")
        return ""

_current_model = None

def run_groq(user_query: str, base_prompt: str, structured_block: str) -> tuple[str, str]:
    """
    Send the prompt to Groq, including a compact summary of the last 3 turns.
    Records the turn and writes context.txt.
    """
    global _current_model, _conversation_history
    if _current_model is None:
        _current_model = DEFAULT_MODEL

    # Inject previous turn summaries
    prev_context = _build_context_annotation(_conversation_history)
    current_full_prompt = (prev_context + "\n\n" + base_prompt) if prev_context else base_prompt

    resp = api_chat(current_full_prompt, _current_model)

    _conversation_history.append({
        "query": user_query,
        "structured": structured_block,
        "response": resp,
        "prompt": current_full_prompt
    })
    _write_context_file(current_full_prompt, structured_block)
    return resp, current_full_prompt

def _brief_result_for_llm(result: Dict[str, Any], user_query: str) -> str:
    """
    Build structured block for the LLM.
    Now groups recipes under per-slot headers instead of listing slot headers at the top.
    Grouping heuristic:
      - ingredient slots: recipe appears under an ingredient if that ingredient string is contained
        (case-insensitive) in any of its ingredient entries.
      - tag slots: recipe appears under a tag if tag (case-insensitive) matches any recipe tag.
      - recipe_name slots: recipe appears if its name contains that string.
      - cooking_time slot: recipes with minutes <= cooking_time.
    A recipe will only be printed once per group type; duplicates across different groups are allowed.
    """
    lines = []
    orig = result.get("original_query") or user_query
    slots = result.get("slots") or {}
    detected_lang = slots.get("detected_language") or result.get("detected_language")
    translated = slots.get("translated_query") or result.get("translated_query")
    intent = result.get("intent")
    intent_conf = result.get("intent_confidence") or result.get("intent_score") \
                  or (result.get("intent_scores") or [None])[0]

    lines.append(f"original_query={orig}")
    if detected_lang:
        lines.append(f"detected_language={detected_lang}")
    if translated and translated != orig:
        lines.append(f"translated_query={translated}")
    if intent:
        if isinstance(intent_conf, (int, float)):
            lines.append(f"intent={intent} confidence={intent_conf:.2f}")
        else:
            lines.append(f"intent={intent}")

    # Slot summaries (keep original compact representation)
    for k, v in slots.items():
        if k in {"original_query","translated_query","detected_language"}:
            continue
        if k == "cooking_time" and isinstance(v, int):
            lines.append(f"slot_cooking_time={v}m")
            continue
        if isinstance(v, list):
            rendered = []
            for item in v:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], (int, float)):
                    rendered.append(f"{item[0]}({item[1]:.2f})")
                else:
                    rendered.append(str(item))
            lines.append(f"slot_{k}=" + ", ".join(rendered))
        else:
            lines.append(f"slot_{k}={v}")

    recipes = result.get("kg_results") or []

    def _print_recipe(r):
        name = r.get("recipe_name") or r.get("recipe_uri")
        minutes = r.get("minutes")
        tags = r.get("tags") or []
        ing = r.get("ingredients") or []
        lines.append(f"Recipe: {name}")
        if minutes is not None:
            lines.append(f"Minutes: {minutes}")
        if tags:
            lines.append(f"Tags: {', '.join(tags[:12])}")
        if ing:
            lines.append("Ingredients: " + ", ".join(ing[:15]))
        steps = r.get("steps") or []
        if steps:
            lines.append("Steps:")
            for i, s in enumerate(steps[:8], start=1):
                lines.append(f"{i:02d}. {s}")

    # Group by ingredient slot
    ing_list = slots.get("ingredient") or []
    if ing_list:
        lines.append("")  # spacer
        for ing_slot in ing_list:
            ing_name = ing_slot[0] if isinstance(ing_slot, (list, tuple)) else str(ing_slot)
            group = [r for r in recipes if any(ing_name.lower() in (ri.lower()) for ri in (r.get("ingredients") or []))]
            if group:
                lines.append(f"recipes with ingredient: {ing_name}")
                for r in group:
                    _print_recipe(r)
                lines.append("")  # spacer after group

    # Group by tag slot
    tag_list = slots.get("tag") or []
    if tag_list:
        for tag_slot in tag_list:
            tag_name = tag_slot[0] if isinstance(tag_slot, (list, tuple)) else str(tag_slot)
            group = [r for r in recipes if any(tag_name.lower() == t.lower() for t in (r.get("tags") or []))]
            if group:
                lines.append(f"recipes with tag: {tag_name}")
                for r in group:
                    _print_recipe(r)
                lines.append("")

    # Group by recipe_name slot
    name_list = slots.get("recipe_name") or []
    if name_list:
        for name_slot in name_list:
            nm = name_slot[0] if isinstance(name_slot, (list, tuple)) else str(name_slot)
            group = [r for r in recipes if nm.lower() in (r.get("recipe_name","").lower())]
            if group:
                lines.append(f"recipes with name: {nm}")
                for r in group:
                    _print_recipe(r)
                lines.append("")

    # Group by cooking_time slot
    ct = slots.get("cooking_time")
    if isinstance(ct, int):
        group = [r for r in recipes if isinstance(r.get("minutes"), int) and r["minutes"] <= ct]
        if group:
            lines.append(f"recipes with cooking_time: <= {ct} minutes")
            for r in group:
                _print_recipe(r)
            lines.append("")

    # Fallback: if no groups matched, list all recipes once
    if not ing_list and not tag_list and not name_list and not isinstance(ct, int):
        if recipes:
            lines.append("")
            for r in recipes:
                _print_recipe(r)

    return "\n".join(lines)


def _summarize_with_llm(user_query: str, result: Dict[str, Any]) -> tuple[str, str]:
    brief = _brief_result_for_llm(result, user_query)
    sim_snips = (result.get("similar_description_chunks") or [])[:3]
    desc_lines = ""
    if sim_snips:
        lines = []
        for i, c in enumerate(sim_snips, start=1):
            txt = (c.get("text") or "").strip()
            score = c.get("score")
            score_str = f"{float(score):.3f}" if isinstance(score,(int,float)) else "NA"
            lines.append(f"{i}. ({score_str}) {txt}")
        desc_lines = "Description snippets (semantic matches):\n" + "\n".join(lines) + "\n\n"

    base_prompt = (
        "You are a bilingual PT/EN assistant.\n"
        "Available context:\n"
        "- Current user query.\n"
        "- Up to the last 3 messages: prior prompts, structured SPARQL summaries, and previous assistant responses (use only if clearly relevant).\n"
        "- Description snippets (may contain noise; ignore generic or off‑topic content such as posting notes or dates).\n"
        "- Structured recipe data extracted from the KG.\n\n"
        "Decision policy:\n"
        "1) If the user asks about previous messages (e.g., “o que perguntei na última query”, “what did I ask last time?”), answer ONLY using the Conversation context block. "
        "If that info is not available, say you cannot access it and provide ONE example query the user can try.\n"
        "2) If the query is off‑topic, unclear, or nonsense, do NOT produce recipes. Ask for clarification briefly OR provide ONE example query to trigger the correct intent/SPARQL.\n"
        "3) Otherwise, answer using the structured recipe data; use description snippets cautiously and only when clearly relevant. Prefer structured data over snippets.\n\n"
        "Formatting and coverage:\n"
        "- Start with 1–2 concise sentences (PT first, then EN).\n"
        "- Be modular: if there are multiple detected ingredients/tags/names (e.g., melon; bitter melon; melon liqueur), create a separate section for EACH detected item and list its corresponding recipes under that section. Do NOT merge everything into one group.\n"
        "- In each section, list ALL recipes returned for that item (name required; include time and key tags if helpful).\n"
        "- Tables are optional: use a table only if it improves clarity; otherwise bullets are fine.\n"
        "- If the requested info is missing or insufficient, say so and provide ONE example user query, e.g.:\n"
        "  • receitas com [ingrediente]\n"
        "  • receitas com tag [tag]\n"
        "  • receitas em até [min] minutos\n"
        "  • procurar receita [nome]\n"
        "- Minimize invention: do not fabricate ingredients, steps, times, tags, or external sources.\n\n"
        f"User query: {user_query}\n\n"
        f"{desc_lines}"
        "Structured recipe data:\n"
        f"{brief}\n\n"
        "Answer:"
    )

    response, full_prompt_sent = run_groq(user_query, base_prompt, brief)
    return response, full_prompt_sent

def api_chat(prompt: str, model_name: str) -> str:
    """
    Modified: remove internal prompt printing to avoid duplication.
    Outer layer (_process_query) handles printing.
    """
    client = _ensure_client()
    if client is None:
        return ""
    params: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if _TEMP is not None:
        params["temperature"] = _TEMP
    if _MAX_TOK is not None:
        params["max_tokens"] = _MAX_TOK
    if _TOP_P is not None:
        params["top_p"] = _TOP_P
    if _STOP and _STOP.lower() != "none":
        params["stop"] = _STOP
    try:
        completion = client.chat.completions.create(**params)
        choices = getattr(completion, "choices", [])
        if not choices:
            return ""
        return (choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[LLM EXCEPTION] {e}")
        return ""


def _process_query(user_query: str):
    user_query = user_query.strip()
    if not user_query:
        return
    handle_query_fn, _ = _load_pipeline()
    result = handle_query_fn(user_query, intent_top_k=1, sparql_top_k=5)

    # Print query info (restored)
    intent = result.get("intent") or "(unknown)"
    intent_conf = result.get("intent_confidence") or result.get("intent_score") \
                  or (result.get("intent_scores") or [None])[0]
    slots = result.get("slots") or {}
    print("\n========== QUERY INFO ==========")
    if isinstance(intent_conf, (int,float)):
        print(f"Intent: {intent} (confidence={intent_conf:.2f})")
    else:
        print(f"Intent: {intent}")
    print(f"Query: {user_query}")

    if slots:
        print("\n=== Normalization & Translation ===")
        orig = slots.get("original_query") or user_query
        print(f"Original: {orig}")
        if "detected_language" in slots:
            print(f"Detected language: {slots['detected_language']}")
        if "translated_query" in slots and slots["translated_query"] != orig:
            print(f"Translated (for intent/NER): {slots['translated_query']}")

        print("\nDetected slots:")
        for k, v in slots.items():
            if k in {"original_query","translated_query","detected_language"}:
                continue
            if k == "cooking_time" and isinstance(v, int):
                print(f"  - cooking_time: {v} minutes")
                continue
            if isinstance(v, list):
                rendered = []
                for item in v:
                    if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], (int,float)):
                        rendered.append(f"{item[0]} ({item[1]:.2f})")
                    else:
                        rendered.append(str(item))
                print(f"  - {k}: {', '.join(rendered)}")
            else:
                print(f"  - {k}: {v}")

    print("\n========== RESULTS (Sequential Execution) ==========\n(Below, each SPARQL query is executed and printed one by one.)\n")

    # Similarity query augmentation
    slots = result.get("slots") or {}
    slotq = []
    ing = [s[0] if isinstance(s,(list,tuple)) else str(s) for s in (slots.get("ingredient") or [])][:3]
    tg  = [s[0] if isinstance(s,(list,tuple)) else str(s) for s in (slots.get("tag") or [])][:3]
    nm  = [s[0] if isinstance(s,(list,tuple)) else str(s) for s in (slots.get("recipe_name") or [])][:2]
    if ing: slotq.append("ingredients: " + ", ".join(ing))
    if tg:  slotq.append("tags: " + ", ".join(tg))
    if nm:  slotq.append("names: " + ", ".join(nm))
    sim_query = f"{user_query} | " + " | ".join(slotq) if slotq else user_query

    try:
        from RAG.description_index import similarity_search
        # Request only top-3; retrieval already dedups by chunk_id
        result["similar_description_chunks"] = similarity_search(sim_query, top_k=3) or []
    except Exception as e:
        result["similar_description_chunks"] = []
        print(f"(similarity_search failed: {e})")

    # Console display: filtered top-3 (already deduped in RAG.similarity_search)
    sim_snips = (result.get("similar_description_chunks") or [])[:3]
    print("=== SIMILARITY SNIPPETS (top 3) ===")
    if sim_snips:
        for i, c in enumerate(sim_snips, start=1):
            txt = (c.get("text","") or "").strip()
            if len(txt) > 220:
                txt = txt[:217].rstrip() + "..."
            score = c.get("score")
            if isinstance(score,(int,float)):
                print(f"{i}. ({score:.3f}) {txt}")
            else:
                print(f"{i}. {txt}")
    else:
        print("(none)")
    print("=== END SIMILARITY SNIPPETS ===\n")

    summary, used_prompt = _summarize_with_llm(user_query, result)

    print(f"POST https://api.groq.com/v1/chat/completions model=" + (_current_model or DEFAULT_MODEL))
    if len(_conversation_history) >= 1:
        print("current message:")
    print("=== LLM PROMPT (truncated to 800 chars) ===")
    print(used_prompt[:800] + ("..." if len(used_prompt) > 800 else ""))
    print("=== END PROMPT ===")

    print("\n=== LLM SUMMARY ===")
    print(summary if summary else "(LLM empty response)")
    print("\nTurn complete.\n")

# -------------------------
# CLI / interactive
# -------------------------
def main():
    print("==============================================")
    print("Interactive Recipe Chatbot (Pipeline + Groq)")
    print("Type a recipe-related query in PT or EN.")
    print("Examples: receitas com manteiga de amendoim | quick vegan pasta")
    print("Commands: /exit to quit")
    print("==============================================")
    if len(sys.argv) > 1:
        _process_query(" ".join(sys.argv[1:]))
        return
    if not sys.stdin.isatty():
        print("No interactive TTY. Provide a query as CLI argument.")
        sys.exit(1)
    try:
        while True:
            q = input("you> ").strip()
            if not q or q.lower() in {"/exit","exit","quit"}:
                break
            _process_query(q)
    except (KeyboardInterrupt, EOFError):
        pass
    print("Goodbye!")

if __name__ == "__main__":
    main()