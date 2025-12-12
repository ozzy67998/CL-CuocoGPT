import os
import sys
from typing import Dict, Any, Optional

# Robust imports (avoid swallowing errors and ensure package context)
if __package__ is None:
    # Running as a script: add src root so "NLP.*" imports work
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

try:
    # Preferred absolute package imports
    from NLP.infer_intent import predict_intent
    from NLP.entity_extraction import build_spacy_pipeline, KGIndex, extract_and_link, load_kg_cached
    from NLP import sparql_queries
except ImportError:
    try:
        # Fallback to relative (when executed via `python -m NLP.pipeline`)
        from .infer_intent import predict_intent
        from .entity_extraction import build_spacy_pipeline, KGIndex, extract_and_link
        from . import sparql_queries
    except Exception:
        import traceback
        print("Failed to import pipeline dependencies. Full traceback:")
        traceback.print_exc()
        raise  # stop early; do not continue with undefined names

BASE_DIR = os.path.dirname(__file__)
DEFAULT_TTL = os.path.normpath(os.path.join(BASE_DIR, "../../data/curated/recipes_graph_cleaned.ttl"))
TTL_PATH = os.environ.get("RECIPES_TTL_PATH", DEFAULT_TTL)

# Ensure build_spacy_pipeline is defined before using
try:
    NLP = build_spacy_pipeline(lang_priority="pt")
except NameError:
    raise RuntimeError("build_spacy_pipeline is not defined in entity_extraction.py. Verify its name.")
KG = load_kg_cached(TTL_PATH)

INTENTS_NEEDING_EXTRACTION = {
    "find_recipe",
    "get_prep_time",
    "retrieve_ingredients",
    "list_by_ingredient",
    "list_by_tag",
    "list_by_time",
}

# NEW: SLOT EXTRACTION WITHOUT RUNNING SPARQL
def extract_slots_only(text: str, top_k: int = 1):
    """Extract intent + slots WITHOUT running SPARQL queries."""
    preds = predict_intent(text, top_k=top_k)
    top_intent, conf = preds[0] if preds else (None, 0.0)

    slots = {}
    if top_intent in INTENTS_NEEDING_EXTRACTION:
        slots = extract_and_link(text, intent=top_intent, nlp=NLP, kg=KG)

    return top_intent, conf, slots


# SEQUENTIAL PRINT HELPERS
def _fmt_nutrition(n: Dict[str, Any]) -> str:
    if not n:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in n.items())

def _fmt_steps(steps):
    if not steps:
        return ["      (no steps)"]
    return [f"      {i+1:02d}. {s}" for i, s in enumerate(steps)]

def _fmt_recipe(r: Dict[str, Any]):
    out = []
    name = r.get('recipe_name') or r.get('recipe_uri')
    out.append(f"    • {name}")
    if 'recipe_uri' in r:
        out.append(f"      URI: {r['recipe_uri']}")
    if 'origin' in r:
        out.append(f"      Origin: {r['origin']}")
    if 'minutes' in r and r.get('minutes') is not None:
        out.append(f"      Minutes: {r['minutes']}")
    if 'n_steps' in r:
        out.append(f"      #Steps: {r['n_steps']}")
    if 'n_ingredients' in r:
        out.append(f"      #Ingredients: {r['n_ingredients']}")
    tags = r.get('tags')
    if tags is not None:
        out.append(f"      Tags: {', '.join(tags) if tags else '(none)'}")
    ings = r.get('ingredients')
    if ings is not None:
        out.append(f"      Ingredients ({len(ings)}): {', '.join(ings) if ings else '(none)'}")
    nutrition = r.get('nutrition')
    if nutrition is not None:
        out.append(f"      Nutrition: {_fmt_nutrition(nutrition)}")
    steps = r.get('steps')
    if steps is not None:
        out.append("      Steps:")
        out.extend(_fmt_steps(steps))
    return out


# UTIL SLOT FUNCTIONS
def _slot_top_label(slots: Dict[str, Any], key: str) -> Optional[str]:
    v = slots.get(key)
    if not v:
        return None
    if isinstance(v, list) and v:
        head = v[0]
        if isinstance(head, (list, tuple)) and head:
            return str(head[0])
        return str(head)
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
        return v[0]
    if isinstance(v, str):
        return v
    return None


def _slot_all_labels(slots: Dict[str, Any], key: str) -> list[str]:
    v = slots.get(key)
    if not v:
        return []
    labels = []
    if isinstance(v, list):
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                labels.append(str(item[0]))
            elif isinstance(item, str):
                labels.append(item)
    elif isinstance(v, (list, tuple)) and len(v) >= 1 and isinstance(v[0], str):
        labels.append(v[0])
    elif isinstance(v, str):
        labels.append(v)
    return labels


def _slot_all_scored(slots: Dict[str, Any], key: str) -> list[tuple[str, float]]:
    v = slots.get(key)
    out: list[tuple[str, float]] = []
    if not v:
        return out
    if isinstance(v, list):
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    out.append((str(item[0]), float(item[1])))
                except Exception:
                    out.append((str(item[0]), 0.0))
            elif isinstance(item, str):
                out.append((item, 0.0))
    elif isinstance(v, (list, tuple)) and len(v) >= 1:
        try:
            out.append((str(v[0]), float(v[1]) if len(v) > 1 else 0.0))
        except Exception:
            out.append((str(v[0]), 0.0))
    elif isinstance(v, str):
        out.append((v, 0.0))
    return out


# MAIN LOGIC WITH SEQUENTIAL QUERY EXECUTION
def handle_query(text: str, intent_top_k: int = 1, sparql_top_k: int = 5) -> Dict[str, Any]:
    preds = predict_intent(text, top_k=intent_top_k)
    top_intent, conf = preds[0] if preds else (None, 0.0)

    top_intent = top_intent.strip() if isinstance(top_intent, str) else top_intent

    result: Dict[str, Any] = {"intent": top_intent, "confidence": conf, "text": text}

    slots = {}
    if top_intent in INTENTS_NEEDING_EXTRACTION:
        slots = extract_and_link(text, intent=top_intent, nlp=NLP, kg=KG)
        result["slots"] = slots

        # INTENT: list_by_ingredient
        if top_intent == "list_by_ingredient":
            ingredients = _slot_all_labels(slots, "ingredient")
            seq: list[Dict[str, Any]] = []
            for ing in ingredients:
                print(f"\n---- Searching recipes with ingredient: {ing}----")
                recs = sparql_queries.query_list_by_ingredient(KG.graph, ing, top_k=sparql_top_k)
                if not recs:
                    print("   ✖ No recipes found.")
                    continue
                print(f"   ✔ Found {len(recs)} recipes:")
                for r in recs:
                    for line in _fmt_recipe(r):
                        print(line)
                seq.extend(recs)
            result["kg_results"] = seq

        # INTENT: list_by_tag
        elif top_intent == "list_by_tag":
            tags = _slot_all_labels(slots, "tag")
            seq: list[Dict[str, Any]] = []
            for tag in tags:
                print(f"\n---- Searching recipes with tag: {tag}----")
                recs = sparql_queries.query_list_by_tag(KG.graph, tag, top_k=sparql_top_k)
                if not recs:
                    print("   ✖ No recipes found.")
                    continue
                print(f"   ✔ Found {len(recs)} recipes:")
                for r in recs:
                    for line in _fmt_recipe(r):
                        print(line)
                seq.extend(recs)
            # deduplicate by recipe_uri
            seen = set()
            dedup = []
            for r in seq:
                uri = r.get("recipe_uri")
                if uri and uri in seen:
                    continue
                if uri:
                    seen.add(uri)
                dedup.append(r)
            result["kg_results"] = dedup

        # INTENT: find_recipe
        elif top_intent == "find_recipe":
            names = _slot_all_labels(slots, "recipe_name")
            seq: list[Dict[str, Any]] = []
            for name in names:
                print(f"\n---- Searching recipe: {name}----")
                r = sparql_queries.query_find_recipe(KG.graph, name)
                if not r:
                    print("   ✖ No recipe found.")
                    continue
                print("   ✔ Found recipe:")
                for line in _fmt_recipe(r):
                    print(line)
                seq.append(r)
            result["kg_results"] = seq

        # INTENT: retrieve_ingredients
        elif top_intent == "retrieve_ingredients":
            names = _slot_all_labels(slots, "recipe_name")
            seq: list[Dict[str, Any]] = []
            for name in names:
                print(f"\n---- Retrieving ingredients for: {name}----")
                r = sparql_queries.query_retrieve_ingredients(KG.graph, name)
                if not r:
                    print("   ✖ No recipe found.")
                    continue
                print("   ✔ Ingredients:")
                print("    • " + (r.get('recipe_name') or r.get('recipe_uri')))
                ings = r.get("ingredients", [])
                print("      " + (", ".join(ings) if ings else "(none)"))
                seq.append(r)
            result["kg_results"] = seq

        # INTENT: get_prep_time
        elif top_intent == "get_prep_time":
            names = _slot_all_labels(slots, "recipe_name")
            seq: list[Dict[str, Any]] = []
            for name in names:
                print(f"\n---- Getting prep time for: {name}----")
                times = sparql_queries.query_get_prep_time(KG.graph, name, top_k=sparql_top_k)
                if not times:
                    print("   ✖ No prep time found.")
                    continue
                for t in times:
                    print(f"   ✔ {t['recipe_name']} -> {t['minutes']} minutes")
                seq.extend(times)
            result["kg_results"] = seq

        # INTENT: list_by_time
        elif top_intent == "list_by_time":
            print("[DEBUG] Entered list_by_time handler")
            
            # Get extracted values
            exact_minutes = slots.get("cooking_time")
            max_minutes = slots.get("max_minutes")
            
            print(f"[DEBUG] Exact: {exact_minutes}, Max: {max_minutes}")

            # CASE 1: Range Query (e.g., "menos de 30 minutos")
            if max_minutes is not None:
                print(f"\n---- Range search: minutes < {max_minutes} ----")
                recs = sparql_queries.query_by_max_minutes(KG.graph, max_minutes, top_k=sparql_top_k)
                if recs:
                    print(f"   ✔ Found {len(recs)} recipes under {max_minutes} mins")
                    for r in recs:
                        for line in _fmt_recipe(r):
                            print(line)
                    result["kg_results"] = recs
                    result["result_basis"] = "range_minutes"
                else:
                    print("   ✖ No recipes found in that time range.")

            # CASE 2: Exact Query (e.g., "30 minutos")
            elif exact_minutes is not None:
                print(f"\n---- Exact time search: minutes == {exact_minutes} ----")
                # Pass as list to match signature
                recs = sparql_queries.query_by_exact_minutes(KG.graph, [exact_minutes], top_k=sparql_top_k)
                if recs:
                    print(f"   ✔ Found {len(recs)} recipes exactly {exact_minutes} mins")
                    for r in recs:
                        for line in _fmt_recipe(r):
                            print(line)
                    result["kg_results"] = recs
                    result["result_basis"] = "exact_minutes"
                else:
                    print("   ✖ No exact matches. Trying fuzzy tag fallback.")
                    # Fallback logic below...
                    
            # CASE 3: No numbers found -> Fuzzy Tag Fallback
            if (exact_minutes is None and max_minutes is None) or (not result.get("kg_results")):
                print("\n---- No time numbers found or no results. Using fuzzy tag fallback. ----")
                tag_slots = extract_and_link(text, intent="list_by_tag", nlp=NLP, kg=KG)
                tags_scored = _slot_all_scored(tag_slots, "tag")
                
                seq: list[Dict[str, Any]] = []
                seen = set()
                
                # If we have a number but no results, try to find tags like "30-minutes-or-less"
                # This handles cases where the number exists but isn't in the exact 'minutes' property
                if exact_minutes or max_minutes:
                    val = exact_minutes or max_minutes
                    if val <= 15:
                        tags_scored.insert(0, ("15-minutes-or-less", 100.0))
                    elif val <= 30:
                        tags_scored.insert(0, ("30-minutes-or-less", 100.0))
                    elif val <= 60:
                        tags_scored.insert(0, ("60-minutes-or-less", 100.0))

                for tag, score in tags_scored[:sparql_top_k]:
                    print(f"\n---- Fuzzy tag fallback: {tag} (score {score:.1f}) ----")
                    recs = sparql_queries.query_list_by_tag(KG.graph, tag, top_k=sparql_top_k)
                    if not recs:
                        continue
                    for r in recs:
                        uri = r.get("recipe_uri")
                        if uri in seen:
                            continue
                        seen.add(uri)
                        for line in _fmt_recipe(r):
                            print(line)
                        seq.append(r)
                result["kg_results"] = seq
                result["result_basis"] = "fuzzy_tag_fallback"

    return result


if __name__ == "__main__":
    # Default values
    text = "Quais são os ingredientes da francesinha?"
    intent_top_k = 1
    sparql_top_k = 5

    # Simple manual arg parsing (no external deps)
    args = sys.argv[1:]
    i = 0
    remaining = []
    while i < len(args):
        a = args[i]
        if a == "--intent-topk" and i + 1 < len(args):
            try:
                intent_top_k = int(args[i+1])
            except Exception:
                pass
            i += 2
            continue
        if a == "--sparql-topk" and i + 1 < len(args):
            try:
                sparql_top_k = int(args[i+1])
            except Exception:
                pass
            i += 2
            continue
        if a == "--topk" and i + 1 < len(args):  # convenience: sets both
            try:
                k = int(args[i+1])
                intent_top_k = max(1, k)
                sparql_top_k = max(1, k)
            except Exception:
                pass
            i += 2
            continue
        remaining.append(a)
        i += 1

    if remaining:
        text = " ".join(remaining)

    # 1) Extract intent & slots FIRST
    intent, conf, slots = extract_slots_only(text, top_k=intent_top_k)

    print("\n========== QUERY INFO ==========")
    print(f"Intent: {intent} (confidence={conf:.2f})")
    print(f"Query: {text}")

    # Print detected slots BEFORE running SPARQL
    if slots:
        # Print translation / normalization info first if available
        if any(k in slots for k in ("original_query", "translated_query", "detected_language")):
            print("\n=== Normalization & Translation ===")
            if "original_query" in slots:
                print(f"Original: {slots['original_query']}")
            else:
                print(f"Original: {text}")
            if "detected_language" in slots:
                print(f"Detected language: {slots['detected_language']}")
            if "translated_query" in slots:
                print(f"Translated (for intent/NER): {slots['translated_query']}")

        print("\nDetected slots:")
        for k, v in slots.items():
            if k in {"detected_language", "original_query", "translated_query"}:
                continue

            if k == "cooking_time" and isinstance(v, int):
                print(f"  - cooking_time: {v} minutes")
                continue

            if isinstance(v, list):
                rendered = []
                for item in v:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        rendered.append(f"{item[0]} ({item[1]:.1f})")
                    elif isinstance(item, (list, tuple)) and len(item) == 1:
                        rendered.append(str(item[0]))
                    else:
                        rendered.append(str(item))
                print(f"  - {k}: {', '.join(rendered)}")
            else:
                print(f"  - {k}: {v}")

    print("\n========== RESULTS (Sequential Execution) ==========\n")
    print("(Below, each SPARQL query is executed and printed one by one.)\n")

    # 2) Now run the real pipeline (prints SPARQL)
    handle_query(text, intent_top_k=intent_top_k, sparql_top_k=sparql_top_k)
