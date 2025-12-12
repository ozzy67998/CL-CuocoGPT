"""Quick demo runner for entity extraction + KG linking.

Usage:
    python src/backend/NLP/demo_entities.py "ingredientes da francesinha"
"""

import os
import sys
import time

# Support both package and direct script execution
try:
    from .entity_extraction import extract_and_link, build_spacy_pipeline, load_kg_cached
except Exception:
    sys.path.append(os.path.dirname(__file__))
    from entity_extraction import extract_and_link, build_spacy_pipeline, load_kg_cached


def main():
    if len(sys.argv) < 2:
        print("Provide a query string, e.g.: python demo_entities.py 'ingredientes da francesinha'")
        sys.exit(1)

    query = sys.argv[1]
    base_dir = os.path.dirname(__file__)
    default_a = os.path.normpath(os.path.join(base_dir, "../../data/curated/recipes_graph_cleaned.ttl"))
    default_b = os.path.normpath(os.path.join(base_dir, "../../data/curated/example.ttl"))
    ttl_path = default_a if os.path.exists(default_a) else default_b

    # ---------------------------------------------------------
    # TEST: measure KG load time
    # ---------------------------------------------------------
    t0 = time.time()
    kg = load_kg_cached(ttl_path)
    print(f"⏱️ KG load time: {time.time() - t0:.3f} sec")

    # spaCy load timing too (for curiosity)
    t1 = time.time()
    nlp = build_spacy_pipeline("pt")
    print(f"⏱️ spaCy load time: {time.time() - t1:.3f} sec")

    intent = "find_recipe"  # or "list_by_ingredient" / "list_by_tag" / "retrieve_ingredients"

    # ---------------------------------------------------------
    # Run extraction
    # ---------------------------------------------------------
    result = extract_and_link(query, kg, nlp, intent)

    print("\nQuery:", query)
    print("Linked slots:")
    print(result)


if __name__ == "__main__":
    main()
