# sparql_queries.py
from rdflib import Graph, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import XSD

EX = Namespace("http://example.org/recipes#")

def _esc(s: str) -> str:
    return s.replace('"', '\\"')

# --- Helpers to enrich a recipe node ---
_NUTRITION_PROPS = [
    "calories", "protein", "saturatedFat", "sodium", "sugar", "totalFat", "unknown",
]

def _nutrition_dict(graph: Graph, recipe) -> dict:
    out = {}
    n = graph.value(recipe, EX.hasNutrition)
    if n is None:
        return out
    for k in _NUTRITION_PROPS:
        v = graph.value(n, EX[k])
        if v is not None:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = str(v)
    return out

def _ingredients_list(graph: Graph, recipe) -> list[str]:
    vals = []
    for ing in graph.objects(recipe, EX.hasIngredient):
        label = graph.value(ing, RDFS.label)
        if label:
            vals.append(str(label))
    # de-duplicate, keep order
    seen = set()
    uniq = []
    for x in vals:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def _tags_list(graph: Graph, recipe) -> list[str]:
    vals = []
    for t in graph.objects(recipe, EX.hasTag):
        label = graph.value(t, RDFS.label)
        if label:
            vals.append(str(label))
    # de-duplicate, keep order
    seen = set()
    uniq = []
    for x in vals:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def _steps_list(graph: Graph, recipe) -> list[str]:
    steps = []
    seq = graph.value(recipe, EX.hasStep)
    if seq is None:
        return steps
    # Iterate rdf:_1, rdf:_2, ...
    i = 1
    while True:
        pred = RDF[f"_{i}"]
        step_node = graph.value(seq, pred)
        if step_node is None:
            break
        comment = graph.value(step_node, RDFS.comment)
        if comment:
            steps.append(str(comment))
        i += 1
    return steps

def query_list_by_ingredient(graph: Graph, ingredient_name: str, top_k: int = 5, debug_print: bool = False):
    """
    Return up to top_k distinct recipes containing the ingredient, enriched with
    tags, ingredients and ordered steps. Nutrition is expanded to a dict.
    Also returns minutes if present.
    """
    needle = _esc(ingredient_name)
    q = f"""
    SELECT ?recipe ?label ?n_steps ?minutes
    WHERE {{
        {{
            SELECT DISTINCT ?recipe WHERE {{
                ?recipe rdf:type ex:Recipe .
                ?recipe ex:hasIngredient ?ing .
                ?ing rdfs:label ?ingLabel .
                FILTER(CONTAINS(LCASE(STR(?ingLabel)), LCASE("{needle}")))
            }} LIMIT {top_k}
        }}
        ?recipe rdfs:label ?label .
        ?recipe ex:n_steps ?n_steps .
        OPTIONAL {{ ?recipe ex:minutes ?minutes . }}
    }}
    """
    if debug_print:
        print("\n[SPARQL QUERY][list_by_ingredient]" )
        print(q.strip())
    results = []
    for row in graph.query(q, initNs={"ex": EX, "rdf": RDF, "rdfs": RDFS}):
        mins = None
        if row.minutes is not None:
            try:
                mins = int(str(row.minutes))
            except Exception:
                try:
                    mins = float(str(row.minutes))
                except Exception:
                    mins = str(row.minutes)
        recipe = row.recipe
        results.append({
            "recipe_uri": str(recipe),
            "recipe_name": str(row.label),
            "n_steps": int(row.n_steps),
            "minutes": mins,
            "nutrition": _nutrition_dict(graph, recipe),
            "tags": _tags_list(graph, recipe),
            "ingredients": _ingredients_list(graph, recipe),
            "steps": _steps_list(graph, recipe),
        })
    return results

def query_list_by_tag(graph: Graph, tag_name: str, top_k: int = 5, debug_print: bool = False):
    """Return a LIST of up to top_k distinct recipes having the tag.
    Also returns minutes if present.
    """
    needle = _esc(tag_name)
    q = f"""
    SELECT ?recipe ?label ?n_steps ?minutes
    WHERE {{
        {{
            SELECT DISTINCT ?recipe WHERE {{
                ?recipe rdf:type ex:Recipe .
                ?recipe ex:hasTag ?t .
                ?t rdfs:label ?tagLabel .
                FILTER(CONTAINS(LCASE(STR(?tagLabel)), LCASE("{needle}")))
            }} LIMIT {top_k}
        }}
        ?recipe rdfs:label ?label .
        ?recipe ex:n_steps ?n_steps .
        OPTIONAL {{ ?recipe ex:minutes ?minutes . }}
    }}
    """
    if debug_print:
        print("\n[SPARQL QUERY][list_by_tag]")
        print(q.strip())
    results = []
    for row in graph.query(q, initNs={"ex": EX, "rdf": RDF, "rdfs": RDFS}):
        mins = None
        if row.minutes is not None:
            try:
                mins = int(str(row.minutes))
            except Exception:
                try:
                    mins = float(str(row.minutes))
                except Exception:
                    mins = str(row.minutes)
        recipe = row.recipe
        results.append({
            "recipe_uri": str(recipe),
            "recipe_name": str(row.label),
            "n_steps": int(row.n_steps),
            "minutes": mins,
            "nutrition": _nutrition_dict(graph, recipe),
            "tags": _tags_list(graph, recipe),
            "ingredients": _ingredients_list(graph, recipe),
            "steps": _steps_list(graph, recipe),
        })
    return results

def query_find_recipe(graph: Graph, recipe_name: str, debug_print: bool = False):
    """
    Return full details for a single recipe matched by name, including
    nutrition dict, tags, ingredients and ordered steps.
    """
    needle = _esc(recipe_name)
    q = f"""
    SELECT ?recipe ?label ?n_steps ?id ?minutes ?n_ingredients WHERE {{
        ?recipe rdf:type ex:Recipe .
        ?recipe rdfs:label ?label .
        FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{needle}")))
        ?recipe ex:n_steps ?n_steps .
        ?recipe ex:id ?id .
        ?recipe ex:minutes ?minutes .
        ?recipe ex:n_ingredients ?n_ingredients .
    }} LIMIT 1
    """
    if debug_print:
        print("\n[SPARQL QUERY][find_recipe]")
        print(q.strip())
    for row in graph.query(q, initNs={"ex": EX, "rdf": RDF, "rdfs": RDFS}):
        recipe = row.recipe
        return {
            "recipe_uri": str(recipe),
            "recipe_name": str(row.label),
            "n_steps": int(row.n_steps),
            "id": int(row.id),
            "minutes": int(row.minutes),
            "n_ingredients": int(row.n_ingredients),
            "nutrition": _nutrition_dict(graph, recipe),
            "tags": _tags_list(graph, recipe),
            "ingredients": _ingredients_list(graph, recipe),
            "steps": _steps_list(graph, recipe),
        }
    return None

def query_retrieve_ingredients(graph: Graph, recipe_name: str, debug_print: bool = False):
    """
    Return ingredients list for a single recipe matched by name.
    Also returns minutes if present.
    """
    needle = _esc(recipe_name)
    q = f"""
    SELECT ?recipe ?label ?minutes WHERE {{
        ?recipe rdf:type ex:Recipe .
        ?recipe rdfs:label ?label .
        FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{needle}")))
        OPTIONAL {{ ?recipe ex:minutes ?minutes . }}
    }} LIMIT 1
    """
    if debug_print:
        print("\n[SPARQL QUERY][retrieve_ingredients]")
        print(q.strip())
    for row in graph.query(q, initNs={"ex": EX, "rdf": RDF, "rdfs": RDFS}):
        mins = None
        if row.minutes is not None:
            try:
                mins = int(str(row.minutes))
            except Exception:
                try:
                    mins = float(str(row.minutes))
                except Exception:
                    mins = str(row.minutes)
        recipe = row.recipe
        return {
            "recipe_uri": str(recipe),
            "recipe_name": str(row.label),
            "minutes": mins,
            "ingredients": _ingredients_list(graph, recipe),
        }
    return None


def query_get_prep_time(graph: Graph, recipe_name: str, top_k: int = 5, debug_print: bool = False):
    """
    Return up to top_k matches with only:
      - recipe_name
      - time_uri (URI of ex:hasTime node if present; else None)
      - minutes (time attribute)
    """
    needle = _esc(recipe_name)
    q = f"""
    SELECT ?label ?time ?minutes WHERE {{
        ?recipe rdf:type ex:Recipe .
        ?recipe rdfs:label ?label .
        FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{needle}")))

        OPTIONAL {{
            ?recipe ex:hasTime ?time .
            OPTIONAL {{ ?time ex:minutes ?m1 . }}
        }}
        OPTIONAL {{ ?recipe ex:minutes ?m2 . }}
        BIND(COALESCE(?m1, ?m2) AS ?minutes)
    }} LIMIT {top_k}
    """
    if debug_print:
        print("\n[SPARQL QUERY][get_prep_time]")
        print(q.strip())
    results = []
    for row in graph.query(q, initNs={"ex": EX, "rdf": RDF, "rdfs": RDFS}):
        # minutes may be untyped or typed; convert safely
        mins = None
        if row.minutes is not None:
            try:
                mins = int(str(row.minutes))
            except Exception:
                try:
                    mins = float(str(row.minutes))
                except Exception:
                    mins = str(row.minutes)
        results.append({
            "recipe_name": str(row.label),
            "time_uri": str(row.time) if row.time is not None else None,
            "minutes": mins,
        })
    return results

def query_by_exact_minutes(graph, minutes_list, top_k: int = 20):
    """
    Return recipes where ex:minutes equals any value in minutes_list.
    """
    # Flatten inputs
    flat_minutes = []
    def _flatten(item):
        if isinstance(item, (list, tuple)):
            for sub in item:
                _flatten(sub)
        elif isinstance(item, (int, float)):
            flat_minutes.append(int(item))
        elif isinstance(item, str) and item.isdigit():
            flat_minutes.append(int(item))
    _flatten(minutes_list)

    # --- CRITICAL FIX: Return empty immediately if list is empty ---
    if not flat_minutes:
        return []
    # -------------------------------------------------------------

    values = ", ".join(str(m) for m in sorted(set(flat_minutes)))
    q = f"""
    SELECT ?r ?label ?minutes WHERE {{
      ?r a ex:Recipe ;
         rdfs:label ?label ;
         ex:minutes ?minutes .
      FILTER(xsd:integer(?minutes) IN ({values}))
    }}
    LIMIT {top_k}
    """
    
    out = []
    for row in graph.query(q, initNs={"ex": EX, "rdf": RDF, "rdfs": RDFS, "xsd": XSD}):
        recipe = row.r
        out.append({
            "recipe_uri": str(recipe),
            "recipe_name": str(row.label),
            "minutes": int(row.minutes),
            "nutrition": _nutrition_dict(graph, recipe),
            "tags": _tags_list(graph, recipe),
            "ingredients": _ingredients_list(graph, recipe),
            "steps": _steps_list(graph, recipe),
        })
    return out

def query_by_max_minutes(graph, max_minutes: int, top_k: int = 20):
    """
    Return recipes where ex:minutes <= max_minutes
    """
    if max_minutes is None:
        return []
    
    # CHANGED: Added DESC() to ORDER BY to prioritize recipes closer to the limit
    q = f"""
    SELECT ?r ?label ?minutes WHERE {{
      ?r a ex:Recipe ;
         rdfs:label ?label ;
         ex:minutes ?minutes .
      FILTER(xsd:integer(?minutes) < {max_minutes} && xsd:integer(?minutes) > 0)
    }}
    ORDER BY DESC(?minutes)
    LIMIT {top_k}
    """
    out = []
    for row in graph.query(q, initNs={"ex": EX, "rdf": RDF, "rdfs": RDFS, "xsd": XSD}):
        recipe = row.r
        out.append({
            "recipe_uri": str(recipe),
            "recipe_name": str(row.label),
            "minutes": int(row.minutes),
            "nutrition": _nutrition_dict(graph, recipe),
            "tags": _tags_list(graph, recipe),
            "ingredients": _ingredients_list(graph, recipe),
            "steps": _steps_list(graph, recipe),
        })
    return out
