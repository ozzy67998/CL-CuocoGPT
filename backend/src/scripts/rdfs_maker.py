import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import RDF, RDFS
import ast
import re

# --- Helper function to clean strings for URIs ---
def clean_uri(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Remove or replace unsafe characters
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)  # only keep alphanumeric + underscore
    text = re.sub(r"_+", "_", text)  # collapse multiple underscores
    return text.strip("_")

# Load data
df = pd.read_csv("../../data/curated/filtered_pt_en.csv")

# Define namespace
EX = Namespace("http://example.org/recipes#")

# Create RDF graph
g = Graph()
g.bind("ex", EX)
g.bind("rdfs", RDFS)

# Define main classes
g.add((EX.Recipe, RDF.type, RDFS.Class))
g.add((EX.Ingredient, RDF.type, RDFS.Class))
g.add((EX.Tag, RDF.type, RDFS.Class))
g.add((EX.Step, RDF.type, RDFS.Class))
g.add((EX.Nutrition, RDF.type, RDFS.Class))

# Iterate through each recipe
for _, row in df.iterrows():
    recipe_uri = URIRef(EX[f"Recipe_{clean_uri(str(row['id']))}"])
    g.add((recipe_uri, RDF.type, EX.Recipe))
    
    # Use rdfs:label for human-readable name
    g.add((recipe_uri, RDFS.label, Literal(row['name'])))
    
    # Optional: keep numeric ID
    g.add((recipe_uri, EX.id, Literal(row['id'])))
    g.add((recipe_uri, EX.minutes, Literal(row['minutes'])))
    g.add((recipe_uri, EX.n_steps, Literal(row['n_steps'])))
    g.add((recipe_uri, EX.n_ingredients, Literal(row['n_ingredients'])))
    g.add((recipe_uri, EX.origin, Literal(row['origin'])))

    # Parse text fields that are stored as stringified lists
    for col in ["tags", "steps", "ingredients", "nutrition"]:
        try:
            row[col] = ast.literal_eval(row[col]) if isinstance(row[col], str) else row[col]
        except Exception:
            row[col] = []

    # Add tags
    if isinstance(row["tags"], list):
        for tag in row["tags"]:
            tag_uri = URIRef(EX[f"Tag_{clean_uri(tag)}"])
            g.add((tag_uri, RDF.type, EX.Tag))
            g.add((recipe_uri, EX.hasTag, tag_uri))
            g.add((tag_uri, RDFS.label, Literal(tag)))

    # Add ingredients
    if isinstance(row["ingredients"], list):
        for ing in row["ingredients"]:
            ing_uri = URIRef(EX[f"Ingredient_{clean_uri(ing)}"])
            g.add((ing_uri, RDF.type, EX.Ingredient))
            g.add((recipe_uri, EX.hasIngredient, ing_uri))
            g.add((ing_uri, RDFS.label, Literal(ing)))

    # Add steps as ordered rdf:Seq
    if isinstance(row["steps"], list) and len(row["steps"]) > 0:
        seq_uri = URIRef(EX[f"Recipe_{clean_uri(str(row['id']))}_StepsSeq"])
        g.add((seq_uri, RDF.type, RDF.Seq))
        g.add((recipe_uri, EX.hasStep, seq_uri))
        for idx, step in enumerate(row["steps"], 1):
            step_uri = URIRef(EX[f"Recipe_{clean_uri(str(row['id']))}_Step_{idx}"])
            g.add((step_uri, RDF.type, EX.Step))
            g.add((step_uri, RDFS.comment, Literal(step)))
            g.add((seq_uri, RDF[f"_{idx}"], step_uri))

    # Add nutrition as nested object
    if isinstance(row["nutrition"], list) and len(row["nutrition"]) == 7:
        nutrition_uri = URIRef(EX[f"Recipe_{clean_uri(str(row['id']))}_Nutrition"])
        g.add((nutrition_uri, RDF.type, EX.Nutrition))
        g.add((recipe_uri, EX.hasNutrition, nutrition_uri))
        nutrients = ["calories", "totalFat", "sugar", "sodium", "protein", "saturatedFat", "unknown"]
        for nutrient, value in zip(nutrients, row["nutrition"]):
            g.add((nutrition_uri, EX[nutrient], Literal(value)))

# Serialize graph to Turtle
output_path = "../../data/curated/recipes_graph_cleaned.ttl"
g.serialize(output_path, format="turtle")

print(f"✅ Cleaned Knowledge Graph saved as {output_path}")
