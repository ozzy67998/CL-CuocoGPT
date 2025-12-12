import rdflib
import random
import networkx as nx
import matplotlib.pyplot as plt

# --- Load full KG ---
g = rdflib.Graph()
g.parse("../../data/curated/recipes_graph_cleaned.ttl", format="turtle")

# --- Extract all recipes ---
EX = rdflib.Namespace("http://example.org/recipes#")
recipes = list(g.subjects(rdflib.RDF.type, EX.Recipe))
print(f"Total recipes: {len(recipes)}")

# --- Sample 100 recipes ---
sample_recipes = random.sample(recipes, 100)

# --- Build NetworkX graph ---
G = nx.DiGraph()

for recipe in sample_recipes:
    # Add recipe node with label
    recipe_label = g.value(recipe, rdflib.RDFS.label)
    G.add_node(str(recipe), label=str(recipe_label), type="recipe")

    # Add ingredients
    for ing in g.objects(recipe, EX.hasIngredient):
        ing_label = g.value(ing, rdflib.RDFS.label)
        G.add_node(str(ing), label=str(ing_label), type="ingredient")
        G.add_edge(str(recipe), str(ing), label="hasIngredient")

    # Add steps
    for seq in g.objects(recipe, EX.hasStep):
        for idx, step in enumerate(g.objects(seq, None)):
            if str(idx) == "":  # skip non-rdf:_n
                continue
            step_label = g.value(step, rdflib.RDFS.comment)
            G.add_node(str(step), label=str(step_label), type="step")
            G.add_edge(str(recipe), str(step), label="hasStep")

    # Add tags
    for tag in g.objects(recipe, EX.hasTag):
        tag_label = g.value(tag, rdflib.RDFS.label)
        G.add_node(str(tag), label=str(tag_label), type="tag")
        G.add_edge(str(recipe), str(tag), label="hasTag")

# --- Draw the graph ---
plt.figure(figsize=(18, 12))

# Position nodes using spring layout
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Draw nodes with different colors for type
colors = []
for n, d in G.nodes(data=True):
    if d["type"] == "recipe":
        colors.append("orange")
    elif d["type"] == "ingredient":
        colors.append("lightgreen")
    elif d["type"] == "step":
        colors.append("skyblue")
    elif d["type"] == "tag":
        colors.append("pink")
    else:
        colors.append("gray")

nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800, alpha=0.9)
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", alpha=0.6)

# Draw labels for clarity
labels = {n: d["label"] for n, d in G.nodes(data=True)}
nx.draw_networkx_labels(G, pos, labels, font_size=8)

plt.title("Sample Recipes Knowledge Graph (100 recipes)")
plt.axis("off")
plt.show()
