import sys
from rdflib import Graph, RDF
from pyvis.network import Network
from urllib.parse import urlparse

def short_name(uri):
    """Shorten long URIs for nicer labels."""
    if "#" in uri:
        return uri.split("#")[-1]
    elif "/" in uri:
        return uri.split("/")[-1]
    return uri

def visualize_ttl(ttl_file, output_html="kg_visualization.html"):
    print(f"Loading RDF graph from {ttl_file}...")
    g = Graph()
    g.parse(ttl_file, format="ttl")
    print(f"Graph has {len(g)} triples.")

    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.force_atlas_2based()  # More stable layout than barnes_hut

    # Assign colors by RDF type
    color_map = {
        "Recipe": "#97C2FC",
        "Ingredient": "#FB7E81",
        "Tag": "#F5A45D",
        "Step": "#7BE141",
        "Nutrition": "#E0A8F0"
    }
    default_color = "#CCCCCC"

    for s, p, o in g:
        s_str, p_str, o_str = str(s), str(p), str(o)
        s_label, o_label = short_name(s_str), short_name(o_str)
        p_label = short_name(p_str)

        # determine node color based on type
        color_s = default_color
        color_o = default_color
        for rdf_type in g.objects(s, RDF.type):
            color_s = color_map.get(short_name(str(rdf_type)), default_color)
        for rdf_type in g.objects(o, RDF.type):
            color_o = color_map.get(short_name(str(rdf_type)), default_color)

        net.add_node(s_str, label=s_label, color=color_s)
        net.add_node(o_str, label=o_label, color=color_o)
        net.add_edge(s_str, o_str, title=p_label)

    # Simplified config to avoid 0% stuck bug
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "size": 12,
        "font": {"size": 12, "strokeWidth": 2}
      },
      "edges": {
        "smooth": false,
        "color": {"inherit": true}
      },
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "stabilization": {"enabled": true, "iterations": 1000}
      }
    }
    """)

    net.write_html(output_html)
    print(f"✅ Visualization saved to: {output_html}")

if __name__ == "__main__":
    ttl_file = "../../data/curated/example.ttl"
    output_html = "../../data/curated/kg_visualization.html"
    visualize_ttl(ttl_file, output_html)
