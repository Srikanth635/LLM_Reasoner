from owlready2 import get_ontology
import networkx as nx

# Step 1: Load the ontology
# Replace with your ontology file path or IRI
ontology = get_ontology("SOMA.owl").load()

# Step 2: Initialize a directed graph
G = nx.DiGraph()

# Step 3: Add nodes and edges for classes (subclass relationships)
for cls in ontology.classes():
    # Add class as a node with its, using IRI or label
    class_name = cls.name or str(cls)
    G.add_node(class_name, type="class")

    # Add edges for subclass relationships
    for parent in cls.is_a:
        if isinstance(parent, ontology.Thing.__class__):  # Ensure it's a class
            parent_name = parent.name or str(parent)
            G.add_node(parent_name, type="class")
            G.add_edge(class_name, parent_name, relationship="subClassOf")

# Step 4: Add nodes and edges for individuals and object properties
for individual in ontology.individuals():
    # Add individual as a node
    individual_name = individual.name or str(individual)
    G.add_node(individual_name, type="individual")

    # Add edges for class assertions (individual is instance of class)
    for cls in individual.is_a:
        if isinstance(cls, ontology.Thing.__class__):
            class_name = cls.name or str(cls)
            G.add_edge(individual_name, class_name, relationship="instanceOf")

    # Add edges for object properties
    for prop in ontology.object_properties():
        for target in prop[individual]:
            target_name = target.name or str(target)
            G.add_node(target_name, type="individual")
            G.add_edge(individual_name, target_name, relationship=prop.name or str(prop))

# Step 5: (Optional) Print graph summary
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Step 6: (Optional) Visualize the graph
# Requires matplotlib: pip install matplotlib

# import matplotlib.pyplot as plt
#
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8)
# edge_labels = nx.get_edge_attributes(G, "relationship")
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
# plt.show()

import networkx as nx
from langchain.llms import OpenAI
from langchain_graph_retriever import GraphRetriever
from langchain.chains import RetrievalQA

llm = OpenAI(temperature=0)
graph_retriever = GraphRetriever(G)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=graph_retriever)

query = "What are the classes that talk about movement"
result = qa({"query":query})
print(result['result'])