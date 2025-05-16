from langchain_community.graphs import RdfGraph
from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.sparql import GraphSparqlQAChain
from langchain.chains import GraphQAChain

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# graph = RdfGraph(
#     source_file="SOMA.owl",
#     serialization="xml",  # Use 'xml' for OWL files
#     standard="owl"        # Specify the standard as 'owl'
# )
graph = RdfGraph(
    source_file="http://www.w3.org/People/Berners-Lee/card",
    standard="rdf",
    local_copy="test.ttl",
)
graph.load_schema()
print(graph.get_schema)

llm = ChatOpenAI(model="gpt-4o")


# graph_qa = GraphSparqlQAChain(graph=graph)
# qa = graph_qa.from_llm(llm=llm)

# response = qa.invoke({"user": "what are the subclasses of PhysicalArtifact"})
# print(response)

from openai import OpenAI
chain = GraphQAChain.from_llm(OpenAI(), graph=graph, verbose=True)
chain.run("What is Tim Berners-Lee's work homepage?")