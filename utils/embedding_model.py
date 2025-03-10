import os
from dotenv import load_dotenv
_ = load_dotenv()

from llama_index.core import SimpleDirectoryReader
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings

from typing import Literal
import time

from neo4j import GraphDatabase

import os

# Load the environment variables
AZURE_OPENAI_NORTH_API_KEY = os.getenv("AZURE_OPENAI_NORTH_API_KEY")
AZURE_OPENAI_NORTH_ENDPOINT = os.getenv("AZURE_OPENAI_NORTH_ENDPOINT")
AZURE_OPENAI_EAST_ENDPOINT = os.getenv("AZURE_OPENAI_EAST_ENDPOINT")
AZURE_OPENAI_EAST_API_KEY = os.getenv("AZURE_OPENAI_EAST_API_KEY")
API_VERSION = "2024-06-01"

NEO4J_URI=os.getenv("NEO4J_URI")
NEO4J_USER=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")

# Instantiate the LLM and Embedding models
llm_4o = AzureOpenAI(

    #credentials for the model
    azure_deployment="gpt-4o-mini",
    api_key=AZURE_OPENAI_EAST_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_EAST_ENDPOINT,

    #variables for the model
    temperature=0.5,
    max_tokens=1000,
    top_p=0.95,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

# Instantiate the Embedding model
embedding_model = AzureOpenAIEmbedding(
    azure_deployment="text-embedding-ada-002",
    api_key=AZURE_OPENAI_NORTH_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_NORTH_ENDPOINT,
    embed_batch_size=42,
)

# Set the LLM and Embedding models in the Settings
Settings.embed_model = embedding_model
Settings.llm = llm_4o


#Instantiate the Property Graph Index
graph_store = Neo4jPropertyGraphStore(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
)

entities = Literal[
    "PHISHING_EMAIL",  # Represents a phishing email instance
    "SENDER",  # The sender of the phishing email
    "RECIPIENT",  # The target of the phishing attempt
    "SUBJECT",  # Email subject line
    "CONTENT",  # The body of the email
    "PHONE_NUMBER",  # Extracted phone numbers
    "URL",  # Links in the email
    "KEYWORD",  # Important words (e.g., URGENT, GIFT CARD)
    "ORGANIZATION",  # Impersonated companies (e.g., Amazon, University)
    "PAYMENT_METHOD",  # Payment-related mentions (e.g., Gift Card, Bank Transfer)
    "DATE",  # Date of the email
    "SCAM_PATTERN",  # Identified phishing patterns
]



relations = Literal[
    "SENT_BY",  # (PHISHING_EMAIL) -> (SENDER)
    "SENT_TO",  # (PHISHING_EMAIL) -> (RECIPIENT)
    "HAS_SUBJECT",  # (PHISHING_EMAIL) -> (SUBJECT)
    "HAS_CONTENT",  # (PHISHING_EMAIL) -> (CONTENT)
    "CONTAINS_PHONE_NUMBER",  # (CONTENT) -> (PHONE_NUMBER)
    "CONTAINS_URL",  # (CONTENT) -> (URL)
    "MENTIONS_KEYWORD",  # (CONTENT) -> (KEYWORD)
    "MENTIONS_ORGANIZATION",  # (CONTENT) -> (ORGANIZATION)
    "MENTIONS_PAYMENT_METHOD",  # (CONTENT) -> (PAYMENT_METHOD)
    "HAS_DATE",  # (PHISHING_EMAIL) -> (DATE)
    "MATCHES_SCAM_PATTERN",  # (PHISHING_EMAIL) -> (SCAM_PATTERN)
    "IMPERSONATES",  # (SENDER) -> (ORGANIZATION) (For spoofing detection)
]



schema = {
    "PHISHING_EMAIL": ["SENT_BY", "SENT_TO", "HAS_SUBJECT", "HAS_CONTENT", "HAS_DATE", "MATCHES_SCAM_PATTERN"],
    "CONTENT": ["CONTAINS_PHONE_NUMBER", "CONTAINS_URL", "MENTIONS_KEYWORD", "MENTIONS_ORGANIZATION", "MENTIONS_PAYMENT_METHOD"],
    "SENDER": ["IMPERSONATES"],  # Links senders to organizations they pretend to be
    "RECIPIENT": [],
    "SUBJECT": [],
    "PHONE_NUMBER": [],
    "URL": [],
    "KEYWORD": [],
    "ORGANIZATION": [],
    "PAYMENT_METHOD": [],
    "DATE": [],
    "SCAM_PATTERN": []
}




def build_graph(path):
    start = time.time()

    documents = SimpleDirectoryReader(input_files=[path]).load_data()

    extractor = SchemaLLMPathExtractor(
        llm=Settings.llm,
        possible_entities=entities,
        possible_relations=relations,
        kg_validation_schema=schema,
        strict=False,
        num_workers=5,
    )

    index = PropertyGraphIndex.from_documents(
        documents=documents,
        kg_extractors=[extractor],
        property_graph_store=graph_store,
        embed_kg_nodes=True,
        embedding_model=Settings.embed_model,
        show_progress=True,
    )

    return index

def clear_graph():
    start = time.time()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    end = time.time()
    print(f"Graph cleared in {end-start:.2f} seconds")

def connect_chunks_lexically():
    start = time.time()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    get_chunks = """
    MATCH (n1:Chunk), (n2:Chunk)
    WHERE toInteger(n1.page_label) = toInteger(n2.page_label) - 1
    AND n1.file_name = n2.file_name
    MERGE (n1)-[:NEXT]->(n2)"""
    
    with driver.session() as session:
        session.run(get_chunks)    
    
    end = time.time()
    print(f"Chunks connected in {end-start:.2f} seconds")


def main():

    CLEAR_GRAPH = 1
    BUILD_GRAPH = 1
    CONNECT_CHUNKS = 1

    print(NEO4J_URI)

    if CLEAR_GRAPH:
        clear_graph()
        
    if BUILD_GRAPH:
        for file in os.listdir('datasets'):
            print(file, '-----------------')
            path = os.path.join('datasets', file)
            build_graph(path)

    if CONNECT_CHUNKS:
        connect_chunks_lexically()

if __name__ == "__main__":
    main()