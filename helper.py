from langchain_databricks.vectorstores import DatabricksVectorSearch
from langchain_databricks import DatabricksEmbeddings
from langchain_core.documents import Document
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import lit
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import os
import yaml
import logging
import uuid
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def push_data(filters, task):
    logger.info(f"Pushing data to vector store: {filters}, {task}")
    endpoint_name = "vector-search-glb-endpoint"
    index_name = "workspace.default.glb_index_direct"
    embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
    vector_store = DatabricksVectorSearch(
        endpoint=endpoint_name,
        index_name=index_name,
        embedding=embeddings,
        text_column="text",
    )
    document_1 = Document(page_content=task, metadata={"filters": str(filters)})

    documents = [document_1]

    vector_store.add_documents(documents=documents, ids=[str(uuid.uuid4())])

def dbrx_vs(search_value):
    endpoint_name = "vector-search-glb-endpoint"
    index_name = "workspace.default.glb_index_direct"
    embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
    vector_store = DatabricksVectorSearch(
        endpoint=endpoint_name,
        index_name=index_name,
        embedding=embeddings,
        text_column="text",
        columns=['id', 'text', 'filters']
    )
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, 'score_threshold': 0.7})
    results = retriever.invoke(search_value)
    similar_filters = ""
    if results:
        similar_filters = "These are filters developed for similar tasks previously, use these as examples:"
        for doc in results:
            data = eval(doc.metadata['filters'])
            yaml_data = yaml.dump(data, sort_keys=False)
            similar_filters = similar_filters + f"\nUser's Query '{doc.page_content}' Corresponding filter\n{yaml_data}"
    return similar_filters