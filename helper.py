from langchain_databricks.vectorstores import DatabricksVectorSearch
from langchain_databricks import DatabricksEmbeddings
from langchain_core.documents import Document
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import lit
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import os
import logging
import uuid
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def push_data(filters, task):
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

# def get_user_info():
#     headers = st.context.headers
#     return dict(
#         user_name=headers.get("X-Forwarded-Preferred-Username"),
#         user_email=headers.get("X-Forwarded-Email"),
#         user_id=headers.get("X-Forwarded-User"),
#     )

# user_info = get_user_info()
# Initialize the Databricks Workspace Client
# w = WorkspaceClient()

# Ensure environment variable is set correctly
# assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."
# Display assistant response in chat message container
# def query_serving_endpoint(messages):
#     response = w.serving_endpoints.query(
#         name=os.getenv("SERVING_ENDPOINT"),
#         messages=messages,
#         prompt="Summarize these messages"
#     )
#     assistant_response = response.choices.message.content
#     return assistant_response
# push_data("testing", "tasktest")

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
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, 'score_threshold': 0.5})
    results = retriever.invoke(search_value)
    similar_filters = ""
    if results:
        similar_filters = "These are filters developed for similar tasks previously, use these as examples:"
        for doc in results:
            similar_filters = similar_filters + f"\nUser's Query '{doc.page_content}' Corresponding filter {doc.metadata['filters']}"
    return similar_filters