from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv(override=True)

# Initialize the AzureOpenAI class with the model name
lc_llm = AzureChatOpenAI(
    model_name=os.getenv["AZURE_OPENAI_DEPLOYMENT"],
    azure_endpoint=os.getenv["AZURE_OPENAI_ENDPOINT"],
    deployment_name=os.getenv["AZURE_OPENAI_DEPLOYMENT"],
    openai_api_key=os.getenv["AZURE_OPENAI_KEY"],
    openai_api_version=os.getenv["AZURE_OPENAI_VERSION"],
)