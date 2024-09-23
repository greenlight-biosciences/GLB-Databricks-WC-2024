from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize the AzureOpenAI class with the model name
lc_llm = AzureChatOpenAI(
    model_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    openai_api_key=os.environ["AZURE_OPENAI_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_VERSION"],
)