import logging
import os
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Databricks Workspace Client
w = WorkspaceClient()

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

user_info = get_user_info()

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.title("GreenLight Databricks PoC")
st.write(f"A basic chatbot using the your own serving endpoint")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for i in range(len(st.session_state.messages)+1):
    if i < len(st.session_state.messages):
        message = st.session_state.messages[i]
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message("U"):
            st.file_uploader("Choose an image...", type=["png"], label_visibility=st.session_state.visibility)

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [ChatMessage(role=ChatMessageRole.SYSTEM, content="You are a helpful assistant."),
                ChatMessage(role=ChatMessageRole.USER, content=prompt)]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Query the Databricks serving endpoint
        try:
            response = w.serving_endpoints.query(
                name=os.getenv("SERVING_ENDPOINT"),
                messages=messages,
                max_tokens=400,
            )
            assistant_response = response.choices[0].message.content
            st.markdown(assistant_response)
        except Exception as e:
            st.error(f"Error querying model: {e}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})