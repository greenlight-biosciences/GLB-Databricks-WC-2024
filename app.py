import logging
import os
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import base64
from langchain_core.messages import HumanMessage, AIMessage
from src.ImgProcessingAgent import agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.current_image = None

if new_img := st.sidebar.file_uploader("Choose an image...", type=["png"], label_visibility=st.session_state.visibility):
    st.session_state.current_image = base64.b64encode(new_img.read()).decode("utf-8")
    new_img = None
if st.session_state.current_image is not None:
    st.sidebar.image(f'data:image/png;base64,{st.session_state.current_image}', caption='Current Image')

chat_area, finetune = st.columns([4,1])

# Accept user input
if prompt := st.chat_input("How can i process your image?", disabled=(st.session_state.current_image==None)):
    
    input_message = HumanMessage(content=prompt)
    st.session_state.messages.append(input_message)

    # NOTE: Add Streamlit progress
    st.session_state.current_image, plan_json = agent.process_image(prompt, st.session_state.current_image)
    content = [
        {"type": "text", "text": "Processed Image"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{st.session_state.current_image}"},
        },
        {
         "type":"json", "data":plan_json
        }
    ]
    input_message = AIMessage(content=content)
    st.session_state.messages.append(input_message)
    st.rerun()

# Display chat messages from history on app rerun
for i in range(len(st.session_state.messages)):
    msg = st.session_state.messages[i]
    if isinstance(msg, HumanMessage):
        with st.chat_message("User"):
            if "image_url" in msg.content[1]:
                st.image(msg.content[1]["image_url"]["url"], caption='Uploaded Image')
            else:
                st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("AI"):
            if "image_url" in msg.content[1]:
                st.image(msg.content[1]["image_url"]["url"], caption='Processed Image')
            else:
                st.write(msg.content[0]["text"])
                
            # Custom CSS to center the filter name
            st.markdown(
                """
                <style>
                .centered {
                    text-align: center;
                    font-size: 20px;
                    font-weight: bold;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }
                </style>
                """, 
                unsafe_allow_html=True
            )
            if i == len(st.session_state.messages)-1:
                filters = msg.content[2]['data']['filters']
                # Generate sliders dynamically
                for filter_item in filters:
                    st.markdown(f"<div class='centered'>{filter_item['name'].capitalize()} Filter</div>", unsafe_allow_html=True)
                    
                    for key, value in filter_item.items():
                        if key == 'name': continue
                        with st.expander(f"{key}"):
                            if isinstance(value, list):
                                for j, v in enumerate(value):
                                    keyName = key.capitalize()+f"[{j}]"
                                    filter_item[key][j] = st.slider(keyName, 0, 255, v, key=keyName+f"-{i}")
                            elif isinstance(value, int):
                                filter_item[key] = st.slider(key.capitalize(), 1, 50, value, step=2, key=f"{i}")

                # Display updated filters
                st.sidebar.write("Updated Filters:", filters)
                changed_img = agent._apply_filters(st.session_state.current_image, filters)
                st.sidebar.image(f'data:image/png;base64,{changed_img}')

# Initialize the Databricks Workspace Client
# w = WorkspaceClient()

# Ensure environment variable is set correctly
# assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."
# # Display assistant response in chat message container
# with st.chat_message("assistant"):
#     # Query the Databricks serving endpoint
#     try:
#         response = w.serving_endpoints.query(
#             name=os.getenv("SERVING_ENDPOINT"),
#             messages=messages,
#             max_tokens=400,
#         )
#         assistant_response = response.choices[0].message.content
#         st.markdown(assistant_response)
#     except Exception as e:
#         st.error(f"Error querying model: {e}")