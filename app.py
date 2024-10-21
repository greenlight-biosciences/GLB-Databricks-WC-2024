import logging
import os
import uuid
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import base64
from langchain_core.messages import HumanMessage, AIMessage
from ImgProcessingAgent import agent
from helper import push_data, dbrx_vs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(page_title="AdaptiveFilters", layout="wide")
os.environ['DATABRICKS_HOST'] = os.getenv("DATABRICKS_HOST_PROXY")

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.title("AdaptiveFilters: Image Processing Agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.plan_json = dict()
    st.session_state.original_image = None
    st.session_state.current_image = None
    st.session_state.lg_msg = []
    st.session_state.db_msg = []
    st.session_state.latest_prompt = ""

if new_img := st.sidebar.file_uploader("Choose an image...", type=["png"], label_visibility=st.session_state.visibility):
    st.session_state.original_image = base64.b64encode(new_img.read()).decode("utf-8")
    st.session_state.current_image = st.session_state.original_image
    new_img = None

if st.session_state.original_image is not None:
    st.sidebar.image(f'data:image/png;base64,{st.session_state.original_image}', caption='Original Image')

chat_area, finetune = st.columns([4,1])

# Accept user input
if prompt := st.chat_input("How can I process your image?", disabled=(st.session_state.current_image is None)):
    st.session_state.latest_prompt = prompt
    st.session_state.db_msg.append(ChatMessage(role=ChatMessageRole.USER, content=prompt))
    input_message = HumanMessage(content=prompt)
    st.session_state.messages.append(input_message)

    # NOTE: Add Streamlit progress
    with st.spinner("Processing Image"):
        st.session_state.current_image, st.session_state.plan_json = agent.process_image(prompt, st.session_state.current_image, st.session_state.plan_json)
        
    content = [
        {"type": "text", "text": "Processed Image"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{st.session_state.current_image}"},
        },
        {
         "type":"json", "data":st.session_state.plan_json
        }
    ]
    input_message = AIMessage(content=content)
    st.session_state.messages.append(input_message)
    # st.session_state.db_msg.append(ChatMessage(role=ChatMessageRole.SYSTEM, content=content))
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
                ftr, rsn = st.columns(2)
                filters = msg.content[2]['data']['filters']
                planner_reason = msg.content[2]['data']['planner_reason']
                # Generate sliders dynamically
                for filter_item in filters:
                    ftr.markdown(f"<div class='centered'>{filter_item['name'].capitalize()} Filter</div>", unsafe_allow_html=True)
                    
                    for key, value in filter_item.items():
                        if key == 'name': continue
                        with st.expander(f"{key}"):
                            if isinstance(value, list):
                                for j, v in enumerate(value):
                                    keyName = key.capitalize()+f"[{j}]"
                                    filter_item[key][j] = ftr.slider(keyName, 0, 255, v, key=keyName+f"-{i}{str(uuid.uuid4())}")
                            elif isinstance(value, int):
                                filter_item[key] = ftr.slider(key.capitalize(), 1, 50, value, step=2, key=f"{i}{str(uuid.uuid4())}")

                # Display updated filters
                rsn.write(planner_reason)
                changed_img = agent._apply_filters(st.session_state.current_image, filters)
                rsn.image(f'data:image/png;base64,{changed_img}', caption='Updated Image')
                if st.button("Store this pipeline!", use_container_width=True, key=str(uuid.uuid4())):
                    push_data(filters, st.session_state.latest_prompt)