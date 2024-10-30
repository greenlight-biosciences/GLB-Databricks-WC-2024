import logging
import os
import uuid
import time
import numpy as np
import cv2
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
    st.session_state.prev_image = None
    st.session_state.lg_msg = []
    st.session_state.db_msg = []
    st.session_state.latest_prompt = ""

if new_img := st.sidebar.file_uploader("Choose an image...", type=["png"], label_visibility=st.session_state.visibility):
    st.session_state.original_image = base64.b64encode(new_img.read()).decode("utf-8")
    if st.session_state.current_image is None: st.session_state.current_image = st.session_state.original_image
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

    st.session_state.prev_image = st.session_state.current_image

    with st.spinner("Processing Image"):
        updated_new_img, st.session_state.plan_json = agent.process_image(prompt, st.session_state.current_image, st.session_state.plan_json)
        # Decode the Base64 string to bytes
        image_bytes = base64.b64decode(updated_new_img)

        # Convert bytes data to a NumPy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Decode the image using OpenCV
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        success, encoded_image = cv2.imencode(".png", image)
        image_bytes = encoded_image.tobytes()

        with open("./new.png", 'wb') as f:
            f.write(image_bytes)
        time.sleep(1)

    with open("./new.png", "rb") as image_file:
        st.session_state.current_image = base64.b64encode(image_file.read()).decode("utf-8")
        
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
            if i == len(st.session_state.messages)-1 and 'filters' in msg.content[2]['data']:
                ftr, rsn = st.columns(2)
                filters = msg.content[2]['data']['filters']
                planner_reason = msg.content[2]['data']['planner_reason']
                # Generate sliders dynamically
                for j, filter_item in enumerate(filters):
                    ftr.markdown(f"<div class='centered'>{filter_item['name'].capitalize()} Filter</div>", unsafe_allow_html=True)
                    
                    filter_name = filter_item['name']
                        
                    with ftr.expander(f"{filter_name}"):
                        if filter_name == 'blur':
                            # Add Gaussian Blur slider for kernel size (must be odd)
                            key = 'kernel'
                            value = filter_item[key]
                            filter_item[key] = st.slider("Kernel Size", 1, 49, value, step=2, key=f"{key}+{i}{j}")
                        
                        elif filter_name == 'hsv':
                            key = 'lower_hsv'
                            value = filter_item[key]
                            # Add sliders for HSV lower and upper thresholds
                            st.text("Lower HSV:")
                            lower_h = st.slider("Lower H", 0, 180, value[0], key=f"lower_h-{i}{j}")
                            lower_s = st.slider("Lower S", 0, 255, value[1], key=f"lower_s-{i}{j}")
                            lower_v = st.slider("Lower V", 0, 255, value[2], key=f"lower_v-{i}{j}")
                            filter_item[key]= [lower_h, lower_s, lower_v]

                            st.text("Upper HSV:")
                            key = 'upper_hsv'
                            value = filter_item[key]
                            upper_h = st.slider("Upper H", 0, 180, value[0], key=f"upper_h-{i}{j}")
                            upper_s = st.slider("Upper S", 0, 255, value[1], key=f"upper_s-{i}{j}")
                            upper_v = st.slider("Upper V", 0, 255, value[2], key=f"upper_v-{i}{j}")
                            filter_item[key] = [upper_h, upper_s, upper_v]
                                
                        elif filter_name == 'canny':
                            # Add sliders for Canny Edge thresholds
                            
                            lower = st.slider("Lower Threshold", 0, 255, filter_item['t_lower'], key=f"canny_lower-{i}{j}")
                            filter_item['t_lower'] = lower
                            upper = st.slider("Upper Threshold", 0, 255, filter_item['t_upper'], key=f"canny_upper-{i}{j}")
                            filter_item['t_upper'] = upper

                # Display updated filters
                rsn.write(planner_reason)
                changed_img = agent._apply_filters(st.session_state.original_image, filters)
                rsn.image(f'data:image/png;base64,{changed_img}', caption='Updated Image')

                if st.button("Store this pipeline!", use_container_width=True, key=f"store_pipeline-{i}"):
                    push_data(filters, st.session_state.latest_prompt)