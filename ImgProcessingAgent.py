import json
from typing import Dict, List, Tuple, Union
from langgraph.graph import END, StateGraph
import numpy as np
from config import lc_llm
from langchain_core.prompts import ChatPromptTemplate
import base64
from typing_extensions import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field, Json
from langchain_core.messages import HumanMessage
from langgraph.graph.message import AnyMessage, add_messages
from ImgTools import parser, image_tools
import cv2
import yaml
from helper import dbrx_vs
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = {"configurable": {"thread_id": "1"}}

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    image: bytes
    iterations: int
    desc: str
    processed_image: bytes
    plan_json: Json
    is_correct: bool
    critique_reason: str
    planner_reason: str  

class CritiqueOutput(BaseModel):
    """Critique output"""
    is_correct: bool = Field(description="True if the image addresses the user's task description, otherwise it is False")
    reason: str = Field(description="Provide a detailed explanation of why the final image does or does not meet the user's task requirements, including an analysis of how the filters were applied and any discrepancies or improvements that can be made.")

# Defining the structure for the image processing agent
class ImageProcessingAgent:
    def __init__(self, image_tools=None):
        self.llm = lc_llm
        self.image_tools = image_tools
        self.graph = self._create_graph()
        self.max_retries = 5
    
    def _create_graph(self):
        # Create a LangGraph workflow
        workflow = StateGraph(State)

        # Define the nodes
        workflow.add_node("planner", self._planner) 
        workflow.add_node("executor", self._executor) 
        workflow.add_node("critique", self._critique) 
        
        # Build graph
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "critique")
        workflow.add_conditional_edges(
            "critique",
            self.decide_to_finish,
            {
                "end": END,
                "planner": "planner",
            },
        )
        return workflow.compile()

    def _planner(self, state: Dict):
        # This is the planner agent which outputs a plan for the image processing pipeline
        if state['plan_json']: system_msg = f"""As an expert in image processing, use the user's description to create or modify a JSON object that defines a sequence of filters. Form a sequence of filters out of the available filters: {image_tools.keys()}, repeatitions of filters with different parameter values are allowed, to achieve the desired outcome. Follow the advice of the critiquer, if any. The current filters used is\n'{yaml.dump(state['plan_json'])}'\n** ALWAYS ANSWER IN THE JSON FORMAT PROVIDED, NO CONTENT SHOULD BE OUT OF THE JSON. Strictly adhere to the provided structure with no extra text or comments before or after the json. Your output should always start and end with curly braces **"""  + "\n{format_instructions}\n"
        else:
            system_msg = f"""As an expert in image processing, use the user's description to create or modify a JSON object that defines a sequence of filters. Form a sequence of filters out of the available filters: {image_tools.keys()}, repeatitions of filters with different parameter values are allowed, to achieve the desired outcome. Follow the advice of the critiquer, if any. ** ALWAYS ANSWER IN THE JSON FORMAT PROVIDED, NO CONTENT SHOULD BE OUT OF THE JSON. Strictly adhere to the provided structure with no extra text or comments before or after the json. Your output should always start and end with curly braces **"""  + "\n{format_instructions}\n"

        planning_agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"""{system_msg}"""),
                ("placeholder", "{messages}")
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        plan_agent = planning_agent_prompt | lc_llm

        if state['iterations'] == 0:
            task = f"""{state['desc']}"""
            examples = dbrx_vs(state['desc'])
            task += f"\n{examples}"
            state['messages'] = [("user", [
                {
                    "type": "text", "text": f"{task}"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{state['image']}"},
                },
            ])]
        result = plan_agent.invoke({"messages": state['messages']})
        try:
            logger.info(result.content)
            plan_json = json.loads(result.content.strip('```json').strip('```'))
            # print(plan_json)
            state['plan_json'] = plan_json
            planner_msg = [("ai", f"""{plan_json}""")]
            state['messages'] = planner_msg
        except Exception as e:
            # print(e, result)
            logger.info(e)
        state["iterations"] += 1
        return state
    
    def _apply_filters(self, image, filters):
        # Function to apply filters
        # Decode the Base64 string to bytes
        image_bytes = base64.b64decode(image)

        # Convert bytes data to a NumPy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Decode the image using OpenCV
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        for filter in filters:
            name = filter['name']  # Remove 'name' from the dictionary
            if name in image_tools.keys():
                params = {k: v for k, v in filter.items() if k != 'name'}
                image = self.image_tools[name](image, **params)  # Pass the rest of the dictionary as params
        
        success, encoded_image = cv2.imencode(".png", image)
        image_bytes = encoded_image.tobytes()

        with open("./dummy.png", 'wb') as f:
            f.write(image_bytes)
        with open("./dummy.png", "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        return image_data

    def _executor(self, state: Dict):
        # This agent applies filters according to the plan
        image = state['image']
        try:
            filters = state['plan_json']["filters"]

            image_data = self._apply_filters(image, filters)
            
            state['processed_image'] = image_data

            state['messages'] = [("user", [
                {
                    "type": "text", "text": f"Please review this image as per the user's task description\n#{state['desc']}#"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{state['processed_image']}"},
                },
            ])]
        except Exception:
            state['plan_json'] = dict()
            state['processed_image'] = state['image']
        
        return state
    
    def _critique(self, state: Dict):
        # The critiquer agent validates the output plan with the user's request
        filters = parser.get_format_instructions().split("Here is the output schema:")[1]
        system_prompt = """Your task is to critique the latest filter JSON generated by the planner agent. The JSON outlines the sequence of filters and parameters applied to an image based on the user's task description. \
            Evaluate whether the selected filters and their parameters precisely address the user's requirements. If they do not, clearly specify what must be changed in the JSON. \
            Your response must strictly adhere to this format and focus solely on the critique. Note: The image processing agent can only use the filters provided {avail_filters}"""

        critique_agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"""{system_prompt}"""),
                ("placeholder", "{messages}")
            ]
        ).partial(avail_filters=filters)

        replan_agent = critique_agent_prompt | lc_llm.with_structured_output(CritiqueOutput)
        result = replan_agent.invoke({"messages": state['messages']})
        # print(result)
        logger.info(result)
        if result.is_correct == False:
            critiquer_msg = [("ai", f"""This plan is wrong. Adjust according to this advise.\n{result.reason}""")]
        else:
            critiquer_msg = [("ai", f"""The correctness of this filter Json is {result.is_correct} because\n{result.reason}""")]
        state['messages'] = critiquer_msg
        state['is_correct'] = result.is_correct
        return state

    def decide_to_finish(self, state: Dict):
        # Termination condition
        iterations = state["iterations"]
        is_correct = state["is_correct"]
        
        if iterations == self.max_retries or is_correct == True:
            # print(f"---DECISION: FINISH---iter: {iterations}/{self.max_retries}")
            logger.info(f"---DECISION: FINISH---iter: {iterations}/{self.max_retries}")
            return "end"
        else:
            # print(f"---DECISION: RE-TRY SOLUTION---iter: {iterations}/{self.max_retries}")
            logger.info(f"---DECISION: RE-TRY SOLUTION---iter: {iterations}/{self.max_retries}")
            return "planner"
    
    def process_image(self, desc, img, plant_json):
        final_state = self.graph.invoke({"desc": desc, "image": img, "iterations": 0, "plan_json": plant_json}, config=config)
        return final_state["processed_image"], final_state["plan_json"]

# Create the Image Processing Agent
agent = ImageProcessingAgent(image_tools)