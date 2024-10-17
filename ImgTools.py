import cv2
from typing_extensions import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Union
from langchain_core.output_parsers import JsonOutputParser

def gaussian_blur(img, kernel):
    res = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return res

def hsv_threshold(img, lower_hsv:tuple=(0, 0, 0), upper_hsv:tuple=(180, 255, 255)):
    lower_hsv, upper_hsv = tuple(lower_hsv), tuple(upper_hsv)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def canny_edge(img, t_lower:float, t_upper:float):
    edge_img = cv2.Canny(img, t_lower, t_upper)
    return edge_img

class GaussianBlur(BaseModel):
    """
    Gaussian Blur
    """
    name: Literal["blur"] = Field(description="The name of the filter, always 'blur'")
    kernel: int = Field(description="Kernel size for the gaussian blur. It is always an odd positive integer")

class HSVThreshold(BaseModel):
    """
    HSV Threshold
    """
    name: Literal["hsv"] = Field(description="The name of the filter, always 'hsv'")
    lower_hsv: tuple = Field(description="lower_hsv tuple for the HSV Threshold example : (0, 0, 0)")
    upper_hsv: tuple = Field(description="upper_hsv tuple for the HSV Threshold example : (180, 255, 255)")

class CannyEdge(BaseModel):
    """
    Canny Edge Detection
    """
    name: Literal["canny"] = Field(description="The name of the filter, always 'canny'")
    t_lower: float = Field(description="t_lower float value")
    t_upper: float = Field(description="t_upper float value")

image_tools = {
    'blur': gaussian_blur,
    'hsv': hsv_threshold,
    # 'canny': canny_edge
}

class Filters(BaseModel):
    """Filters Json"""
    filters: List[Union[HSVThreshold, GaussianBlur]] = Field(description="List of filters available to apply")
    planner_reason: str = Field(description="Provide a human readable formatted concise definition of each filter used, and explain what each parameter for each filter does when increased or decreased. Present this information in bullet points. Additionally, recommend which parameters I should tweak, and explain the expected effects of those changes. Briefly explain the rationale behind the sequence of filters.")
parser = JsonOutputParser(pydantic_object=Filters)