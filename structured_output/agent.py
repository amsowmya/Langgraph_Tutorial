from pydantic import BaseModel, Field
from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')


# DEFINE THE STRUCTURED OUTPUT

class WeatherResponse(BaseModel):
    """Response to the user"""
    temperature: float = Field(description="The temperature in fahreheit")
    wind_direction: str = Field(description="The direction of wind in abbreviated form")
    wind_speed: float = Field(description="The speed of the wind in km/h")
    

# DEFINE THE INPUT AND OUTPUT STATE

class AgentInput(TypedDict):
    pass 

class AgentOutput(TypedDict):
    # Final structured response from the agent
    final_response: WeatherResponse
    
class AgentState(MessagesState):
    # Final structured response from the agent
    final_response: WeatherResponse
    
# DEFINE THE TOOLS AND MODEL

@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information"""
    if city == "nyc":
        return "It is cloudy in NYC, with 5 mph winds in the North-East direction and a temperature of 70 degrees"
    elif city == "sf":
        return "It is 75 degrees and sunny in SF, with 3 mph in the South-East direction"
    else:
        raise AssertionError("Unknown city")
    
    
model = ChatGroq(groq_api_key=groq_api_key, model_name="qwen-2.5-32b")

tools = [get_weather]

model_with_tool = model.bind_tools(tools)
model_with_structured_output = model.with_structured_output(WeatherResponse)

def call_model(state: AgentState):
    response = model_with_tool.invoke(state['messages'])
    return {"messages": [response]}

def respond(state: AgentState):
    response = model_with_structured_output.invoke([HumanMessage(content=state['messages'][-2].content)])
    return {"final_response": response}


def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return "respond"
    else:
        return "continue"
    
    
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "respond": "respond"
    }
)

workflow.add_edge("tools", "agent")
workflow.add_edge("respond", END)

graph = workflow.compile()

# for event in graph.stream({"messages": [("user", "What is the weather in sf?")]}):
#     print(event)

# answer = graph.invoke(input={"messages": [("human", "what's the weather in SF?")]})[
#     "final_response"
# ]

# print(answer)

answer = graph.invoke(input={"messages": [("human", "what's the weather in SF?")]})
print(answer)
print("***********")
print(answer["final_response"])