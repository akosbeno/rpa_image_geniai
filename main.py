from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import base64
import httpx
from langgraph.graph import END, StateGraph, START
from typing import Annotated
import functools
from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage
from tools import tavily_tool, plot_line_graph, generate_csv

import operator
from typing import Annotated, Sequence, TypedDict

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

load_dotenv()

__tavily_api = os.getenv("TAVILY_API_KEY")
__api_key = os.getenv("AZURE_OPENAI_API_KEY")
__endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
__api_version = "2024-02-01"
__model = "gpt-4o"

llm = AzureChatOpenAI(
    api_version=__api_version,
    azure_deployment=__model,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=10,
    default_headers={"Ocp-Apim-Subscription-Key": __api_key},
    azure_endpoint=__endpoint
)

research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You should oversee a process I am trying to solve.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

chart_agent = create_agent(
    llm,
    [plot_line_graph, generate_csv],
    system_message="You should generate low code which I can insert into UI Path Studio and execute. In your responses only executable code should be present.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

from langgraph.prebuilt import ToolNode

tools = [tavily_tool, plot_line_graph, generate_csv]
tool_node = ToolNode(tools)

# Either agent can decide to end
from typing import Literal

# Define edge logic
# --------------------------------------------------------------------------
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"

workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
)

workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "chart_generator": "chart_generator",
    },
)
workflow.add_edge(START, "Researcher")
graph = workflow.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

print("Please enter the start message (end with an empty line):")
# lines = []
# while True:
#     line = input()
#     if line == "":
#         break
#     lines.append(line)
# start_message = "\n".join(lines)
img_path = "C:\\Users\\akos.beno\\Desktop\\rpa_genai\\class_cards_1.png"
with open(img_path, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode("utf-8")

message = HumanMessage(
    content=[
        {"type": "text", "text": "Given this image give me all the information form the MainteneceItem card, then end the conversation"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
)

for s in graph.stream(
    {"messages": [message]},
     # Maximum number of steps to take in the graph
    {"recursion_limit": 100},
):
    if "__end__" not in s:
        print(s)
        print("----")
