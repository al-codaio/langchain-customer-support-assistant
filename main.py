import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Literal
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from config import Config
from tools import search_product_knowledge_base, request_human_handoff

load_dotenv()

AVAILABLE_TOOLS = {
    "search_product_knowledge_base": search_product_knowledge_base,
    "request_human_handoff": request_human_handoff,
}

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Customer Support Assistant Template")

# 1. Define Graph State
class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: A list of messages exchanged in the conversation.
        tools_used: A list of names of tools that have been used in the current turn.
        human_handoff_requested: A boolean indicating if a human handoff is requested.
        retry_count: An integer to track retry attempts for LLM calls.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    tools_used: List[str]
    human_handoff_requested: bool
    retry_count: int

# 2. Define Tools and LLM
tools = [search_product_knowledge_base, request_human_handoff]
llm = ChatOpenAI(model=Config.DEFAULT_LLM_MODEL, temperature=0)

# 3. Define the Agent Node (LLM with tool calling)
class ToolCallingAgent:
    def __init__(self, llm: ChatOpenAI, tools: list):
        self.llm = llm
        self.tools = tools
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful customer support assistant. Your goal is to resolve customer inquiries efficiently.
            You have access to the following tools: {tool_names}.

            If a tool is available that can answer the user's question, you MUST use it.
            If the user asks about policies, shipping, or general information, use the 'search_product_knowledge_base' tool.
            If you cannot resolve the issue with the available tools, or if the user explicitly asks to speak to a human,
            use the 'request_human_handoff' tool with a clear reason.
            Always try to use a tool before escalating to a human, unless a direct human request is made or the issue is clearly out of scope for automated tools.
            When using tools, ensure all required arguments are provided.
            Provide concise and helpful responses.
            """),
            ("placeholder", "{messages}"),
        ]).partial(tool_names=", ".join([t.name for t in tools]))
        self.runnable = self.prompt | self.llm.bind_tools(tools=self.tools)

    def __call__(self, state: AgentState):
        messages = state["messages"]
        try:
            response = self.runnable.invoke({"messages": messages})
            return {"messages": [response]}
        except Exception as e:
            print(f"LLM call failed: {e}")
            if state.get("retry_count", 0) < 2:
                print("Retrying LLM call...")
                return {"messages": [AIMessage("I'm having a bit of trouble understanding. Could you please rephrase your question?")]}
            else:
                return {"messages": [AIMessage("I apologize, but I'm unable to process your request at this moment. Let me connect you to a human agent.")],
                        "human_handoff_requested": True}

# 4. Define Nodes for the Graph
def custom_tool_execution_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    tool_outputs = []
    tools_executed = []

    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_call_id = tool_call['id']

        if tool_name not in AVAILABLE_TOOLS:
            error_msg = f"Error: The assistant tried to use an unknown tool: '{tool_name}'. This suggests an issue with the agent's logic."
            print(error_msg)
            tool_outputs.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id))
            tool_outputs.append(AIMessage(content="There was an error trying to use a tool. Let me connect you to a human agent."))
            return {"messages": tool_outputs, "human_handoff_requested": True}
        try:
            tool_func = AVAILABLE_TOOLS[tool_name]
            output = tool_func.invoke(tool_args)
            tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call_id))
            tools_executed.append(tool_name)
            if tool_name == "request_human_handoff":
                tool_outputs.append(AIMessage(content=f"Handoff requested. Reason: {tool_args.get('reason', 'No specific reason provided.')}. A human agent will be with you shortly."))
                return {"messages": tool_outputs, "human_handoff_requested": True}
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {e}. This issue requires human intervention."
            print(error_msg)
            tool_outputs.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id))
            tool_outputs.append(AIMessage(content=f"I encountered an error while trying to fulfill your request using a tool. Let me connect you to a human agent."))
            return {"messages": tool_outputs, "human_handoff_requested": True}

    return {"messages": tool_outputs, "tools_used": tools_executed}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Decides whether the agent should continue by calling tools or end the conversation.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    elif state.get("human_handoff_requested"):
        return END 
    else:
        return END

# 5. Build the LangGraph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", ToolCallingAgent(llm, tools))
workflow.add_node("execute_tools", custom_tool_execution_node)

# Set entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "execute_tools",
        END: END,
    },
)
workflow.add_edge("execute_tools", "agent")
graph = workflow.compile() 

# 6. Run the Assistant Locally
def run_assistant():
    print("Customer Support Assistant: Hello! How can I help you today? (Type 'exit' to quit)")
    thread_id = "user_123"

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Customer Support Assistant: Goodbye!")
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}
        for s in graph.stream(inputs, config={"configurable": {"thread_id": thread_id}}):
            if "__end__" not in s:
                print(s)
            else:
                final_state = s["__end__"]
                last_message = final_state["messages"][-1]
                print(f"Customer Support Assistant: {last_message.content}")
                if final_state.get("human_handoff_requested"):
                    print("Connecting you to a human agent now...")
                print("-" * 30)

# Comment out the lines below if you will be deploying to LangGraph Platform
# if __name__ == "__main__":
#     run_assistant()