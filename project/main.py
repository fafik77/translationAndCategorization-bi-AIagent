import time
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from langchain.agents import create_agent
import json
import textwrap


## Config:
MODEL_NAME = "qwen3.5:2b"
TEMPERATURE = 0.2
REASONING = True  # Set to False to enable <think> blocks
SYSTEM_MESSAGE = """
You are a helpful AI interface that translates and categorizes words. Use available tools to help the user.
Respond shortly in a format: 
  Translated from Language in => Language out
  (Word in, Word out) categories:[list of categories]
"""


# --- Classes ---

# --- TOOLS ---


# --- AGENT INITIALIZATION ---

def get_agent():
    llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE, model_kwargs={"think": REASONING})
    tools = []
    memory = InMemorySaver()
    agent = create_agent(llm, tools, system_prompt=SYSTEM_MESSAGE, checkpointer=memory)
    return agent

agent_executor = get_agent()
