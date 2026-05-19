import time
import os
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import List, Literal
import json
import uuid
import textwrap


## Config:
MODEL_NAME = "translategemma"
TEMPERATURE = 0.2  # Set to 0 for strict data tasks
REASONING = False  # Set to True to enable <think> blocks
# SYSTEM_MESSAGE is in --AGENT INITIALIZATION-- section
DATA_FOLDER = "Dictionary"


# --- Classes ---
 
# --- TOOLS ---
def load_dictionary(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Filter out the lines and empty spaces
        return {line.strip().lower() for line in f if line.strip()}

pl_path = os.path.join(DATA_FOLDER, "polski pl 10000.txt")
en_path = os.path.join(DATA_FOLDER, "english uk 5000.txt")
polish_words = load_dictionary(pl_path)
english_words = load_dictionary(en_path)

@tool
def dictionary_lookup(word: str, direction: Literal["pl_to_en", "en_to_pl"]) -> str:
    """Looks up a word in the local dictionary files to verify its existence."""
    word = word.lower()
    
    if direction == "pl_to_en":
        exists = word in polish_words
        status = "is in the approved Polish list" if exists else "is NOT in the approved list"
    else:
        exists = word in english_words
        status = "is in the approved English list" if exists else "is NOT in the approved list"
        
    return f"The word '{word}' {status}. Please provide the most natural translation based on this verification."

# --- AGENT INITIALIZATION ---
SYSTEM_MESSAGE = """You are a professional linguistic assistant. 
Your goal is to translate words between source and target language.
Produce only the target language translation, without any additional explanations or commentary.

"""

def get_agent():
    # ChatOllama supports structured_output via constrained decoding
    llm = ChatOllama(
        model=MODEL_NAME, 
        temperature=TEMPERATURE,
        reasoning=REASONING,
        # format="json" # Instructs Ollama to use JSON mode
    )
    memory = InMemorySaver()
    # In LangChain, passing response_format ensures the agent returns a typed object
    agent = create_agent(
        llm, 
        # tools=[dictionary_lookup],    # translategemma does not support tools
        system_prompt=SYSTEM_MESSAGE, 
        checkpointer=memory,
    )
    return agent


# --- EXECUTION ---


def run_translator():
    # 1. Initialize the agent
    agent = get_agent()
    
    # 2. Set up a thread ID for the session memory
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print("--- Local Dictionary Translator (Ollama) ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            # 3. Get user input
            user_query = input("User >> ").strip()
            
            if user_query.lower() in ["exit", "quit"]:
                break
            
            if not user_query:
                continue

            # 4. Call the agent
            # We use stream to see the reasoning steps or partial output
            input_message = HumanMessage(content=user_query)
            
            # Using stream for LangGraph agents to handle tool outputs and reasoning
            for event in agent.stream({"messages": [input_message]}, config):
                for node_name, output in event.items():
                    # If using JSON mode, the output might be structured
                    # This prints the content of the last AIMessage generated
                    if "messages" in output:
                        last_msg = output["messages"][-1]
                        if isinstance(last_msg, AIMessage):
                            print(f"\nAI: {last_msg.content}")
                            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_translator()
    