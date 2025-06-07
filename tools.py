from langchain_core.tools import tool
from typing import Literal
from config import Config
from knowledge_base import load_knowledge_base, search_knowledge_base
import json

KB_DATA = load_knowledge_base(Config.KNOWLEDGE_BASE_PATH)

@tool
def search_product_knowledge_base(query: str) -> list[dict]:
    """
    Searches the internal product knowledge base for answers to customer queries.
    Use this tool for questions about policies, shipping, product features, or general information.
    The query should be a concise summary of the user's question related to the knowledge base.
    """
    results = search_knowledge_base(query, KB_DATA)
    if results:
        return results
    else:
        return [{"title": "No relevant information found", "content": "The knowledge base does not contain information directly answering your query."}]

@tool
def request_human_handoff(reason: str) -> str:
    """
    Initiates a transfer to a human customer support agent.
    Use this tool when the AI cannot resolve the customer's issue, requires personal information,
    or if the customer explicitly requests to speak with a human.
    The 'reason' should concisely explain why a human handoff is needed.
    """
    print(f"\n--- Escalating to Human Agent ---")
    print(f"Reason for handoff: {reason}")
    print(f"Please provide relevant context to the human agent.")
    return "Your request has been escalated to a human agent. Please wait while we connect you."