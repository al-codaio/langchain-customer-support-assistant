class Config:
    KNOWLEDGE_BASE_PATH = "knowledge_base.json" 
    HUMAN_HANDOFF_THRESHOLD = 0.7 # Confidence score below which to hand off to human
    DEFAULT_LLM_MODEL = "gpt-4o-mini"