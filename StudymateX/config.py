import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration management for StudyMate"""
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    
    # Groq Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    # IBM Granite Configuration
    IBM_GRANITE_USE_LOCAL = os.getenv("IBM_GRANITE_USE_LOCAL", "true").lower() == "true"
    IBM_GRANITE_DEVICE = os.getenv("IBM_GRANITE_DEVICE", "auto")
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_WORDS", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_WORDS", 100))
    BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH", 32))
    TOP_K = int(os.getenv("TOP_K", 3))
    MIN_SCORE = float(os.getenv("MIN_SCORE", 0.2))
    
    @classmethod
    def get_llm_config(cls):
        """Get LLM configuration based on provider"""
        return {
            "provider": cls.LLM_PROVIDER,
            "groq": {
                "api_key": cls.GROQ_API_KEY,
                "model": cls.GROQ_MODEL
            },
            "ibm_granite": {
                "use_local": cls.IBM_GRANITE_USE_LOCAL,
                "device": cls.IBM_GRANITE_DEVICE
            }
        }
