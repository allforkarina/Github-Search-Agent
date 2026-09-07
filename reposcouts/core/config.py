# API Config to access the LLM models
# Date: 2026-09-06
# Author: zKarina

import os

from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek

def get_llm(thinking: bool = True) -> ChatDeepSeek:
    """
    Returns a ChatDeepSeek instance configured with the API key and model type.
    """

    thinking_type = (
        "enabled"
        if thinking
        else "disabled"
    )

    api_key = os.getenv("DEEPSEEK_API_KEY")
    # Default to deepseek-v4-flash if not set
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")  

    if not api_key:
        raise ValueError(
            "DeepSeek API key is not set."
            "Please configure in your .env file."
        )

    return ChatDeepSeek(
        model=model,
        api_key=api_key,
        temperature=0.0, 
        extra_body={
            "thinking": {
                "type": thinking_type
            }
        },
    )