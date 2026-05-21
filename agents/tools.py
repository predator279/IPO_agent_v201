# # agents/tools.py
# """
# Shared utilities for the IPO analysis agents:
#  - LangChain @tool wrappers for cross-agent use
#  - LLM factory for dynamic model switching (Groq, OpenRouter, HuggingFace)
#  - Simple invoke_model() helper
#  - check_models() sanity checker
# """
#
# import os
# import logging
# from typing import List, Dict, Any, Optional
#
# # Only import other agents **after** get_llm and invoke_model definitions
#
#
# # --- LangChain tool decorator ---
# from langchain.tools import tool
#
# # --- Agent logic imports ---
# from .ipo_details_agent import fetch_ipo_details
# from .sentiment_agent import analyze_sentiment as analyze_sentiment_logic
# from .rhp_agent import analyze_rhp as analyze_rhp_logic
# from .peer_agent import analyze_peers as analyze_peers_logic
#
# # --- Optional LLM providers ---
# try:
#     from langchain_groq import ChatGroq
# except Exception:
#     ChatGroq = None
# try:
#     from langchain_openai import ChatOpenAI
# except Exception:
#     ChatOpenAI = None
# try:
#     # from langchain_hub import HuggingFaceHub
#     from langchain_community.llms import HuggingFaceHub
# except Exception:
#     HuggingFaceHub = None
#
#
# # =============================================================================
# # 💬 LLM FACTORY
# # =============================================================================
# def get_llm(
#     model_name: Optional[str] = None,
#     provider: Optional[str] = None,
#     temperature: float = 0.25,
#     **kwargs,
# ):
#     """
#     Returns a LangChain-compatible chat LLM based on environment variables.
#     Supported providers: groq | openrouter | hf (huggingface)
#     Env vars:
#       LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE
#       GROQ_API_KEY / OPENROUTER_API_KEY / HUGGINGFACEHUB_API_TOKEN
#     """
#     provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
#     model_name = model_name or os.getenv("LLM_MODEL", "llama-3.3-70b")
#     temperature = float(os.getenv("LLM_TEMPERATURE", temperature))
#     logging.info(f"[tools.get_llm] provider={provider}, model={model_name}")
#
#     if provider in ("groq", "groq.ai"):
#         if ChatGroq is None:
#             raise RuntimeError("langchain-groq not installed (pip install langchain-groq)")
#         api_key = os.getenv("GROQ_API_KEY")
#         if not api_key:
#             raise RuntimeError("Missing GROQ_API_KEY in environment")
#         return ChatGroq(model=model_name, api_key=api_key, temperature=temperature, **kwargs)
#
#     if provider in ("openrouter", "or"):
#         if ChatOpenAI is None:
#             raise RuntimeError("langchain-openai not installed (pip install langchain-openai)")
#         os.environ.setdefault("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
#         api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise RuntimeError("Missing OPENROUTER_API_KEY or OPENAI_API_KEY")
#         return ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key, **kwargs)
#
#     if provider in ("hf", "huggingface", "huggingfacehub"):
#         if HuggingFaceHub is None:
#             raise RuntimeError("langchain-hub not installed (pip install langchain-hub)")
#         token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#         if not token:
#             raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN")
#         return HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token=token, temperature=temperature, **kwargs)
#
#     raise RuntimeError(f"Unknown LLM provider: {provider}")
#
#
# def check_models() -> Dict[str, bool]:
#     """Returns a dict showing which libraries and keys are available."""
#     return {
#         "langchain_groq": ChatGroq is not None,
#         "langchain_openai": ChatOpenAI is not None,
#         "langchain_hub": HuggingFaceHub is not None,
#         "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
#         "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
#         "HUGGINGFACEHUB_API_TOKEN": bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")),
#     }
#
#
# def invoke_model(llm, prompt_chain, **invoke_kwargs) -> str:
#     """
#     Unified way to call any LLM/chain. Works with prompt | llm pipelines.
#     """
#     try:
#         if hasattr(prompt_chain, "invoke"):
#             res = prompt_chain.invoke(invoke_kwargs)
#             if hasattr(res, "content"):
#                 return res.content
#             return str(res)
#         if hasattr(llm, "predict"):
#             return llm.predict(prompt_chain)
#         if hasattr(llm, "generate"):
#             gen = llm.generate([prompt_chain])
#             return gen.generations[0][0].text
#         return str(prompt_chain)
#     except Exception as e:
#         raise RuntimeError(f"invoke_model() failed: {e}")
#
#
# # =============================================================================
# # 🔧 LangChain @tools wrappers (your original content)
# # =============================================================================
#
# @tool
# def get_ipo_details_tool(ipo_name: str) -> dict:
#     """Fetches key IPO details such as GMP, price band, lot size, and dates."""
#     return fetch_ipo_details(ipo_name)
#
#
# @tool
# def analyze_sentiment_tool(ipo_names: List[str]) -> dict:
#     """Analyzes market sentiment for one or more IPOs."""
#     results = {}
#     for name in ipo_names:
#         results[name] = analyze_sentiment_logic(name)
#     return results
#
#
# @tool
# def query_rhp_tool(ipo_name: str) -> dict:
#     """Performs a deep analysis of the official Red Herring Prospectus (RHP)."""
#     return analyze_rhp_logic(ipo_name)
#
#
# @tool
# def compare_peers_tool(ipo_name: str, peer_names: List[str] = None) -> dict:
#     """Compares an IPO against peer companies."""
#     peers_to_analyze = peer_names or []
#     if not peers_to_analyze:
#         if "groww" in ipo_name.lower():
#             peers_to_analyze = ["ZEE Entertainment Enterprises"]
#         elif "tata" in ipo_name.lower():
#             peers_to_analyze = ["Bajaj Finance"]
#     if not peers_to_analyze:
#         return {"status": f"Could not automatically determine peers for {ipo_name}."}
#     return analyze_peers_logic(peers_to_analyze)
#

# agents/tools.py
"""
Shared LLM utilities for the IPO Analyzer agents.
This file should NOT import any other agent modules (to avoid circular imports).
"""

import os
import logging
from typing import Any, Dict, Optional

# --- Optional LLM providers ---
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None
try:
    from langchain_community.llms import HuggingFaceHub
except Exception:
    HuggingFaceHub = None


def get_llm(
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: float = 0.25,
    **kwargs,
):
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    model_name = model_name or os.getenv("LLM_MODEL", "llama-3.3-70b")
    temperature = float(os.getenv("LLM_TEMPERATURE", temperature))
    logging.info(f"[tools.get_llm] provider={provider}, model={model_name}")

    if provider in ("groq", "groq.ai"):
        if ChatGroq is None:
            raise RuntimeError("langchain-groq not installed")
        # api_key = os.getenv("GROQ_API_KEY") or "gsk....."
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY")
        return ChatGroq(model=model_name, api_key=api_key, temperature=temperature, **kwargs)

    if provider in ("openrouter", "or"):
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed")
        os.environ.setdefault("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")
        return ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key, **kwargs)

    if provider in ("hf", "huggingface", "huggingfacehub"):
        if HuggingFaceHub is None:
            raise RuntimeError("langchain-community not installed")
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN")
        return HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token=token, temperature=temperature, **kwargs)

    raise RuntimeError(f"Unknown LLM provider: {provider}")


def check_models() -> Dict[str, bool]:
    return {
        "langchain_groq": ChatGroq is not None,
        "langchain_openai": ChatOpenAI is not None,
        "langchain_huggingface": HuggingFaceHub is not None,
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
        "HUGGINGFACEHUB_API_TOKEN": bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")),
    }


# def invoke_model(llm, prompt_chain, **invoke_kwargs) -> str:
#     """Unified wrapper to invoke any LangChain LLM or chain."""
#     try:
#         if hasattr(prompt_chain, "invoke"):
#             res = prompt_chain.invoke(invoke_kwargs)
#             return getattr(res, "content", str(res))
#         if hasattr(llm, "predict"):
#             return llm.predict(prompt_chain)
#         if hasattr(llm, "generate"):
#             gen = llm.generate([prompt_chain])
#             return gen.generations[0][0].text
#         return str(prompt_chain)
#     except Exception as e:
#         raise RuntimeError(f"invoke_model() failed: {e}")

def invoke_model(llm, prompt_chain, **invoke_kwargs) -> str:
    """
    Unified wrapper to invoke any LangChain LLM or chain.
    Supports direct prompt, ChatPromptTemplate, or (system,user) tuples.
    """
    try:
        # Case 1: If it's a list of tuples -> treat as chat messages
        if isinstance(prompt_chain, (list, tuple)) and all(isinstance(x, (list, tuple)) for x in prompt_chain):
            messages = [{"role": role, "content": content} for role, content in prompt_chain]
            res = llm.invoke(messages)
            return getattr(res, "content", str(res))

        # Case 2: If it's a LangChain chain (like prompt | llm)
        if hasattr(prompt_chain, "invoke"):
            res = prompt_chain.invoke(invoke_kwargs)
            return getattr(res, "content", str(res))

        # Case 3: If it’s a plain text prompt
        if isinstance(prompt_chain, str):
            if hasattr(llm, "invoke"):
                res = llm.invoke(prompt_chain)
                return getattr(res, "content", str(res))
            if hasattr(llm, "predict"):
                return llm.predict(prompt_chain)
            if hasattr(llm, "generate"):
                gen = llm.generate([prompt_chain])
                return gen.generations[0][0].text

        # Fallback
        return str(prompt_chain)

    except Exception as e:
        raise RuntimeError(f"invoke_model() failed: {e}")

