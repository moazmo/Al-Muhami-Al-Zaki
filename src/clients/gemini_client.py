"""
Google Gemini API client wrapper for Al-Muhami Al-Zaki.

Provides a configured Gemini client for answer generation and query rewriting.
"""

from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from src.utils.config import get_settings


_gemini_client: Optional[ChatGoogleGenerativeAI] = None


def get_gemini_client(
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> ChatGoogleGenerativeAI:
    """
    Get a configured Gemini client (for generation).

    Uses singleton pattern to reuse client across calls.

    Args:
        model: Model name (default from settings)
        temperature: LLM temperature (0.3 for balanced generation)

    Returns:
        Configured ChatGoogleGenerativeAI client
    """
    global _gemini_client

    settings = get_settings()
    model_name = model or settings.generator_model

    # Return cached client if available
    if _gemini_client is not None:
        return _gemini_client

    logger.info(f"Initializing Gemini client: {model_name}")

    _gemini_client = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        max_retries=3,
    )

    return _gemini_client


async def generate_answer(question: str, context: str) -> str:
    """
    Quick helper to generate a legal answer.

    Args:
        question: User's legal question
        context: Formatted context from relevant documents

    Returns:
        Generated answer with citations
    """
    from src.prompts.generator import get_generator_prompt

    client = get_gemini_client()
    prompt = get_generator_prompt(question, context)

    try:
        response = await client.ainvoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "عذراً، حدث خطأ أثناء إنشاء الإجابة. يرجى المحاولة مرة أخرى."


async def rewrite_query(question: str) -> str:
    """
    Quick helper to rewrite a query for better retrieval.

    Args:
        question: Original question that yielded no results

    Returns:
        Reformulated question
    """
    from src.prompts.rewriter import get_rewriter_prompt

    client = get_gemini_client(temperature=0.7)
    prompt = get_rewriter_prompt(question)

    try:
        response = await client.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Rewrite error: {e}")
        return question  # Fall back to original
