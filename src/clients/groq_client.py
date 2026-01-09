"""
Groq API client wrapper for Al-Muhami Al-Zaki.

Provides a configured Llama-3 client for fast relevance grading.
"""

from typing import Optional

from langchain_groq import ChatGroq
from loguru import logger

from src.utils.config import get_settings


_groq_client: Optional[ChatGroq] = None


def get_groq_client(
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> ChatGroq:
    """
    Get a configured Groq client (Llama-3 for grading).

    Uses singleton pattern to reuse client across calls.

    Args:
        model: Model name (default from settings)
        temperature: LLM temperature (0.0 for deterministic grading)

    Returns:
        Configured ChatGroq client
    """
    global _groq_client

    settings = get_settings()
    model_name = model or settings.grader_model

    # Return cached client if settings match
    if _groq_client is not None:
        return _groq_client

    logger.info(f"Initializing Groq client: {model_name}")

    _groq_client = ChatGroq(
        model=model_name,
        api_key=settings.groq_api_key,
        temperature=temperature,
        max_retries=3,
    )

    return _groq_client


async def grade_relevance(question: str, document: str) -> bool:
    """
    Quick helper to grade document relevance.

    Args:
        question: User's legal question
        document: Document text to evaluate

    Returns:
        True if relevant, False otherwise
    """
    from src.prompts.grader import get_grader_prompt

    client = get_groq_client()
    prompt = get_grader_prompt(question, document)

    try:
        response = await client.ainvoke(prompt)
        grade = response.content.strip().lower()
        return "relevant" in grade and "irrelevant" not in grade
    except Exception as e:
        logger.error(f"Grading error: {e}")
        return True  # Fail-safe: include document on error
