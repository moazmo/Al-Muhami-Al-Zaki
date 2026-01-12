"""
Node functions for Al-Muhami Al-Zaki CRAG graph.

Each function receives state and returns state updates.
Nodes:
- retrieve: Fetch documents from Qdrant
- grade_documents: Score relevance with Llama-3 (Groq)
- generate: Generate answer with Gemini
- rewrite_query: Reformulate query for retry
- no_answer: Return "not found" response
"""

from typing import Any, Dict, List

from langchain_core.documents import Document
from loguru import logger

from src.graph.state import GraphState
from src.prompts.grader import get_grader_prompt
from src.prompts.generator import get_generator_prompt
from src.prompts.rewriter import get_rewriter_prompt
from src.utils.config import get_settings


async def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve relevant documents from Qdrant.

    Uses the current question to search for top-k documents.

    Args:
        state: Current graph state with 'question'

    Returns:
        State update with 'documents' list
    """
    from src.ingest.embedder import get_embedder

    settings = get_settings()
    question = state["question"]

    logger.info(f"Retrieving documents for: {question[:50]}...")

    # Use cached embedder (singleton) to avoid model reload
    embedder = get_embedder()
    results = embedder.search(
        query=question,
        top_k=settings.retrieval_top_k,
    )

    # Convert to LangChain Documents
    documents = []
    for result in results:
        doc = Document(
            page_content=result.get("text", ""),
            metadata={
                "source_name": result.get("source_name", ""),
                "article_number": result.get("article_number", ""),
                "law_number": result.get("law_number", ""),
                "law_year": result.get("law_year", ""),
                "score": result.get("score", 0),
            },
        )
        documents.append(doc)

    logger.info(f"Retrieved {len(documents)} documents")

    return {"documents": documents}


async def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Grade document relevance using Llama-3 (Groq).

    Each document is scored for relevance to the question.
    Documents scoring above threshold are kept.

    Args:
        state: Current graph state with 'question' and 'documents'

    Returns:
        State update with 'graded_documents' and 'grade_decision'
    """
    from langchain_groq import ChatGroq

    settings = get_settings()
    question = state["question"]
    documents = state["documents"]

    if not documents:
        logger.warning("No documents to grade")
        return {
            "graded_documents": [],
            "grade_decision": "rewrite"
            if state["rewrite_count"] < settings.max_rewrite_attempts
            else "no_answer",
        }

    logger.info(f"Grading {len(documents)} documents")

    # Initialize Ollama client (local, unlimited)
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=settings.grader_model,
        temperature=0.0,
    )

    graded_documents = []

    for doc in documents:
        # Get grader prompt
        prompt = get_grader_prompt(
            question=question,
            document=doc.page_content,
        )

        try:
            response = await llm.ainvoke(prompt)

            # Parse response (expecting "relevant" or "irrelevant")
            grade = response.content.strip().lower()
            is_relevant = "relevant" in grade and "irrelevant" not in grade

            if is_relevant:
                graded_documents.append(doc)
                logger.debug(
                    f"Document RELEVANT: {doc.metadata.get('article_number', 'N/A')}"
                )
            else:
                logger.debug(
                    f"Document IRRELEVANT: {doc.metadata.get('article_number', 'N/A')}"
                )

        except Exception as e:
            logger.error(f"Grading failed: {e}")
            # On error, include document (fail-safe)
            graded_documents.append(doc)

    logger.info(f"Grading complete: {len(graded_documents)}/{len(documents)} relevant")

    return {"graded_documents": graded_documents}


async def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate answer using Gemini with relevant documents.

    Synthesizes an answer with proper legal citations.

    Args:
        state: Current graph state with 'question' and 'graded_documents'

    Returns:
        State update with 'generation' (the answer)
    """
    from langchain_ollama import ChatOllama

    settings = get_settings()
    question = state["question"]
    documents = state["graded_documents"]

    logger.info(f"Generating answer with {len(documents)} documents")

    # Initialize Ollama (local, unlimited)
    llm = ChatOllama(
        model=settings.generator_model,
        temperature=0.3,
    )

    # Format context from documents
    context_parts = []
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata
        citation = (
            f"[{metadata.get('source_name', 'مصدر غير معروف')}"
            f" - المادة {metadata.get('article_number', 'غير محدد')}"
            f" ({metadata.get('law_year', '')})]"
        )
        context_parts.append(f"المستند {i} {citation}:\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    # Get generator prompt
    prompt = get_generator_prompt(
        question=question,
        context=context,
    )

    try:
        response = await llm.ainvoke(prompt)
        generation = response.content

        logger.info("Answer generated successfully")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        generation = "عذراً، حدث خطأ أثناء إنشاء الإجابة. يرجى المحاولة مرة أخرى."

    return {"generation": generation}


async def rewrite_query(state: GraphState) -> Dict[str, Any]:
    """
    Rewrite the query for better retrieval.

    Uses Ollama to reformulate the question for a retry.

    Args:
        state: Current graph state with 'question'

    Returns:
        State update with new 'question', incremented 'rewrite_count',
        reset 'documents' and 'graded_documents'
    """
    from langchain_ollama import ChatOllama

    settings = get_settings()
    original_question = state["question"]

    logger.info(f"Rewriting query (attempt {state['rewrite_count'] + 1})")

    # Initialize Ollama (local, unlimited)
    llm = ChatOllama(
        model=settings.generator_model,
        temperature=0.7,
    )

    # Get rewriter prompt
    prompt = get_rewriter_prompt(original_question)

    try:
        response = await llm.ainvoke(prompt)
        new_question = response.content.strip()

        logger.info(f"Rewritten: {original_question[:30]}... -> {new_question[:30]}...")

    except Exception as e:
        logger.error(f"Rewrite failed: {e}")
        # Fall back to original question
        new_question = original_question

    return {
        "question": new_question,
        "rewrite_count": state["rewrite_count"] + 1,
        "query_history": state["query_history"] + [new_question],
        "documents": [],
        "graded_documents": [],
    }


async def no_answer(state: GraphState) -> Dict[str, Any]:
    """
    Return a "not found" response.

    Used when no relevant documents are found after max retries.

    Args:
        state: Current graph state

    Returns:
        State update with 'generation' containing the no-answer message
    """
    logger.warning("No relevant documents found - returning no_answer response")

    generation = (
        "عذراً، لم أتمكن من العثور على معلومات قانونية ذات صلة بسؤالك "
        "في قاعدة البيانات المتاحة.\n\n"
        "يرجى:\n"
        "1. إعادة صياغة السؤال بشكل أكثر تحديداً\n"
        "2. تحديد القانون أو المادة المطلوبة (إن أمكن)\n"
        "3. استشارة محامٍ متخصص للحالات المعقدة"
    )

    return {"generation": generation}
