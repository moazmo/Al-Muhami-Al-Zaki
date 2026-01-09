"""
Grader prompt for relevance scoring.

Used by Groq/Llama-3 to determine if a document is relevant
to the user's legal question.
"""

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage


GRADER_SYSTEM_PROMPT = """أنت مقيّم قانوني. مهمتك تحديد ما إذا كان المستند مرتبطاً بالسؤال.

## القاعدة البسيطة:
- إذا كان المستند يتحدث عن نفس الموضوع العام للسؤال = **relevant**
- إذا كان المستند يتحدث عن موضوع مختلف تماماً = **irrelevant**

## أمثلة:
- سؤال عن "الملكية" + مستند عن "العقود" أو "الحقوق" = relevant
- سؤال عن "الملكية" + مستند عن "الزواج" = irrelevant

## تعليمات مهمة:
- أجب بكلمة واحدة فقط: relevant أو irrelevant
- عند الشك، اختر relevant (أفضل أن نعطي معلومات إضافية)
- لا تكن صارماً جداً - أي ارتباط ولو بسيط يكفي"""


def get_grader_prompt(
    question: str,
    document: str,
) -> List:
    """
    Build the grader prompt for Llama-3.

    Args:
        question: User's legal question
        document: Document text to evaluate

    Returns:
        List of messages for the chat model
    """
    human_content = f"""## السؤال القانوني:
{question}

## المستند للتقييم:
{document}

## الحكم (relevant أو irrelevant فقط):"""

    return [
        SystemMessage(content=GRADER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
