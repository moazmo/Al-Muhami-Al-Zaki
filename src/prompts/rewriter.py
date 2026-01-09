"""
Query rewriter prompt for search retry.

Used by Gemini to reformulate questions when initial retrieval fails.
"""

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage


REWRITER_SYSTEM_PROMPT = """أنت متخصص في تحسين استعلامات البحث القانوني.

## المهمة
أعد صياغة السؤال القانوني ليكون أكثر فعالية في البحث.

## استراتيجيات إعادة الصياغة

### 1. التوسيع
- إذا كان السؤال محدداً جداً، وسّعه
- مثال: "عقوبة السرقة ليلاً" ← "عقوبة السرقة وظروفها المشددة"

### 2. التخصيص
- إذا كان السؤال عاماً، خصصه
- مثال: "حقوق المرأة" ← "حقوق المرأة في الميراث"

### 3. المرادفات القانونية
- استخدم مصطلحات قانونية بديلة
- مثال: "فسخ العقد" ← "إنهاء العقد" أو "انحلال الرابطة التعاقدية"

### 4. ذكر القانون المحتمل
- أضف اسم القانون المتوقع
- مثال: "عقوبة القتل" ← "عقوبة القتل في قانون العقوبات"

## التعليمات
- أعد صياغة السؤال فقط (بدون شرح)
- حافظ على المعنى الأصلي
- اجعل الصياغة الجديدة باللغة العربية الفصحى
- السؤال المعاد صياغته يجب أن يكون جملة واحدة واضحة"""


def get_rewriter_prompt(question: str) -> List:
    """
    Build the rewriter prompt for Gemini.

    Args:
        question: Original question that yielded no results

    Returns:
        List of messages for the chat model
    """
    human_content = f"""## السؤال الأصلي الذي لم يحقق نتائج:
{question}

## السؤال المعاد صياغته:"""

    return [
        SystemMessage(content=REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
