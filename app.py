"""
Al-Muhami Al-Zaki — Streamlit Interface

The Intelligent Lawyer: A Corrective RAG system for Egyptian Law.
"""

import asyncio
from typing import List, Dict

import streamlit as st

from src.graph.builder import run_query
from src.graph.state import create_initial_state


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="المحامي الذكي | Al-Muhami Al-Zaki",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* RTL Support for Arabic */
    .stApp {
        direction: rtl;
    }
    
    /* Main title styling */
    .main-title {
        text-align: center;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #e0f2fe;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-right: 4px solid #0284c7;
    }
    
    .assistant-message {
        background-color: #f0fdf4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-right: 4px solid #16a34a;
    }
    
    /* Source card styling */
    .source-card {
        background-color: #fefce8;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #fde047;
    }
    
    /* Warning box */
    .disclaimer {
        background-color: #fef3c7;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f59e0b;
        margin-top: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<h1 class="main-title">⚖️ المحامي الذكي</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">مساعدك القانوني الذكي للقانون المصري</p>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar: Source Documents
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 المستندات المصدرية")
    st.markdown("---")

    if st.session_state.sources:
        for i, source in enumerate(st.session_state.sources, 1):
            with st.expander(
                f"المستند {i}: {source.get('article_number', 'غير محدد')}",
                expanded=False,
            ):
                st.markdown(f"**المصدر:** {source.get('source_name', 'غير معروف')}")
                st.markdown(f"**السنة:** {source.get('law_year', 'غير محدد')}")
                st.markdown(f"**درجة الملاءمة:** {source.get('score', 0):.2%}")
                st.markdown("---")
                st.markdown(source.get("text", "")[:500] + "...")
    else:
        st.info("ستظهر هنا المستندات القانونية المستخدمة في الإجابة")

    st.markdown("---")
    st.markdown("### ⚙️ الإعدادات")

    # Settings (future expansion)
    st.selectbox(
        "نوع القانون",
        ["جميع القوانين", "القانون المدني", "قانون العقوبات", "الدستور"],
        disabled=True,  # Enable when filters are implemented
    )

    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()


# -----------------------------------------------------------------------------
# Chat Interface
# -----------------------------------------------------------------------------
# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(
            f'<div class="user-message">👤 {content}</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="assistant-message">⚖️ {content}</div>', unsafe_allow_html=True
        )


# Input box
user_input = st.chat_input("اكتب سؤالك القانوني هنا...")

if user_input:
    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # Display user message
    st.markdown(
        f'<div class="user-message">👤 {user_input}</div>', unsafe_allow_html=True
    )

    # Run query through CRAG
    with st.spinner("جاري البحث في القوانين المصرية..."):
        try:
            # Run async query
            result = asyncio.run(run_query(user_input))

            answer = result.get("generation", "حدث خطأ في توليد الإجابة")

            # Extract sources
            sources = []
            for doc in result.get("graded_documents", []):
                sources.append(
                    {
                        "source_name": doc.metadata.get("source_name", ""),
                        "article_number": doc.metadata.get("article_number", ""),
                        "law_year": doc.metadata.get("law_year", ""),
                        "score": doc.metadata.get("score", 0),
                        "text": doc.page_content,
                    }
                )

            st.session_state.sources = sources

        except Exception as e:
            answer = f"عذراً، حدث خطأ: {str(e)}"
            st.session_state.sources = []

    # Add assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    # Display assistant message
    st.markdown(
        f'<div class="assistant-message">⚖️ {answer}</div>', unsafe_allow_html=True
    )

    # Rerun to update sidebar
    st.rerun()


# -----------------------------------------------------------------------------
# Footer Disclaimer
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="disclaimer">
    <strong>⚠️ تنبيه قانوني:</strong><br>
    هذا النظام هو أداة مساعدة للبحث القانوني ولا يعتبر بديلاً عن الاستشارة القانونية المتخصصة.
    جميع الإجابات مبنية على النصوص القانونية المتاحة في قاعدة البيانات.
    للحالات القانونية الجدية، يُرجى استشارة محامٍ مرخص.
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Footer Info
# -----------------------------------------------------------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔒 الخصوصية**")
    st.caption("متوافق مع قانون حماية البيانات 151/2020")

with col2:
    st.markdown("**📖 المصادر**")
    st.caption("القانون المدني - قانون العقوبات - الدستور")

with col3:
    st.markdown("**🤖 التقنية**")
    st.caption("Corrective RAG + LangGraph + Gemini")
