"""Streamlit chat UI:  streamlit run app.py"""
import streamlit as st
 
from src.config import DEFAULT_DB_PATH, load_settings
from src.database import make_engine, seed_database
from src.llm import LLMConfigError, get_llm
from src.pipeline import CannotAnswer, NL2SQLPipeline, QueryError
 
st.set_page_config(page_title="Inventory Chatbot", page_icon="📦", layout="wide")
settings = load_settings()
 
EXAMPLES = [
    "Which products are below their reorder level?",
    "What is the total stock value in each warehouse?",
    "Top 5 products by units shipped in the last 30 days",
    "How many products does each supplier provide?",
    "Which Electronics items have zero stock in any warehouse?",
]
 
 
@st.cache_resource(show_spinner=False)
def load_engine():
    if settings.database_url.startswith("sqlite") and not DEFAULT_DB_PATH.exists():
        seed_database(DEFAULT_DB_PATH)  # first run: create the demo data
    return make_engine(settings.database_url)
 
 
@st.cache_resource(show_spinner=False)
def load_pipeline(token: str | None):
    llm = get_llm(settings.backend, settings.hf_model, token)
    return NL2SQLPipeline(llm, load_engine(), settings.dialect, settings.max_rows)
 
 
# ------------------------------- sidebar ------------------------------------ #
with st.sidebar:
    st.header("📦 Inventory Chatbot")
    st.caption(f"Model: `{settings.hf_model}` · backend: `{settings.backend}`")
    if settings.hf_token:
        token = settings.hf_token  # stays on the server, never sent to the browser
        st.success("Hugging Face token loaded from .env")
    else:
        token = st.text_input("Hugging Face token", type="password",
                              help="Or set HF_TOKEN in your .env file.") or None
    summarize = st.toggle("Explain results in plain English", value=True)
 
    st.subheader("Try asking")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            st.session_state["pending"] = example
 
    with st.expander("Database schema"):
        try:
            st.code(load_pipeline(token).schema, language="text")
        except LLMConfigError:
            st.caption("Add your token to load the schema view.")
 
    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.rerun()
 
# ------------------------------- chat --------------------------------------- #
st.title("Ask your inventory a question")
st.caption("Natural language → SQL → results. Demo data only; generated queries are read-only and validated.")
 
st.session_state.setdefault("messages", [])
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sql"):
            with st.expander("Generated SQL"):
                st.code(msg["sql"], language="sql")
        if msg.get("data") is not None:
            st.dataframe(msg["data"], width="stretch", hide_index=True)
 
question = st.chat_input("e.g. Which items need restocking?") or st.session_state.pop("pending", None)
 
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    history = [(m["question"], m["sql"]) for m in st.session_state["messages"]
               if m["role"] == "assistant" and m.get("sql")]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = load_pipeline(token).ask(question, history=history, summarize=summarize)
            content = result.answer or f"Found {len(result.data)} row(s)."
            st.markdown(content)
            with st.expander("Generated SQL"):
                st.code(result.sql, language="sql")
            st.dataframe(result.data, width="stretch", hide_index=True)
            st.session_state["messages"].append({
                "role": "assistant", "content": content, "sql": result.sql,
                "data": result.data, "question": question})
        except LLMConfigError as exc:
            st.error(str(exc))
        except CannotAnswer as exc:
            st.warning(str(exc))
            st.session_state["messages"].append({"role": "assistant", "content": str(exc)})
        except QueryError as exc:
            st.error(str(exc))
            if exc.sql:
                st.code(exc.sql, language="sql")