"""Streamlit chat app for Saathi, the ShopSaathi FAQ chatbot.

Run locally:   streamlit run app.py
API key:       .streamlit/secrets.toml (GROQ_API_KEY = "...") or the GROQ_API_KEY environment variable.
"""
import os
from pathlib import Path

import streamlit as st

import saathi

DATA_PATH = Path(__file__).parent / "data" / "shopsaathi_faqs.csv"
EXAMPLES = [
    "How many days do I have to return a product?",
    "Is cash on delivery available?",
    "Do you offer EMI?",
    "Are gift cards available?",
    "Do you sell live pets?",
]
GREETING = "Hi! I'm Saathi, ShopSaathi's helper. Ask me about orders, delivery, returns, payments and more."

st.set_page_config(page_title="Saathi | ShopSaathi FAQ chatbot", page_icon="🛍️")


def secret(name, default=None):
    """Read a setting from Streamlit secrets, then the environment."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets file configured
        pass
    return os.environ.get(name, default)


@st.cache_resource
def get_index():
    return saathi.load_index(DATA_PATH)


@st.cache_resource
def get_client(api_key):
    if not api_key:
        return None
    from groq import Groq
    return Groq(api_key=api_key)


index = get_index()
client = get_client(secret("GROQ_API_KEY"))
model = secret("MODEL", saathi.DEFAULT_MODEL)
max_questions = int(secret("MAX_QUESTIONS", 25))

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    st.session_state.asked = 0

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Saathi")
    st.caption("An FAQ chatbot for the fictional online store ShopSaathi.")
    if client:
        st.success(f"AI replies on ({model})")
    else:
        st.info("No API key set: Saathi shows the plain FAQ answer.")
    st.metric("FAQs in the knowledge base", len(index.faqs))
    st.caption(f"Questions used this session: {st.session_state.asked}/{max_questions}")

    st.subheader("Try an example")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            st.session_state.pending = example

    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
        st.session_state.asked = 0
        st.rerun()

    st.divider()
    st.caption(
        "Privacy: your message is sent to a third-party AI service (Groq) to write the reply. "
        "Please do not type personal details such as phone numbers or order IDs."
    )

# ---------------- Main page ----------------
st.title("🛍️ Saathi")
st.write("Ask a question about ShopSaathi. Saathi answers only from the store's FAQ file, "
         "and hands you to a human agent when it is not sure.")

with st.expander("Browse the FAQ knowledge base"):
    categories = sorted(index.faqs["category"].unique())
    chosen = st.multiselect("Filter by category", categories)
    table = index.faqs if not chosen else index.faqs[index.faqs["category"].isin(chosen)]
    st.dataframe(table[["faq_id", "category", "question", "answer"]], hide_index=True, width="stretch")


def render(message):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        meta = message.get("meta")
        if not meta:
            return
        if meta["escalated"]:
            st.caption("No FAQ matched well, so this question is handed to a human agent.")
        if meta["error"]:
            st.warning("The AI reply failed, so the approved FAQ text is shown instead.")
        with st.expander("How Saathi answered"):
            st.write(f"**Matched FAQ:** {meta['faq_id']} ({meta['category']})")
            st.write(f"**Matched question:** {meta['matched_question']}")
            st.write(f"**Match score:** {meta['score']:.4f} (hand-over below {saathi.THRESHOLD})")
            st.write(f"**Raw FAQ answer:** {meta['raw_answer']}")
            if meta["error"]:
                st.code(meta["error"])


for message in st.session_state.messages:
    render(message)

prompt = st.chat_input("Ask Saathi a question...")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render(st.session_state.messages[-1])
    if st.session_state.asked >= max_questions:
        reply = {"role": "assistant",
                 "content": "You have reached the question limit for this session. "
                            "Please use 'Clear chat' to start again, or contact a support agent."}
    else:
        st.session_state.asked += 1
        with st.spinner("Saathi is thinking..."):
            result = saathi.answer(index, prompt, client=client, model=model)
        reply = {"role": "assistant", "content": result["reply"], "meta": result}
    st.session_state.messages.append(reply)
    render(reply)
    st.rerun()
