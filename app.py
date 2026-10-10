"""Streamlit chat app for Saathi, the ShopSaathi FAQ chatbot.

Run locally:   streamlit run app.py
API key:       .streamlit/secrets.toml (GROQ_API_KEY = "...") or the GROQ_API_KEY environment variable.
"""
import os
import random
import time
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
GREETING = ("Hi! 👋 I'm **Saathi**, ShopSaathi's helper. Ask me about orders, delivery, "
            "returns, payments and more, or tap one of the questions below.")
BOT_AVATAR = "🛍️"
USER_AVATAR = "🙂"
TYPING_DELAY = 0.025  # seconds per word for the typing effect

st.set_page_config(page_title="Saathi | ShopSaathi FAQ chatbot", page_icon=BOT_AVATAR)

st.markdown(
    """
    <style>
    .bot-header {display:flex; align-items:center; gap:14px; padding:14px 18px; margin-bottom:12px;
                 border-radius:16px; color:#fff;
                 background:linear-gradient(135deg,#4F46E5 0%,#7C3AED 100%);}
    .bot-avatar {width:48px; height:48px; border-radius:50%; background:rgba(255,255,255,.2);
                 display:flex; align-items:center; justify-content:center; font-size:26px;}
    .bot-name {font-size:1.35rem; font-weight:700; line-height:1.2;}
    .bot-status {font-size:.85rem; opacity:.92;}
    .dot {display:inline-block; width:9px; height:9px; border-radius:50%; background:#4ADE80;
          margin-right:6px; box-shadow:0 0 0 0 rgba(74,222,128,.7); animation:pulse 2s infinite;}
    @keyframes pulse {70% {box-shadow:0 0 0 8px rgba(74,222,128,0);} 100% {box-shadow:0 0 0 0 rgba(74,222,128,0);}}
    [data-testid="stChatMessage"] {border-radius:16px; padding:.7rem 1rem;}
    div.stButton > button {border-radius:999px; text-align:left;}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def fresh_chat():
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    st.session_state.asked = 0
    st.session_state.handed_off = False


if "messages" not in st.session_state:
    fresh_chat()


def suggestions(meta):
    """Follow-up questions to offer as quick replies."""
    if meta and not meta["escalated"] and not meta["small_talk"] and not meta["off_topic"]:
        faqs = index.faqs
        same = faqs[(faqs["category"] == meta["category"]) & (faqs["faq_id"] != meta["faq_id"])]
        questions = same["question"].head(3).tolist()
        if questions:
            return questions
    return EXAMPLES[:3]


def transcript():
    return "\n\n".join(
        f"{'Customer' if m['role'] == 'user' else 'Saathi'}: {m['content']}" for m in st.session_state.messages
    )


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

    st.download_button("Download this chat", transcript(), file_name="saathi_chat.txt",
                       width="stretch")
    if st.button("Clear chat", width="stretch"):
        fresh_chat()
        st.rerun()

    st.divider()
    st.caption(
        "Privacy: your message is sent to a third-party AI service (Groq) to write the reply. "
        "Please do not type personal details such as phone numbers or order IDs."
    )

# ---------------- Main page ----------------
st.markdown(
    f"""<div class="bot-header"><div class="bot-avatar">{BOT_AVATAR}</div>
    <div><div class="bot-name">Saathi</div>
    <div class="bot-status"><span class="dot"></span>Online · answers only from ShopSaathi's FAQs</div></div></div>""",
    unsafe_allow_html=True,
)


with st.expander("Browse the FAQ knowledge base"):
    categories = sorted(index.faqs["category"].unique())
    chosen = st.multiselect("Filter by category", categories)
    table = index.faqs if not chosen else index.faqs[index.faqs["category"].isin(chosen)]
    st.dataframe(table[["faq_id", "category", "question", "answer"]], hide_index=True, width="stretch")


def render_details(meta):
    if meta["off_topic"]:
        st.caption("This is outside Saathi's job (shopping help), so no FAQ was used.")
        with st.expander("How Saathi answered"):
            st.write(f"**Type of question:** off topic ({meta['off_topic']})")
            st.write("Saathi only helps with ShopSaathi orders, delivery, returns, payments and offers.")
        return
    if meta["small_talk"]:
        st.caption("Friendly chit-chat, so no FAQ was needed for this reply.")
        with st.expander("How Saathi answered"):
            st.write(f"**Type of message:** {meta['small_talk'].replace('_', ' ')}")
            st.write(f"**Best FAQ match score:** {meta['score']:.4f} "
                     f"(below {saathi.THRESHOLD}, so no FAQ answer was used)")
        return
    if meta["ai_declined"]:
        st.caption("The closest FAQ does not answer this question, so it is handed to a human agent.")
    elif meta["escalated"]:
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


def render(i, message, is_last):
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        meta = message.get("meta")
        if meta:
            if not (meta["escalated"] or meta["small_talk"] or meta["off_topic"]):
                st.feedback("thumbs", key=f"feedback_{i}")
            render_details(meta)
    if message["role"] != "assistant" or not is_last:
        return
    # Quick actions under the latest bot message
    if meta and meta["escalated"] and not st.session_state.handed_off:
        if st.button("👩‍💼 Talk to a human agent", key=f"handoff_{i}"):
            ticket = f"SS-{random.randint(10000, 99999)}"
            st.session_state.handed_off = True
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Done! I've passed your question to our support team (reference **{ticket}**). "
                           "An agent will get back to you soon. *(This is a demo, so no real ticket is created.)*",
            })
            st.rerun()
    st.caption("Quick replies" if i else "Try asking")
    options = EXAMPLES if i == 0 else suggestions(meta)
    for n, question in enumerate(options):
        if st.button(question, key=f"chip_{i}_{n}"):
            st.session_state.pending = question


def typing(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(TYPING_DELAY)


last = len(st.session_state.messages) - 1
for i, message in enumerate(st.session_state.messages):
    render(i, message, is_last=(i == last))

prompt = st.chat_input("Ask Saathi a question...")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(prompt)
    if st.session_state.asked >= max_questions:
        reply = {"role": "assistant",
                 "content": "You have reached the question limit for this session. "
                            "Please use 'Clear chat' to start again, or contact a support agent."}
    else:
        st.session_state.asked += 1
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("Saathi is typing..."):
                result = saathi.answer(index, prompt, client=client, model=model)
            st.write_stream(typing(result["reply"]))
        reply = {"role": "assistant", "content": result["reply"], "meta": result}
    st.session_state.messages.append(reply)
    st.rerun()
