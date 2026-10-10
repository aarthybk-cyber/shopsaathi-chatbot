"""Core logic for Saathi, the ShopSaathi FAQ chatbot (no UI code here).

Steps: find the best FAQ -> escalate if the match is weak -> otherwise let a
Groq-hosted model rewrite the FAQ answer in a warm tone, using only that text.
"""
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL = "openai/gpt-oss-20b"
THRESHOLD = 0.12
ESCALATION = "Sorry, I'm not sure about that. Let me connect you to a support agent."

# Friendly replies for chit-chat. They are fixed text, so nothing can be made up.
# They are only used when no FAQ matches well, so real store questions always win.
SMALL_TALK = [
    ("greeting", r"\b(hi+|hello+|hey+|hiya|namaste|namaskar|good (morning|afternoon|evening))\b",
     "Hi there! 👋 I'm Saathi. Ask me about orders, delivery, returns, payments or offers."),
    ("identity", r"\b(your name|who are you|what are you|who (made|built|created) you|are you (a |an )?(bot|robot|human|real|ai|person))\b",
     "I'm Saathi, ShopSaathi's FAQ chatbot. I'm an AI assistant, not a person, and I answer from the store's FAQs."),
    ("how_are_you", r"\b(how are you|how r u|how do you do|how'?s it going)\b",
     "I'm doing great, thanks for asking! 😊 How can I help with your ShopSaathi order today?"),
    ("capabilities", r"\b(what can you do|what do you know|how can you help|can you help( me)?|help me|i need help)\b|^\s*help\s*[.!?]*\s*$",
     "I can help with tracking orders, delivery charges and times, returns and refunds, payments and EMI, coupons and gift cards. What would you like to know?"),
    ("thanks", r"\b(thanks+|thank you|thankyou|thx|ty)\b",
     "You're welcome! 😊 Is there anything else I can help you with?"),
    ("bye", r"\b(bye+|goodbye|see you|see ya|take care)\b",
     "Bye for now! 👋 Come back any time you have a question."),
]


def small_talk(query: str):
    """Return (intent, reply) for greetings and similar chit-chat, or None."""
    text = query.lower().strip()
    for intent, pattern, reply in SMALL_TALK:
        if re.search(pattern, text):
            return intent, reply
    return None


SYSTEM_PROMPT = """Role: You are ShopSaathi's warm and friendly customer support assistant.
Task: Answer the customer's query using only the provided FAQ answer context.
Rules:
1. Reply in a warm, friendly voice: start with a short greeting, answer in your own words using only the facts in the FAQ answer, and end with a short offer to help further. Keep it to 2-3 sentences.
2. Do not use any outside information or make up details; if the answer is not in the provided FAQ context, politely say "Sorry, I'm not sure about that. Let me connect you to a support agent." """


@dataclass
class FaqIndex:
    faqs: pd.DataFrame
    vectorizer: TfidfVectorizer
    vectors: object


def load_index(path) -> FaqIndex:
    """Read the FAQ CSV and build the TF-IDF search index."""
    faqs = pd.read_csv(path)
    search_text = faqs["question"] + " " + faqs["keywords"]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(search_text)
    return FaqIndex(faqs, vectorizer, vectors)


def find_best_faq(index: FaqIndex, query: str):
    """Return (best FAQ row, similarity score between 0 and 1)."""
    scores = cosine_similarity(index.vectorizer.transform([query]), index.vectors)[0]
    i = scores.argmax()
    return index.faqs.iloc[i], float(scores[i])


def answer(index: FaqIndex, query: str, client=None, model=DEFAULT_MODEL, threshold=THRESHOLD) -> dict:
    """Answer one customer question.

    Returns a dict with: reply, faq_id, category, matched_question, raw_answer,
    score, escalated, used_ai, error, small_talk (intent name, or None).
    """
    row, score = find_best_faq(index, query)
    result = {
        "reply": "",
        "faq_id": row["faq_id"],
        "category": row["category"],
        "matched_question": row["question"],
        "raw_answer": row["answer"],
        "score": score,
        "escalated": False,
        "used_ai": False,
        "error": None,
        "small_talk": None,
    }

    if score < threshold:  # nothing matches well
        chat = small_talk(query)
        if chat:  # greeting, thanks, "who are you"...: friendly fixed reply
            result["small_talk"], result["reply"] = chat
            return result
        # otherwise hand over to a human, no AI call
        result["reply"] = ESCALATION
        result["escalated"] = True
        return result

    if client is None:  # no API key: show the plain FAQ answer
        result["reply"] = row["answer"]
        return result

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context (FAQ Answer): {row['answer']}\nCustomer Query: {query}"},
            ],
            temperature=0.2,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("The model returned an empty reply.")
        result["reply"] = text
        result["used_ai"] = True
    except Exception as exc:  # e.g. retired model, bad key, network problem
        result["reply"] = row["answer"]  # fall back to the approved FAQ text
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
