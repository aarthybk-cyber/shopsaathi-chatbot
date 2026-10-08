"""Core logic for Saathi, the ShopSaathi FAQ chatbot (no UI code here).

Steps: find the best FAQ -> escalate if the match is weak -> otherwise let a
Groq-hosted model rewrite the FAQ answer in a warm tone, using only that text.
"""
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL = "openai/gpt-oss-20b"
THRESHOLD = 0.12
ESCALATION = "Sorry, I'm not sure about that. Let me connect you to a support agent."

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
    score, escalated, used_ai, error.
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
    }

    if score < threshold:  # nothing matches well: hand over to a human, no AI call
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
