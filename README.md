# Saathi: an FAQ chatbot for ShopSaathi

Saathi is a small FAQ chatbot for a fictional Indian online store, **ShopSaathi**. It answers customer questions only from the store's own FAQ file, and hands over to a human when it is not sure.

## How it works

1. **Find**: the customer's question is compared with every FAQ (question + keywords) using TF-IDF and cosine similarity.
2. **Escalate**: if the best match scores below `0.12`, Saathi says it is not sure and offers a support agent. No AI call is made.
3. **Reply**: otherwise a Groq-hosted model rewrites the matched FAQ answer in a warm tone. The prompt tells it to use *only* that FAQ text (grounding), so it does not invent facts.

## Files

| File | Purpose |
|---|---|
| `ShopSaathi_Chatbot.ipynb` | The chatbot notebook (runs in Google Colab or Jupyter) |
| `data/shopsaathi_faqs.csv` | The knowledge base: 47 FAQs, 10 categories, columns `faq_id, category, question, answer, keywords` |
| `requirements.txt` | Python packages: `groq`, `pandas`, `scikit-learn` |

## Run it

1. Get a free API key at [console.groq.com](https://console.groq.com) (API Keys, then Create API Key).
2. **Colab:** upload the notebook and the CSV, open the key icon (Secrets), add a secret named `GROQ_API_KEY`, and switch on notebook access. **Elsewhere:** `pip install -r requirements.txt` and set the `GROQ_API_KEY` environment variable.
3. Run the cells from top to bottom. Without a key, Saathi still runs and shows the plain FAQ answer.

**Never put your API key in the notebook or commit it to Git.**

## Add a new FAQ

Add a row to `data/shopsaathi_faqs.csv` with all five columns, then rebuild the search:

```python
faqs, vectorizer, faq_vectors = build_index()
```

## Known limitations

- **Keyword matching:** the search compares shared words, so heavy typos or slang can match the wrong FAQ. Embedding-based search would handle meaning better.
- **Model names change:** Groq retires models over time. If you see a "decommissioned" error, run `client.models.list()` and update `MODEL` in the settings cell.
- **Privacy:** customer messages are sent to a third-party AI service. Do not send personal data (phone numbers, order details) without masking it.
- The FAQ content is example data for a fictional store (FAQ047, the gift cards entry, was written as an example).
