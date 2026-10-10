# Saathi: an FAQ chatbot for ShopSaathi

Saathi is a small FAQ chatbot for a fictional Indian online store, **ShopSaathi**. It answers customer questions only from the store's own FAQ file, and hands over to a human when it is not sure.

## How it works

1. **Find**: the customer's question is compared with every FAQ (question + keywords) using TF-IDF and cosine similarity.
2. **Escalate**: if the best match scores below `0.12`, Saathi says it is not sure and offers a support agent. No AI call is made.
3. **Reply**: otherwise a Groq-hosted model rewrites the matched FAQ answer in a warm tone. The prompt tells it to use *only* that FAQ text (grounding), so it does not invent facts.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web app (chat interface) |
| `.streamlit/config.toml` | App theme colour (safe to commit; never put keys here) |
| `saathi.py` | The chatbot logic used by the app: search, hand-over rule, Groq reply |
| `ShopSaathi_Chatbot.ipynb` | The same chatbot as a notebook (runs in Google Colab or Jupyter) |
| `data/shopsaathi_faqs.csv` | The knowledge base: 48 FAQs, 10 categories, columns `faq_id, category, question, answer, keywords` |
| `requirements.txt` | Python packages: `groq`, `pandas`, `scikit-learn`, `streamlit` |

## Run the notebook

1. Get a free API key at [console.groq.com](https://console.groq.com) (API Keys, then Create API Key).
2. **Colab:** upload the notebook and the CSV, open the key icon (Secrets), add a secret named `GROQ_API_KEY`, and switch on notebook access. **Elsewhere:** `pip install -r requirements.txt` and set the `GROQ_API_KEY` environment variable.
3. Run the cells from top to bottom. Without a key, Saathi still runs and shows the plain FAQ answer.

**Never put your API key in the notebook or commit it to Git.**

## Run the web app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Give the app your key in one of two ways:

- create `.streamlit/secrets.toml` (this file is git-ignored) containing `GROQ_API_KEY = "your-key"`, or
- set the `GROQ_API_KEY` environment variable.

Optional settings (same places): `MODEL` (default `openai/gpt-oss-20b`) and `MAX_QUESTIONS` (questions allowed per browser session, default `25`). Without a key the app still works and shows the plain FAQ answer. If the AI call fails (for example the model was retired), the app falls back to the approved FAQ text and says so.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (a private repo works once you let Streamlit access it).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and choose **Create app**.
3. Pick this repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings**, paste your secrets in TOML format, then click **Deploy**:

   ```toml
   GROQ_API_KEY = "your-groq-key"
   # MODEL = "openai/gpt-oss-20b"
   # MAX_QUESTIONS = 25
   ```

**Before you share the link:** anyone who opens a public app uses *your* Groq key and quota. Use a dedicated key, keep `MAX_QUESTIONS` low, restrict viewers under the app's **Share** settings, and revoke the key when you no longer need the app.

## Example

```text
Customer Query : how many days do i get to return something?
Matched FAQ    : FAQ012 | score 0.9123
Raw FAQ answer : You have 7 days from the delivery date to request a return for most products.
Saathi reply   : Hi there! You have 7 days from the delivery date to request a return for most
                 products. If you need anything else, just let me know.

Customer Query : Do you sell live pets like puppies?
Matched FAQ    : FAQ001 | score 0.0
Saathi         : Sorry, I'm not sure about that. Let me connect you to a support agent.
```

The second question is not in the FAQ file, so the score is below the threshold and Saathi escalates instead of guessing. (The matched FAQ ID is not used when the score is that low.)

## Project structure

```text
shopsaathi-chatbot/
├── app.py
├── saathi.py
├── ShopSaathi_Chatbot.ipynb
├── README.md
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── .gitignore
└── data/
    └── shopsaathi_faqs.csv
```

## Add a new FAQ

Add a row to `data/shopsaathi_faqs.csv` with all five columns, then rebuild the search:

```python
faqs, vectorizer, faq_vectors = build_index()
```

## Known limitations

- **Keyword matching:** the search compares shared words, so heavy typos or slang can match the wrong FAQ. Embedding-based search would handle meaning better.
- **Model names change:** Groq retires models over time. If you see a "decommissioned" error, run `client.models.list()` and update `MODEL` in the settings cell.
- **Privacy:** customer messages are sent to a third-party AI service. Do not send personal data (phone numbers, order details) without masking it.
- The FAQ content is example data for a fictional store (FAQ047, the gift cards entry, and FAQ048, what the store sells, were written as examples).
