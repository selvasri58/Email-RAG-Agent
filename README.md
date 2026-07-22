# 📧 Email RAG Agent

A 100% free, locally-run, real-time email research agent.
Monitors your Gmail inbox via IMAP IDLE, embeds every message into a local
Qdrant vector database, and answers natural-language questions through a
LangGraph agent that intelligently routes between **semantic search** and
**live mailbox queries**.

## 🧱 Stack

| Layer            | Tool                                                  |
|------------------|-------------------------------------------------------|
| Orchestration    | LangGraph (Python)                                    |
| Vector DB        | Qdrant (Docker, local)                                |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` (local)     |
| LLM              | Groq + Llama 3.3 70B        |
| Email sync       | `IMAPClient` + IDLE for real-time push                |
| Interface        | Rich terminal chat loop                               |

## 🚀 Setup

### 1. Get credentials

- **Gmail App Password** — enable 2FA, then create one at
  https://myaccount.google.com/apppasswords
- **Gemini API key** — free key at https://aistudio.google.com/app/apikey

### 2. Configure

```bash
cp .env.example .env
# edit .env and fill EMAIL_ADDRESS, EMAIL_APP_PASSWORD, GOOGLE_API_KEY
```

### 3. Install deps

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Start Qdrant

```bash
docker compose up -d
```

Dashboard: http://localhost:6333/dashboard

## 🏃 Run (two terminals)

**Terminal 1 — ingestor**
```bash
python ingest.py
```
On first run this backfills your ENTIRE mailbox into Qdrant
(several minutes for a large inbox), then enters IMAP IDLE mode
and indexes new mail in real time.

**Terminal 2 — chat agent**
```bash
python main.py
```

## 💬 Example questions

| Question                                              | Tool the agent will pick    |
|-------------------------------------------------------|-----------------------------|
| `Did Stripe email me today?`                          | `fetch_live_emails`         |
| `Anything from amazon.com on 2025-05-30?`             | `fetch_live_emails`         |
| `Summarize the recruiter thread from Acme last week`  | `search_vector_db_with_filters` |
| `What did GitHub say about my last security alert?`   | `search_vector_db_with_filters` |
| `Did support@stripe.com email me yesterday, and what did they say?` | both tools in sequence |

## 🗂️ Layout

```
email-rag-agent/
├── docker-compose.yml      # local Qdrant
├── requirements.txt        # python deps
├── .env.example            # config template
├── common.py               # shared utils (config, parsing, embeddings)
├── ingest.py               # backfill + IDLE real-time listener
├── agent.py                # LangGraph routing + tools
└── main.py                 # terminal chat loop
```

## 🛠️ Troubleshooting

- **`LOGIN failed`** — make sure you're using a **Google App Password**, not your
  regular Gmail password, and that 2FA is enabled on the account.
- **Embeddings take forever first run** — the `all-MiniLM-L6-v2` model
  (~90 MB) is downloaded from Hugging Face the first time. Subsequent
  runs use the local cache (`~/.cache/huggingface`).
- **Qdrant connection refused** — confirm `docker compose ps` shows
  `email-rag-qdrant` as healthy, then check port 6333 isn't taken.
- **IDLE drops connection** — Gmail closes IDLE after ~29 min;
  `ingest.py` automatically refreshes every 14 min and reconnects on errors.
