# GitHub (private) + Streamlit Community Cloud

Primary private repo for this pilot: **https://github.com/asaikiranb/climate-rag-streamlit** (`origin`).

The original **https://github.com/asaikiranb/RAG-climate** repo is kept as **`upstream`** (public community repo) without the pilot-only deploy bundle on `main`.

## 1. Clone / remotes

```bash
git clone https://github.com/asaikiranb/climate-rag-streamlit.git
# optional: git remote add upstream https://github.com/asaikiranb/RAG-climate.git
```

To create another private repo from a clean copy:

```bash
gh auth login
gh repo create YOUR-ORG-OR-USER/NEW-REPO-NAME --private --source=. --remote=origin --push
```

## 2. Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. **New app** → pick the **private** repo → branch `main`.
3. **Main file path:** `app.py`
4. **Python:** Cloud reads `runtime.txt` (here: Python 3.11.9).
5. **Secrets** (app settings → Secrets): at minimum set your Groq key:

   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

   Optional (only if you use these features):

   ```toml
   HF_TOKEN = "hf_..."
   TAVILY_API_KEY = "tvly_..."
   ```

6. **Advanced** → same settings as local if needed, e.g. `MODEL_DEVICE=cpu` (default in `config.py`).

### Notes

- First boot downloads **embedding / reranker** weights from Hugging Face; cold start can be **several minutes** and may hit free-tier timeouts. If deploy fails, try again after the cache warms, or use a paid workspace / smaller embedding model via env vars.
- **Ollama** is not available on Streamlit Cloud; generation uses **Groq** when `GROQ_API_KEY` is set (`llm.py`).
- Vector data committed with the repo: `qdrant_db_ci/`, `chroma_db/` (see `.gitignore`). Do not commit `.env`.

## 3. Updating the app

Push to `main`; Streamlit Cloud redeploys on each push (unless you pin a deployment).
