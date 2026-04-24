# GitHub (private) + Streamlit Community Cloud

## 1. Create a private GitHub repository

From this directory (`RAG-climate`), after the first commit exists:

```bash
cd "/path/to/RAG-climate"
gh auth login
gh repo create YOUR-ORG-OR-USER/rag-climate --private --source=. --remote=origin --push
```

If the repo already exists on GitHub:

```bash
git remote add origin https://github.com/YOUR-ORG-OR-USER/rag-climate.git
git branch -M main
git push -u origin main
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
