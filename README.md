# Hire Alex: a RAG + agent hiring assistant (Python backend, React frontend)

A first person chatbot for recruiters, built to demonstrate the stack AI engineering roles ask for: retrieval augmented generation, LangChain, Anthropic tool use, and an agent loop, in Python. It answers only from my resume, cites its sources, and scores job descriptions with a fixed rubric.

## Architecture
```
Browser (React, Next.js)
   |  POST /api/chat  {messages}
   v
api/chat.py  (Python serverless function on Vercel)
   |  rate limit, input caps, keys server side
   v
LangChain create_agent  <->  Claude (langchain-anthropic)
   |  tools the model can call:
   |   search_background  -> RAG retrieval over /data/*.md
   |   assess_job_fit     -> retrieval + Claude structured output (Pydantic) + rubric scorer
   |   list_projects      -> projects.json
   v
Retriever: MarkdownTextSplitter -> Voyage embeddings -> InMemoryVectorStore
           (BM25 keyword fallback when embeddings are unavailable or rate limited)
```

## What each piece demonstrates
* RAG: `api/chat.py` section 2 chunks markdown by heading, embeds with Voyage AI, stores in a vector store, retrieves top chunks per query, and falls back to BM25 so retrieval never fails. Sources are shown under every answer.
* LangChain agent: section 4 uses LangChain 1.x `create_agent` with three `@tool` functions. The model decides when to search, when to score, and when to list projects.
* Anthropic: Claude runs the agent through `langchain-anthropic`. Switch models with `ANTHROPIC_MODEL`.
* Structured output plus rubric: Claude tags each posting requirement (category, priority, met/learnable/gap) with `with_structured_output(FitAssessment)`; `score_fit()` computes the percentage from a bucket based rubric so fifteen soft skills cannot outweigh three missing core requirements.
* Frontend: React chat UI with source chips, animated relevancy gauge and breakdown, suggested questions, architecture diagram, exit intent popup.

## Files
* `api/chat.py`: the whole backend (knowledge base, retrieval, rubric, tools, agent, HTTP handler).
* `requirements.txt`: Python dependencies for Vercel.
* `app/page.js`, `app/layout.js`, `app/globals.css`: the frontend.
* `data/*.md`, `data/projects.json`: the knowledge base. Edit these to change what the bot knows.

## Run locally
1. Python 3.11+ and Node 18+.
2. `pip install -r requirements.txt` and `npm install`.
3. Export `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` in your shell (or a .env you source).
4. Run the backend locally with `vercel dev` (installs via `npm i -g vercel`), which serves both the Next.js app and the Python function at http://localhost:3000.

## Deploy to Vercel
1. Push to GitHub (folders: api, app, data; files: package.json, next.config.mjs, vercel.json, requirements.txt, README.md).
2. Import into Vercel. It detects Next.js for the frontend and builds `api/chat.py` as a Python function from `requirements.txt`.
3. Add environment variables `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY`. Deploy.
4. Add a custom domain under Settings, Domains.

## Resume line
Built and deployed a RAG chatbot with a Python LangChain tool calling agent on Anthropic Claude (Vercel serverless, React/Next.js frontend): semantic retrieval over a markdown knowledge base with Voyage embeddings and BM25 fallback, Pydantic structured output feeding a rubric based job fit scorer, source citations, and rate limiting.
