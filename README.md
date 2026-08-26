# Hire Alex: a RAG + agent hiring assistant

A first person chatbot for recruiters, built to demonstrate the stack AI engineering roles ask for: retrieval augmented generation, LangChain, Anthropic tool use, and an agent loop. It answers only from my resume, cites its sources, and scores pasted job descriptions.

## Architecture
```
Browser (React, Next.js)
   |  POST /api/chat  {messages}
   v
Next.js API route (Vercel, Node runtime)
   |  rate limit, input caps, key stays server side
   v
LangChain createAgent  <->  Claude (@langchain/anthropic)
   |  tools the model can call:
   |   search_background  -> RAG retriever over /data/*.md
   |   assess_job_fit     -> retrieval + Claude structured output (Zod schema)
   |   list_projects      -> projects.json
   v
Retriever: MarkdownTextSplitter -> Voyage embeddings -> MemoryVectorStore
           (falls back to BM25 keyword retrieval when no embedding key is set)
```

## What each piece demonstrates
* RAG: `lib/rag.js` chunks markdown by heading, embeds with Voyage AI, stores in a vector store, and retrieves the top chunks per query. Sources are returned to the UI and shown under every answer.
* LangChain agent: `lib/agent.js` uses LangChain 1.x `createAgent` with three `tool()` definitions. The model decides when to search, when to score a job, and when to list projects.
* Anthropic: Claude runs the agent through `@langchain/anthropic`. Switch models with `ANTHROPIC_MODEL`.
* Structured output: job fit scoring uses `withStructuredOutput` and a Zod schema so score, matches, and gaps come back typed.
* Grounding and safety: the system prompt forbids answering from memory, the UI shows retrieval sources, and the route enforces per IP rate limits and size caps.

## Files
* `app/page.js`: chat UI with source chips and a "How this site works" panel.
* `app/api/chat/route.js`: request validation, rate limiting, calls the agent.
* `lib/rag.js`: loading, chunking, embedding, retrieval.
* `lib/agent.js`: system prompt, tools, agent construction.
* `data/*.md`: the knowledge base (resume, FAQ, projects, about this site). Edit these to change what the bot knows.
* `data/projects.json`: structured projects for the `list_projects` tool.

## Run locally
1. Node 18 or newer.
2. `npm install`
3. Copy `.env.example` to `.env.local`. Add `ANTHROPIC_API_KEY`. Add `VOYAGE_API_KEY` (free tier at voyageai.com) to enable vector search; without it the app uses BM25 keyword retrieval.
4. `npm run dev` and open http://localhost:3000

## Deploy to Vercel (free)
1. Push this folder to a new GitHub repo.
2. On vercel.com choose New Project, import the repo (Next.js is detected).
3. Add environment variables: `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, optionally `ANTHROPIC_MODEL`.
4. Deploy. You get a `*.vercel.app` URL immediately.
5. Custom domain: Settings, Domains, add your domain, then set the DNS records shown at your registrar.

## Cost
* Vercel hobby: free. Domain: about $10 to $15 a year.
* Anthropic API: pennies per conversation (`claude-haiku-4-5` for lowest cost).
* Voyage embeddings: free tier covers a personal site many times over. The index is tiny and rebuilds on cold start.

## Resume line
Built and deployed a RAG chatbot with a LangChain tool calling agent on Anthropic Claude (Next.js, Vercel): semantic retrieval over a markdown knowledge base with Voyage embeddings, structured output job fit scoring with Zod, source citations, and server side rate limiting.

## Extend it
Add a `tool()` in `lib/agent.js`. Ideas: fetch GitHub repos live, email a recruiter's message to yourself, book a call through a scheduling link, or swap MemoryVectorStore for Pinecone or pgvector to show a hosted vector database.
