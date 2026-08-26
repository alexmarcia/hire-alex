# About this site (how I built it)

## Architecture
This site is a retrieval augmented generation (RAG) chatbot built with LangChain and Anthropic's Claude.

- Knowledge base: markdown files (resume, FAQ, projects, this page) are split into chunks with LangChain's MarkdownTextSplitter, embedded with Voyage AI embeddings, and stored in an in memory vector store. If no embedding key is configured the app falls back to a BM25 keyword retriever, so retrieval still works.
- Agent: a LangChain 1.x tool calling agent (createAgent) running on Claude through the @langchain/anthropic integration. The model decides when to call tools.
- Tools: search_background (semantic search over the knowledge base), assess_job_fit (retrieves relevant experience, has Claude tag every requirement in the posting as met, learnable, or gap using a Zod schema for structured output, then computes the relevancy percentage in code from a fixed, bucket based rubric: requirements are grouped into skill areas such as languages, ML and LLMs, data, cloud and DevOps, software engineering, domain, education, experience, and soft skills; each area is scored as a weighted average where required items weigh 3x and met earns full credit, learnable earns half, and gap earns none; the overall score is a weighted average of the areas, with core technical areas weighted highest, soft skills lowest, any area containing a required item counting double, and a 4 point penalty for every required hard gap; logistics like travel or location are excluded. Scoring by area means fifteen matched soft skills cannot outweigh three missing core requirements), and list_projects.
- Grounding: the system prompt only allows answers from retrieved chunks and tool results, and the UI shows which sources each answer came from.
- Backend: a Next.js API route on Vercel that holds the API keys server side, enforces per IP rate limits, and caps message and history size.
- Frontend: React with a first person chat UI.

## Why I built it this way
I wanted a real example of the stack that AI engineering roles ask for: RAG, LangChain, Anthropic tool use, and an agent loop, on a problem small enough to finish and deploy in a couple of weeks. Every answer is grounded in my actual resume so recruiters can trust it.
