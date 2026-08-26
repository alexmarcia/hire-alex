import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { MarkdownTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { VoyageEmbeddings } from "@langchain/community/embeddings/voyage";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";

const DATA_DIR = path.join(process.cwd(), "data");
const SOURCE_LABELS = {
  "resume.md": "Resume",
  "faq.md": "FAQ",
  "projects.md": "Projects",
  "about-this-site.md": "About this site",
};

// Load every markdown file in /data and split it into overlapping chunks.
// Each chunk keeps its source file and nearest heading so the UI can cite it.
async function loadChunks() {
  const splitter = new MarkdownTextSplitter({ chunkSize: 700, chunkOverlap: 100 });
  const chunks = [];
  for (const file of readdirSync(DATA_DIR).filter((f) => f.endsWith(".md"))) {
    const text = readFileSync(path.join(DATA_DIR, file), "utf8");
    const pieces = await splitter.splitText(text);
    for (const piece of pieces) {
      const heading = (piece.match(/^#{1,3}\s+(.+)$/m) || [])[1] || "";
      chunks.push({
        pageContent: piece,
        metadata: { source: file, label: SOURCE_LABELS[file] || file, heading },
      });
    }
  }
  return chunks;
}

let retrieverPromise = null;

// Build once per server instance. Vector search with Voyage embeddings when a key
// exists; otherwise BM25 keyword retrieval so the app still runs with zero extra keys.
export function getRetriever() {
  if (!retrieverPromise) {
    retrieverPromise = (async () => {
      const chunks = await loadChunks();
      if (process.env.VOYAGE_API_KEY) {
        const embeddings = new VoyageEmbeddings({
          apiKey: process.env.VOYAGE_API_KEY,
          modelName: process.env.VOYAGE_MODEL || "voyage-3.5-lite",
        });
        const store = await MemoryVectorStore.fromDocuments(chunks, embeddings);
        return { kind: "vector", retriever: store.asRetriever({ k: 4 }) };
      }
      return { kind: "bm25", retriever: BM25Retriever.fromDocuments(chunks, { k: 4 }) };
    })();
  }
  return retrieverPromise;
}

export async function retrieve(query, k = 4) {
  const { retriever, kind } = await getRetriever();
  const docs = await retriever.invoke(query);
  return { kind, docs: docs.slice(0, k) };
}

export function formatDocs(docs) {
  return docs
    .map((d, i) => `[${i + 1}] (${d.metadata.label}${d.metadata.heading ? ` › ${d.metadata.heading}` : ""})\n${d.pageContent}`)
    .join("\n\n");
}
