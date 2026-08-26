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

let chunksPromise = null;
let vectorPromise = null;
let bm25Promise = null;

function getChunks() {
  if (!chunksPromise) chunksPromise = loadChunks();
  return chunksPromise;
}

// Keyword retriever: always available, no external calls.
function getBM25() {
  if (!bm25Promise) bm25Promise = getChunks().then((chunks) => BM25Retriever.fromDocuments(chunks, { k: 4 }));
  return bm25Promise;
}

// Vector retriever: built once per server instance when a Voyage key exists.
// If the build fails (rate limit, network), we forget the attempt so a later request can retry.
function getVector() {
  if (!process.env.VOYAGE_API_KEY) return null;
  if (!vectorPromise) {
    vectorPromise = getChunks()
      .then(async (chunks) => {
        const embeddings = new VoyageEmbeddings({
          apiKey: process.env.VOYAGE_API_KEY,
          modelName: process.env.VOYAGE_MODEL || "voyage-3.5-lite",
          batchSize: 64,
        });
        const store = await MemoryVectorStore.fromDocuments(chunks, embeddings);
        return store.asRetriever({ k: 4 });
      })
      .catch((err) => {
        console.warn("Vector index build failed, will retry later:", err?.message || err);
        vectorPromise = null;
        return null;
      });
  }
  return vectorPromise;
}

export function getRetriever() {
  return getVector() ? getVector().then((v) => ({ kind: v ? "vector" : "bm25", retriever: v })) : Promise.resolve({ kind: "bm25", retriever: null });
}

// Semantic search when possible, keyword search otherwise. Never throws.
export async function retrieve(query, k = 4) {
  const vector = await getVector();
  if (vector) {
    try {
      const docs = await vector.invoke(query);
      return { kind: "vector", docs: docs.slice(0, k) };
    } catch (err) {
      console.warn("Vector search failed, falling back to keyword search:", err?.message || err);
    }
  }
  const bm25 = await getBM25();
  const docs = await bm25.invoke(query);
  return { kind: "bm25", docs: docs.slice(0, k) };
}

export function formatDocs(docs) {
  return docs
    .map((d, i) => `[${i + 1}] (${d.metadata.label}${d.metadata.heading ? ` › ${d.metadata.heading}` : ""})\n${d.pageContent}`)
    .join("\n\n");
}
