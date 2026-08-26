import { runAgent } from "../../../lib/agent.js";

export const runtime = "nodejs";
export const maxDuration = 60;

const MAX_HISTORY = 12;          // messages kept per request (6 turns)
const MAX_CHARS = 8000;          // per message, large enough for a pasted job description
const RATE_LIMIT = 20;           // requests per window per IP
const WINDOW_MS = 10 * 60 * 1000;

// Per serverless instance rate limiting; good enough for a personal site.
const hits = new Map();
function rateLimited(ip) {
  const now = Date.now();
  const entry = hits.get(ip) || { count: 0, start: now };
  if (now - entry.start > WINDOW_MS) {
    entry.count = 0;
    entry.start = now;
  }
  entry.count += 1;
  hits.set(ip, entry);
  return entry.count > RATE_LIMIT;
}

export async function POST(req) {
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "unknown";
  if (rateLimited(ip)) {
    return Response.json({ error: "Too many messages. Please try again in a few minutes." }, { status: 429 });
  }
  if (!process.env.ANTHROPIC_API_KEY) {
    return Response.json({ error: "Server is missing ANTHROPIC_API_KEY." }, { status: 500 });
  }

  let body;
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON." }, { status: 400 });
  }

  const messages = (Array.isArray(body?.messages) ? body.messages : [])
    .filter((m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
    .map((m) => ({ role: m.role, content: m.content.slice(0, MAX_CHARS) }))
    .slice(-MAX_HISTORY);

  if (messages.length === 0 || messages[messages.length - 1].role !== "user") {
    return Response.json({ error: "Send at least one user message." }, { status: 400 });
  }

  try {
    const out = await runAgent(messages);
    if (!out.reply) out.reply = "I got a little tangled up there. Could you ask that again?";
    return Response.json(out);
  } catch (err) {
    console.error(err);
    return Response.json({ error: "Something went wrong talking to the model." }, { status: 502 });
  }
}
