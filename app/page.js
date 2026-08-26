"use client";

import { useEffect, useRef, useState } from "react";

const EMAIL = "alexmarciag@gmail.com";

const GREETING =
  "Hey, I'm Alex. I'm a cost and operations research analyst at Regulus Group supporting FAA and DoD programs, with a master's in data analytic engineering and hands on LLM fine tuning experience, and I'm looking to move into data science and AI engineering work. Ask me anything, or paste a job description and I'll tell you honestly how well I fit.";

const JD_STARTER = { label: "Paste a job description to check fit", primary: true, prefill: "Here is a job description. How well do I fit?\n\n" };

// Pool of FAQ questions. After every reply, a few that have not been asked yet are offered above the composer.
const FAQ_POOL = [
  "Do you have real ML or LLM experience?",
  "What clearance do you hold?",
  "Tell me about your projects.",
  "Why data science?",
  "What do you do day to day?",
  "What's your biggest professional accomplishment?",
  "Why are you leaving your current role?",
  "What languages and tools do you work in?",
  "Have you worked in federal or defense environments?",
  "What kind of role are you looking for?",
  "Are you open to remote, hybrid, or relocation?",
  "When are you available to start?",
  "Do you have any certifications?",
  "Tell me about your education.",
  "Do you have leadership experience?",
  "Do you speak any other languages?",
  "How was this site built?",
  "What do you like to do outside of work?",
  "What's your favorite movie?",
];
const SUGGEST_COUNT = 6;

const TOOL_LABELS = {
  search_background: "Searched my background",
  assess_job_fit: "Scored the job description",
  list_projects: "Pulled up my project list",
};

const STACK = [
  ["RAG", "Markdown knowledge base chunked with LangChain, embedded with Voyage AI, searched in a vector store (BM25 fallback)."],
  ["LangChain agent", "createAgent tool calling loop on Claude via @langchain/anthropic; the model decides when to search or score."],
  ["Structured output", "Claude tags each posting requirement with a skill area, priority, and met/learnable/gap status via a Zod schema; code computes the score from a bucket based rubric so quantity cannot outweigh quality."],
  ["Grounding", "Answers only from retrieved chunks; every reply shows its sources."],
  ["Backend", "Next.js API route on Vercel, keys server side, per IP rate limits, input caps."],
];


const JD_PATTERN = /job description|how well do i fit|good fit for|requirements|responsibilities|qualifications/i;
function looksLikeJobDescription(text) {
  return text.length > 500 || JD_PATTERN.test(text);
}

const STAGES_CHAT = [
  ["Reading your message", 1.5],
  ["Searching my background", 3],
  ["Writing my answer", 4],
];
const STAGES_JD = [
  ["Reading the job description", 2],
  ["Searching my background for relevant experience", 4],
  ["Scoring requirements against my resume", 6],
  ["Weighing the gaps honestly", 4],
  ["Writing my answer", 4],
];

function Thinking({ jd }) {
  const stages = jd ? STAGES_JD : STAGES_CHAT;
  const total = Math.round(stages.reduce((s, [, t]) => s + t, 0));
  const [i, setI] = useState(0);
  useEffect(() => {
    if (i >= stages.length - 1) return;
    const t = setTimeout(() => setI(i + 1), stages[i][1] * 1000);
    return () => clearTimeout(t);
  }, [i, stages]);
  return (
    <div className="ac-thinking" aria-live="polite">
      <div className="ac-thinking-title">Thinking<span className="ac-ellipsis" /></div>
      <div className="ac-thinking-stage">{stages[i][0]}</div>
      <div className="ac-thinking-eta">should take about {total} seconds</div>
    </div>
  );
}

// 0 = light grey, 50 = blue, 100 = green
function scoreColor(score) {
  const s = Math.max(0, Math.min(100, score));
  const grey = [184, 190, 196], blue = [52, 87, 213], green = [15, 154, 102];
  const mix = (a, b, t) => a.map((v, k) => Math.round(v + (b[k] - v) * t));
  const c = s <= 50 ? mix(grey, blue, s / 50) : mix(blue, green, (s - 50) / 50);
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function FitCard({ fit }) {
  const score = Math.round(Number(fit?.score) || 0);
  const r = 44, circ = 2 * Math.PI * r;
  const [offset, setOffset] = useState(circ);
  const [showReqs, setShowReqs] = useState(false);
  useEffect(() => {
    const t = requestAnimationFrame(() => setOffset(circ - (circ * score) / 100));
    return () => cancelAnimationFrame(t);
  }, [score, circ]);
  const color = scoreColor(score);
  return (
    <div className="ac-fit">
      <div className="ac-fit-head">
        <svg className="ac-gauge" viewBox="0 0 110 110" role="img" aria-label={`Relevancy rank ${score} percent`}>
          <circle cx="55" cy="55" r={r} fill="none" stroke="var(--line)" strokeWidth="10" />
          <circle cx="55" cy="55" r={r} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
            strokeDasharray={circ} strokeDashoffset={offset} transform="rotate(-90 55 55)"
            style={{ transition: "stroke-dashoffset 1.1s cubic-bezier(.2,.8,.2,1)" }} />
          <text x="55" y="55" textAnchor="middle" dominantBaseline="central" className="ac-gauge-num" fill={color}>{score}%</text>
        </svg>
        <div>
          <div className="ac-fit-label">Relevancy rank</div>
          <div className="ac-fit-score" style={{ color }}>{score}%</div>
          {fit?.summary && <div className="ac-fit-summary">{fit.summary}</div>}
          {fit?.counts && (
            <div className="ac-fit-rubric">
              {fit.counts.met} met · {fit.counts.learnable} learnable · {fit.counts.gap} gaps across {fit.buckets?.length || 0} skill areas.
              Scored by area, not by count: required items weigh 3x, areas with a required item count double, and each required hard gap costs 4 points.
              <button type="button" className="ac-link" onClick={() => setShowReqs((v) => !v)}>{showReqs ? "Hide breakdown" : "See breakdown"}</button>
            </div>
          )}
        </div>
      </div>
      {showReqs && fit?.buckets?.length > 0 && (
        <div className="ac-buckets">
          {fit.buckets.map((b) => (
            <div key={b.category} className="ac-bucket">
              <div className="ac-bucket-head">
                <span className="ac-bucket-label">{b.label}{b.hasRequired && <span className="ac-req-pri required">required</span>}</span>
                <span className="ac-bucket-score" style={{ color: scoreColor(b.score) }}>{b.score}%</span>
              </div>
              <div className="ac-bar"><div className="ac-bar-fill" style={{ width: `${b.score}%`, background: scoreColor(b.score) }} /></div>
              {b.items.map((r, i) => (
                <div key={i} className="ac-req-row">
                  <span className={`ac-req-status ${r.status}`}>{r.status}</span>
                  <span className="ac-req-text">
                    {r.requirement}
                    <span className={`ac-req-pri ${r.priority}`}>{r.priority}</span>
                    {r.evidence && <span className="ac-req-ev">{r.evidence}</span>}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
      <div className="ac-fit-cols">
        <div className="ac-fit-col">
          <div className="ac-fit-h strong">Where I'm strong</div>
          <ul>{(fit?.matches || []).map((m, i) => <li key={i}>{m}</li>)}</ul>
        </div>
        <div className="ac-fit-col">
          <div className="ac-fit-h weak">Where I'm weaker</div>
          <ul>{(fit?.gaps || []).map((g, i) => <li key={i}>{g}</li>)}</ul>
        </div>
      </div>
      <div className="ac-fit-offer">
        <span className="ac-fit-h plan">If offered the role</span>
        <p>I can begin contributing right away in the areas where I'm strong while ramping up on the rest. I learn fastest when handed real work, which is how I delivered at Accure and at Regulus.</p>
      </div>
    </div>
  );
}


const MAIL_SUBJECT = "Let's schedule a meeting";
const MAIL_BODY = "Hi Alex,\n\nI came across your site and would like to set up a time to talk about a role.\n\nRole:\nCompany:\nTimes that work for me:\n\nThanks,\n";
const MAILTO = `mailto:${EMAIL}?subject=${encodeURIComponent(MAIL_SUBJECT)}&body=${encodeURIComponent(MAIL_BODY)}`;

// Turns any email address inside a reply into a clickable mailto link.
function linkify(text) {
  const parts = String(text).split(/([\w.+-]+@[\w-]+\.[\w.-]+)/g);
  return parts.map((part, i) =>
    /^[\w.+-]+@[\w-]+\.[\w.-]+$/.test(part)
      ? <a key={i} href={part === EMAIL ? MAILTO : `mailto:${part}`} className="ac-inline-mail">{part}</a>
      : part
  );
}

function CopyEmail({ className = "" }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(EMAIL);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {}
  }
  return (
    <button type="button" className={`ac-link ${className}`} onClick={copy} title="Copy email address">
      {copied ? "Copied" : "Copy email"}
    </button>
  );
}


// Inline architecture diagram shown whenever the answer draws on the "About this site" doc.
function ArchitectureDiagram() {
  const box = (x, y, w, h, title, sub, cls = "") => (
    <g className={`ad-box ${cls}`} transform={`translate(${x} ${y})`}>
      <rect width={w} height={h} rx="12" />
      <text x={w / 2} y={sub ? 24 : h / 2 + 5} textAnchor="middle" className="ad-title">{title}</text>
      {sub && <text x={w / 2} y={44} textAnchor="middle" className="ad-sub">{sub}</text>}
    </g>
  );
  const arrow = (x1, y1, x2, y2) => <path d={`M${x1} ${y1} L${x2} ${y2}`} className="ad-arrow" markerEnd="url(#ad-head)" />;
  return (
    <div className="ac-arch">
      <div className="ac-arch-title">How this site works</div>
      <svg viewBox="0 0 760 420" role="img" aria-label="Architecture: browser to Next.js API route to LangChain agent on Claude, which calls search, job fit, and project tools backed by a vector store built from markdown files">
        <defs>
          <marker id="ad-head" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0 0 L8 4 L0 8 z" className="ad-head" />
          </marker>
        </defs>
        {box(20, 20, 200, 60, "Browser", "React chat UI (Next.js)")}
        {arrow(220, 50, 280, 50)}
        {box(280, 20, 200, 60, "API route", "Vercel, rate limits, key server side")}
        {arrow(480, 50, 540, 50)}
        {box(540, 20, 200, 60, "LangChain agent", "createAgent on Claude", "accent")}
        {arrow(640, 80, 640, 130)}
        <text x="655" y="112" className="ad-note">model decides which tool to call</text>
        {box(80, 130, 180, 60, "search_background", "semantic search")}
        {box(290, 130, 180, 60, "assess_job_fit", "Zod structured output")}
        {box(500, 130, 180, 60, "list_projects", "projects.json")}
        {arrow(600, 130, 230, 130)}
        {arrow(610, 130, 400, 130)}
        {arrow(630, 130, 590, 130)}
        {arrow(170, 190, 170, 250)}
        {arrow(380, 190, 300, 250)}
        {box(60, 250, 320, 60, "Vector store", "Voyage embeddings, BM25 fallback", "gold")}
        {arrow(220, 310, 220, 360)}
        {box(40, 360, 360, 44, "resume.md · faq.md · projects.md · about-this-site.md", null, "data")}
        {box(470, 250, 250, 60, "Claude", "grounded first person reply", "accent")}
        {arrow(470, 280, 380, 280)}
        <text x="470" y="340" className="ad-note">answer + sources + fit card back to the browser</text>
      </svg>
    </div>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 19V5" />
      <path d="M5 12l7-7 7 7" />
    </svg>
  );
}

export default function Page() {
  const [messages, setMessages] = useState([{ role: "assistant", content: GREETING }]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyJD, setBusyJD] = useState(false);
  const [asked, setAsked] = useState([]);
  const suggestions = FAQ_POOL.filter((q) => !asked.includes(q)).slice(0, SUGGEST_COUNT);
  const [error, setError] = useState(null);
  const [showStack, setShowStack] = useState(false);
  const [showExit, setShowExit] = useState(false);
  const exitShown = useRef(false);

  // Exit intent: desktop fires when the cursor leaves through the top of the window,
  // mobile fires on a fast upward scroll near the top. Shows once per visit.
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      if (window.sessionStorage.getItem("exitShown")) exitShown.current = true;
    } catch {}
    const open = () => {
      if (exitShown.current) return;
      exitShown.current = true;
      try { window.sessionStorage.setItem("exitShown", "1"); } catch {}
      setShowExit(true);
    };
    const onMouseOut = (e) => {
      if (!e.relatedTarget && e.clientY <= 0) open();
    };
    let lastY = window.scrollY;
    const onScroll = () => {
      const y = window.scrollY;
      if (lastY - y > 120 && y < 80) open();
      lastY = y;
    };
    const onKey = (e) => { if (e.key === "Escape") setShowExit(false); };
    document.addEventListener("mouseout", onMouseOut);
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mouseout", onMouseOut);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("keydown", onKey);
    };
  }, []);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  function autosize() {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }

  async function send(text) {
    const q = (text ?? input).trim();
    if (!q || busy) return;
    setInput("");
    setError(null);
    const next = [...messages, { role: "user", content: q }];
    setMessages(next);
    setBusyJD(looksLikeJobDescription(q));
    setBusy(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Drop the scripted greeting; the API expects the thread to start with a user turn.
        body: JSON.stringify({ messages: next.slice(1) }),
      });
      const data = await res.json();
      if (!res.ok || !data.reply) throw new Error(data.error || "Empty reply");
      setMessages([...next, { role: "assistant", content: data.reply, tools: data.toolsUsed || [], sources: data.sources || [], fit: data.fit || null, architecture: !!data.architecture }]);
    } catch (e) {
      setError(e.message || "Couldn't send that. Please try again.");
      setMessages(messages);
      setInput(q);
    } finally {
      setBusy(false);
      requestAnimationFrame(() => {
        autosize();
        inputRef.current?.focus();
      });
    }
  }

  function onKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function onStarter(s) {
    if (s.prefill) {
      setInput(s.prefill);
      requestAnimationFrame(() => {
        autosize();
        inputRef.current?.focus();
      });
    } else {
      setAsked((a) => [...a, s.label]);
      send(s.label);
    }
  }

  return (
    <div className="ac">
      <header className="ac-header">
        <div className="ac-wrap" style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px 20px" }}>
          <div className="ac-avatar ac-display">AM</div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="ac-display" style={{ fontWeight: 700, fontSize: 17, lineHeight: 1.2 }}>Alex Marcia-Gonzalez</div>
            <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              Cost &amp; OR Analyst, Regulus Group · M.S. GMU · Woodbridge, VA
            </div>
          </div>
        </div>
      </header>

      <div className="ac-wrap">
        <section className="ac-overview">
          <h1 className="ac-display">Hey! I'm Alex</h1>
          <p>
            I'm the lead cost analyst on FAA and DoD programs at Regulus Group with six plus years turning messy program data into funded decisions,
            a master's in data analytic engineering from George Mason, and hands on LLM fine tuning experience. I'm moving into data science and AI
            engineering full time, and I hold an active Public Trust clearance.
          </p>
          <div className="ac-what">
            <strong>What this is:</strong> a chatbot that speaks as me and answers only from my resume, so you get straight answers fast.
            Ask anything, or paste a job description and I'll tell you where I match and where I'd grow.
          </div>
        </section>
      </div>

      <div className="ac-wrap ac-thread">
        {messages.map((m, i) => (
          <div key={i}>
            {m.role === "assistant" && m.tools?.length > 0 && (
              <div className="ac-tool">{m.tools.map((t) => TOOL_LABELS[t] || t).join(" · ")}</div>
            )}
            {m.role === "assistant" && m.fit && <FitCard fit={m.fit} />}
            {m.role === "assistant" && m.architecture && <ArchitectureDiagram />}
            <div className={`ac-row ${m.role === "user" ? "user" : "bot"}`}>
              {m.role === "assistant" && <div className="ac-mini ac-display">AM</div>}
              <div className="ac-bubble">{m.role === "assistant" ? linkify(m.content) : m.content}</div>
            </div>
            {m.role === "assistant" && m.sources?.length > 0 && (
              <div className="ac-sources">
                <span>Sources</span>
                {m.sources.map((s) => <span key={s} className="ac-source">{s}</span>)}
              </div>
            )}
          </div>
        ))}

        {busy && (
          <div className="ac-row bot">
            <div className="ac-mini ac-display">AM</div>
            <div className="ac-bubble">
              <Thinking jd={busyJD} />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="ac-wrap ac-composer">
        {!busy && (
          <div className="ac-suggest-wrap">
            <div className="ac-suggest" role="list" aria-label="Suggested questions"
              onWheel={(e) => { if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) { e.currentTarget.scrollLeft += e.deltaY; } }}>
              <button className="ac-chip primary" onClick={() => onStarter(JD_STARTER)}>{JD_STARTER.label}</button>
              {suggestions.map((q) => (
                <button key={q} className="ac-chip" onClick={() => onStarter({ label: q })}>{q}</button>
              ))}
            </div>
          </div>
        )}
        <div className="ac-field">
          <textarea
            ref={inputRef}
            className="ac-input"
            rows={1}
            value={input}
            onChange={(e) => { setInput(e.target.value); autosize(); }}
            onKeyDown={onKey}
            placeholder="Ask me anything, or paste a job description…"
            aria-label="Your message"
            disabled={busy}
          />
          <button className="ac-send" onClick={() => send()} disabled={busy || !input.trim()} aria-label="Send">
            <SendIcon />
          </button>
        </div>
        {error && <div className="ac-error">{error}</div>}
        <div className="ac-foot">
          I only answer from my resume. Want a real conversation? <a href={MAILTO}>{EMAIL}</a> <span className="ac-sep">·</span> <CopyEmail />
          {" · "}
          <button className="ac-link" onClick={() => setShowStack((v) => !v)}>
            {showStack ? "Hide" : "How this site works"}
          </button>
        </div>
        {showStack && (
          <div className="ac-stack">
            {STACK.map(([k, v]) => (
              <div key={k} className="ac-stack-row">
                <span className="ac-stack-key">{k}</span>
                <span>{v}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {showExit && (
        <div className="ac-modal-backdrop" onClick={() => setShowExit(false)}>
          <div className="ac-modal" role="dialog" aria-modal="true" aria-labelledby="ac-exit-title" onClick={(e) => e.stopPropagation()}>
            <div className="ac-mini ac-display" style={{ width: 40, height: 40, borderRadius: 12, fontSize: 14 }}>AM</div>
            <h2 id="ac-exit-title" className="ac-display">Woah, before you go</h2>
            <p>
              All I need is a chance. Give me the opportunity and I will rise to the occasion. I take pride in delivering good work to clients
              and I always carry more than just my own weight in team settings. Shoot me an email to schedule a meeting and you won't regret it.
            </p>
            <div className="ac-modal-actions">
              <a className="ac-btn primary" href={MAILTO}>Email {EMAIL}</a>
              <button className="ac-btn" onClick={() => setShowExit(false)}>Not right now</button>
            </div>
            <div className="ac-modal-alt">
              No mail app on this device? <CopyEmail /> and paste it into Gmail or Outlook.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
