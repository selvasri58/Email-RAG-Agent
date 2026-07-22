"""
agent.py
─────────
LangGraph-powered Email RAG agent.

Architecture
============

         ┌──────────┐
         │  USER    │
         └────┬─────┘
              ▼
       ┌────────────┐        ┌─────────────────────────────┐
       │   AGENT    │ ─────► │ search_vector_db_with_filters│
       │  (Groq)    │        ├─────────────────────────────┤
       │  + tools   │ ─────► │      fetch_live_emails       │
       └────┬───────┘        └─────────────────────────────┘
            ▼
        ANSWER

The agent node is a Groq LLM (Llama 3.3 70B) bound to two tools:

  • search_vector_db_with_filters(query, sender=?, date_range=?)
        Semantic search in Qdrant; can filter by sender domain/email and
        a date range (YYYY-MM-DD). Use this for "what did X say about Y",
        "summarize emails from last week about onboarding", etc.

  • fetch_live_emails(sender=?, date=?)
        Hits the live mailbox via IMAP (INBOX + Spam + All Mail). Use this
        for strict time-sensitive questions like "Did Acme email me today?"
        where we must not rely on the embedded index.

The router uses LangGraph's prebuilt ToolNode + conditional edges so the
LLM decides which tool (or no tool) to call. Multiple tool hops are allowed
in a single turn.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional, TypedDict

from imapclient import IMAPClient
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from qdrant_client.http import models as qmodels

from common import (
    Config,
    embed_texts,
    ensure_collection,
    get_qdrant,
    load_config,
    log,
)

# Single global config + clients. Built once when the agent module is imported.
CFG: Config = load_config()
QCLIENT = get_qdrant(CFG)
ensure_collection(QCLIENT, CFG)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD (or YYYY/MM/DD) as a UTC date at 00:00:00."""
    s = s.strip().replace("/", "-")
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _resolve_relative_date(token: str) -> Optional[str]:
    """Turn 'today', 'yesterday', 'YYYY-MM-DD' into 'YYYY-MM-DD'. None on failure."""
    t = token.lower().strip()
    today = datetime.now(timezone.utc).date()
    if t == "today":
        return today.isoformat()
    if t == "yesterday":
        return (today - timedelta(days=1)).isoformat()
    try:
        return _parse_date(token).date().isoformat()
    except Exception:
        return None


def _build_qdrant_filter(
    sender: Optional[str], date_range: Optional[tuple[str, str]]
) -> Optional[qmodels.Filter]:
    """Convert user-facing sender/date inputs into a Qdrant filter object."""
    must: list = []

    if sender:
        s = sender.lower().strip()
        if "@" in s:
            # Looks like a full email address — exact match against sender_email.
            must.append(
                qmodels.FieldCondition(
                    key="sender_email", match=qmodels.MatchValue(value=s)
                )
            )
        else:
            # Treat as a brand / name fragment. Match if the substring appears
            # in EITHER the sender_email OR the sender_domain field. Use a
            # `should` (OR) clause nested inside the outer `must` (AND).
            must.append(
                qmodels.Filter(
                    should=[
                        qmodels.FieldCondition(
                            key="sender_domain", match=qmodels.MatchText(text=s)
                        ),
                        qmodels.FieldCondition(
                            key="sender_email", match=qmodels.MatchText(text=s)
                        ),
                    ]
                )
            )

    if date_range:
        start_iso, end_iso = date_range
        try:
            start_ts = int(_parse_date(start_iso).timestamp())
            # end date is inclusive — push to the end of that day
            end_ts = int((_parse_date(end_iso) + timedelta(days=1)).timestamp()) - 1
            must.append(
                qmodels.FieldCondition(
                    key="received_at_ts",
                    range=qmodels.Range(gte=start_ts, lte=end_ts),
                )
            )
        except Exception:
            pass

    return qmodels.Filter(must=must) if must else None


# ─── Tool 1: Vector DB search with metadata filters ───────────────────────────
@tool
def search_vector_db_with_filters(
    query: str = "",
    sender: Optional[str] = None,
    date_range: Optional[list[str]] = None,
    top_k: int = 8,
) -> str:
    """
    Semantic search over the indexed email corpus stored in Qdrant.

    Use this for fuzzy, content-based questions such as:
      • "What did the recruiter from Acme say about the offer?"
      • "Summarize emails about the Q3 budget"
      • "Find emails mentioning AWS migration"

    Args:
      query: Natural-language search query describing what to find.
             Required and must NOT be empty.
      sender: Optional. Either a full email ("ceo@acme.com") or a name
              fragment ("acme", "discord"). ONLY pass this if the user
              explicitly named a sender in the CURRENT question.
      date_range: Optional. A [start_date, end_date] list, each in
                  "YYYY-MM-DD" format. ONLY pass this if the user explicitly
                  mentioned dates or a time window like "last week".
      top_k: How many chunks to return (default 8). Must be a plain integer.

    Returns:
      A formatted string listing the top matching email chunks with their
      sender, subject, received time, and a content snippet. Returns
      "NO_RESULTS" if nothing matches.
    """
    # Defensive coercion — some LLMs (notably Llama on Groq) occasionally
    # emit integer arguments as strings. Don't let that crash the agent.
    try:
        top_k = int(top_k) if top_k is not None else 8
    except (TypeError, ValueError):
        top_k = 8
    top_k = max(1, min(top_k, 50))  # clamp to a sane range

    if sender is not None and not isinstance(sender, str):
        sender = str(sender)

    # If the model called us with no query, salvage it from filter context.
    if not query or not str(query).strip():
        query = (sender or "email") + " message"

    log(
        "🔎 [Tool]",
        f"search_vector_db_with_filters(query='{query[:60]}', sender={sender}, "
        f"date_range={date_range}, top_k={top_k})",
        "yellow",
    )

    # Validate date_range
    dr: Optional[tuple[str, str]] = None
    if date_range:
        if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
            return "ERROR: date_range must be a list of exactly two dates [start, end]."
        dr = (str(date_range[0]), str(date_range[1]))

    qfilter = _build_qdrant_filter(sender, dr)

    [qvec] = embed_texts([query], CFG.embedding_model)

    # qdrant-client 1.15+ removed `.search()` — use `.query_points()` instead.
    # It returns a response object with a `.points` attribute (list of ScoredPoint).
    response = QCLIENT.query_points(
        collection_name=CFG.qdrant_collection,
        query=qvec,
        query_filter=qfilter,
        limit=top_k,
        with_payload=True,
    )
    hits = response.points

    # If filtered search came up empty BUT we had filters, retry without them.
    # Often the filters are too narrow (e.g. model picked the wrong date or
    # a domain that doesn't exactly match what's indexed).
    if not hits and qfilter is not None:
        log("🔎 [Tool]", "Filtered search returned nothing — retrying with no filters.", "yellow")
        response = QCLIENT.query_points(
            collection_name=CFG.qdrant_collection,
            query=qvec,
            query_filter=None,
            limit=max(top_k, 8),
            with_payload=True,
        )
        hits = response.points

    if not hits:
        return (
            "NO_RESULTS — the vector search returned no matching emails. "
            "Tell the user clearly that no relevant emails were found in the "
            "indexed corpus and do NOT invent any details."
        )

    # Deduplicate by email UID — multiple chunks from the same email collapse to one.
    seen_uids: set[int] = set()
    lines: list[str] = []
    for h in hits:
        p = h.payload or {}
        uid = p.get("uid")
        if uid in seen_uids:
            continue
        seen_uids.add(uid)

        snippet = (p.get("text", "") or "")[:400].replace("\n", " ")
        lines.append(
            f"• [{p.get('received_at_iso', '?')}] "
            f"FROM: {p.get('sender_name', '')} <{p.get('sender_email', '')}> | "
            f"SUBJECT: {p.get('subject', '(no subject)')}\n"
            f"  SNIPPET: {snippet}\n"
            f"  RELEVANCE: {h.score:.3f}"
        )

    return "\n\n".join(lines)


# ─── Tool 2: Live IMAP fetch ──────────────────────────────────────────────────
# Gmail-specific folder names. The Spam folder is "[Gmail]/Spam" and the
# combined view (which includes Sent, Spam, Trash, Inbox) is "[Gmail]/All Mail".
GMAIL_FOLDERS = ["INBOX", "[Gmail]/Spam", "[Gmail]/All Mail"]


def _imap_search_folder(
    imap: IMAPClient, folder: str, on_date: datetime, sender: Optional[str]
) -> list[dict]:
    """Search one folder for messages received on `on_date` (optionally from a sender)."""
    try:
        imap.select_folder(folder, readonly=True)
    except Exception as e:
        log("⚠️  [Live]", f"Cannot open folder '{folder}': {e}", "red")
        return []

    # IMAP date format is "DD-Mon-YYYY". Search by SINCE / BEFORE on local date.
    next_day = on_date + timedelta(days=1)
    criteria: list = [
        "SINCE", on_date.strftime("%d-%b-%Y"),
        "BEFORE", next_day.strftime("%d-%b-%Y"),
    ]
    if sender:
        criteria += ["FROM", sender]

    try:
        uids = imap.search(criteria)
    except Exception as e:
        log("⚠️  [Live]", f"Search failed in '{folder}': {e}", "red")
        return []

    if not uids:
        return []

    fetched = imap.fetch(uids, [b"ENVELOPE", b"INTERNALDATE"])
    out: list[dict] = []
    for uid, data in fetched.items():
        env = data.get(b"ENVELOPE")
        internal = data.get(b"INTERNALDATE")
        if env is None:
            continue

        # ENVELOPE.from_ is a tuple of Address objects.
        from_name, from_addr = "", ""
        if env.from_:
            a = env.from_[0]
            try:
                from_name = (a.name or b"").decode(errors="replace") if isinstance(a.name, bytes) else (a.name or "")
                mailbox = (a.mailbox or b"").decode(errors="replace") if isinstance(a.mailbox, bytes) else (a.mailbox or "")
                host = (a.host or b"").decode(errors="replace") if isinstance(a.host, bytes) else (a.host or "")
                from_addr = f"{mailbox}@{host}" if mailbox and host else ""
            except Exception:
                pass

        subject = ""
        if env.subject:
            try:
                subject = env.subject.decode(errors="replace") if isinstance(env.subject, bytes) else env.subject
            except Exception:
                subject = str(env.subject)

        when_iso = ""
        if internal is not None:
            try:
                when_iso = internal.isoformat()
            except Exception:
                when_iso = str(internal)

        out.append(
            {
                "uid": int(uid),
                "folder": folder,
                "from_name": from_name,
                "from_email": from_addr,
                "subject": subject,
                "received_at": when_iso,
            }
        )

    return out


@tool
def fetch_live_emails(
    date: str = "today",
    sender: Optional[str] = None,
) -> str:
    """
    Fetch emails LIVE from the mailbox (INBOX + Spam + All Mail) for a specific
    calendar date. Bypasses the vector index entirely — use this when the user
    needs a definitive answer to a time-sensitive question.

    Examples of when to call this:
      • "Did Acme email me today?"
      • "Show me everything from yesterday."
      • "Did support@stripe.com send anything on 2025-05-30?"

    Args:
      date: Either "today", "yesterday", or an explicit "YYYY-MM-DD" string.
      sender: Optional. A name fragment or email address to filter by.

    Returns:
      A formatted string listing matching emails with sender, subject, time,
      and the folder they were found in. Returns "NO_EMAILS_FOUND" if nothing
      matched.
    """
    # Defensive coercion
    if date is None:
        date = "today"
    date = str(date).strip()
    if sender is not None and not isinstance(sender, str):
        sender = str(sender)

    log("📡 [Tool]", f"fetch_live_emails(date={date}, sender={sender})", "yellow")

    iso = _resolve_relative_date(date)
    if not iso:
        return f"ERROR: could not parse date '{date}'. Use 'today', 'yesterday', or 'YYYY-MM-DD'."

    target_dt = _parse_date(iso)

    imap = IMAPClient(CFG.imap_host, port=CFG.imap_port, ssl=True, use_uid=True)
    try:
        imap.login(CFG.email_address, CFG.email_password)
        results: list[dict] = []
        seen_keys: set[tuple] = set()  # dedupe across folders (All Mail overlaps INBOX)
        for folder in GMAIL_FOLDERS:
            for hit in _imap_search_folder(imap, folder, target_dt, sender):
                key = (hit["from_email"], hit["subject"], hit["received_at"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                results.append(hit)
    finally:
        try:
            imap.logout()
        except Exception:
            pass

    if not results:
        return "NO_EMAILS_FOUND"

    # Sort newest-first
    results.sort(key=lambda r: r["received_at"], reverse=True)

    lines = [
        f"Found {len(results)} email(s) for {iso}" + (f" from '{sender}'" if sender else "") + ":",
        "",
    ]
    for r in results:
        lines.append(
            f"• [{r['received_at']}] FROM: {r['from_name']} <{r['from_email']}> | "
            f"SUBJECT: {r['subject']} | FOLDER: {r['folder']}"
        )
    return "\n".join(lines)


# ─── LangGraph state & graph ──────────────────────────────────────────────────
TOOLS = [search_vector_db_with_filters, fetch_live_emails]


class AgentState(TypedDict):
    """Conversation state — LangGraph appends new messages on each step."""
    messages: Annotated[list[AnyMessage], add_messages]


SYSTEM_PROMPT = """You are an expert email research assistant for the user's personal mailbox.

You have access to TWO tools:

  1. search_vector_db_with_filters(query, sender, date_range)
       — Use for content-based / fuzzy / semantic questions.
       — Use when the user asks about TOPICS, KEYWORDS, or SUMMARIES.
       — Always pass concrete YYYY-MM-DD dates if the user mentions a time window.

  2. fetch_live_emails(date, sender)
       — Use for STRICTLY TIME-SENSITIVE questions like:
            "Did X email me today?"   "What came in yesterday?"
            "Anything from acme.com on 2025-05-12?"
       — Authoritative because it queries the live mailbox.
       — Use for questions about EXISTENCE, COUNTS, or RECENT ARRIVALS.

Routing rules:
  • If the user's question is "Did/Has X emailed me [today|yesterday|on DATE]" → fetch_live_emails.
  • If the question is about WHAT an email said, summarisation, topics, or older history → search_vector_db_with_filters.
  • If unsure between the two, prefer fetch_live_emails for recency questions.
  • You MAY call multiple tools in sequence. For example, confirm existence
    with fetch_live_emails, then dig into content with search_vector_db_with_filters.

CRITICAL — when NOT to call any tool:
  • If the user asks about general knowledge (weather, news, math, coding, definitions,
    trivia, science, history, etc.) → DO NOT call any tool. Politely answer that you
    only help with their email inbox and suggest they ask an email-related question.
  • If the user asks "what can you do" / "help" / "who are you" → DO NOT call any tool.
    Describe your capabilities directly: searching their email by content and fetching
    emails from specific dates/senders.
  • If the user just says hi / thanks / casual chat → DO NOT call any tool. Reply naturally.
  • Only call a tool when the question is genuinely about the user's email.

IMPORTANT — DO NOT add filters the user did not explicitly request:
  • Only set `sender` if the user explicitly named a sender in the CURRENT question.
  • Only set `date_range` if the user used words like "today", "yesterday", "last week",
    or a specific date. If the user did NOT mention any time, leave date_range as None.
  • Do NOT add today's date as a date_range just because the user is asking now.
  • Do NOT carry over senders or dates from previous messages in the conversation.
  • Default top_k is 8. Never use top_k=1 — it's almost always too narrow.
  • When in doubt, leave ALL filters as None and let semantic search do the work.

When you DO call tools, pass arguments with correct types and ALL required fields:
  • query is REQUIRED — must be a non-empty string describing what to find.
    Example: "discord notification" or "tayana academy message".
  • top_k must be a plain integer like 8 (NOT a string "8")
  • date_range must be a list of two strings ["YYYY-MM-DD", "YYYY-MM-DD"]
  • date must be "today", "yesterday", or a "YYYY-MM-DD" string
  • sender must be a plain string

CRITICAL — DO NOT HALLUCINATE:
  • If a tool returns "NO_RESULTS" or "NO_EMAILS_FOUND", you MUST tell the user
    plainly that no matching emails were found. Suggest they rephrase or check
    their inbox directly. NEVER invent email content, senders, or subjects that
    the tool did not return.
  • If a tool returns results, only describe what is actually in those results.
    Do not fabricate details that are not in the snippet.

Today's date: {today}

Always provide a clean natural-language summary of the findings. When listing
emails, include sender, subject, and timestamp. Never invent emails that the
tools did not return. If both tools return nothing, say so plainly."""


def build_agent_graph():
    """Compile the LangGraph state machine."""
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    ).bind_tools(TOOLS)

    today_iso = datetime.now(timezone.utc).date().isoformat()

    def agent_node(state: AgentState) -> dict:
        log("⚙️  [Agent]", "Reasoning about which tool to call (or whether to answer) ...", "bold blue")
        msgs = state["messages"]
        # Prepend the system message exactly once.
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT.format(today=today_iso))] + list(msgs)

        try:
            ai_msg = llm.invoke(msgs)
        except Exception as e:
            # Catch malformed tool-call errors from the model (e.g. wrong arg types
            # that Groq's strict validator rejects) and recover gracefully instead
            # of crashing the entire chat session.
            err_str = str(e)
            if "tool_use_failed" in err_str or "tool call validation" in err_str:
                log("⚠️  [Agent]", "Model emitted a malformed tool call — see details below:", "yellow")
                # Log the raw error so we can diagnose what Llama actually emitted.
                log("⚠️  [Agent]", f"Raw error: {err_str[:1500]}", "red")
                ai_msg = AIMessage(
                    content=(
                        "I had trouble processing that request — I'm specifically "
                        "designed to help you search your email inbox. Could you "
                        "rephrase as an email-related question? For example:\n\n"
                        "  • \"Did Amazon email me today?\"\n"
                        "  • \"Summarize emails from my recruiter last week\"\n"
                        "  • \"What did GitHub send me recently?\""
                    )
                )
            else:
                # Re-raise anything we don't recognise so it surfaces properly.
                raise

        # Surface tool intentions in the terminal.
        if getattr(ai_msg, "tool_calls", None):
            for tc in ai_msg.tool_calls:
                log("⚙️  [Agent]", f"→ Calling tool '{tc['name']}' with args {tc.get('args', {})}", "bold blue")
        else:
            log("⚙️  [Agent]", "No more tools needed — generating final answer.", "bold blue")
        return {"messages": [ai_msg]}

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    # If the LLM emitted tool_calls, go to ToolNode; otherwise we're done.
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# Compile once at import time so main.py can hot-loop without rebuilding.
AGENT = build_agent_graph()


def ask(question: str, history: list[AnyMessage] | None = None) -> tuple[str, list[AnyMessage]]:
    """
    Run one user question through the agent.

    Returns (final_answer_string, updated_message_history).
    """
    history = history or []
    history.append(HumanMessage(content=question))
    final = AGENT.invoke({"messages": history})
    history = final["messages"]
    answer = history[-1].content if history else ""
    return answer, history