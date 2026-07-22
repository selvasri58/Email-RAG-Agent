"""
common.py
─────────
Shared utilities used by ingest.py and agent.py:

  • Config loading (env vars with sensible defaults)
  • Email parsing (multipart MIME → clean text + sender/subject/date)
  • Embedding model singleton (load HF model once)
  • Qdrant client + collection bootstrap
  • Rich console logger
"""

from __future__ import annotations

import email
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime
from typing import Iterable

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rich.console import Console
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─── Console ──────────────────────────────────────────────────────────────────
console = Console()


def log(prefix: str, message: str, style: str = "cyan") -> None:
    """Pretty timestamped log line used everywhere."""
    ts = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{ts}[/dim] [{style}]{prefix}[/{style}] {message}")


# ─── Config ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    email_address: str
    email_password: str
    imap_host: str
    imap_port: int
    imap_mailbox: str
    google_api_key: str
    gemini_model: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int


def load_config() -> Config:
    """Load configuration from environment, validating required fields."""
    required = ["EMAIL_ADDRESS", "EMAIL_APP_PASSWORD", "GROQ_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the values."
        )

    return Config(
        email_address=os.environ["EMAIL_ADDRESS"],
        email_password=os.environ["EMAIL_APP_PASSWORD"],
        imap_host=os.getenv("IMAP_HOST", "imap.gmail.com"),
        imap_port=int(os.getenv("IMAP_PORT", "993")),
        imap_mailbox=os.getenv("IMAP_MAILBOX", "INBOX"),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "emails"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
    )


# ─── Embedding model (singleton) ──────────────────────────────────────────────
_embedder: SentenceTransformer | None = None


def get_embedder(model_name: str) -> SentenceTransformer:
    """Lazy-load the HF embedding model. Downloaded to ~/.cache on first call."""
    global _embedder
    if _embedder is None:
        log("🧠 [Embedder]", f"Loading {model_name} ...", "magenta")
        _embedder = SentenceTransformer(model_name)
        log("🧠 [Embedder]", "Ready.", "magenta")
    return _embedder


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    """Encode a list of strings into normalized embedding vectors."""
    model = get_embedder(model_name)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


# ─── Qdrant client + collection bootstrap ─────────────────────────────────────
def get_qdrant(cfg: Config) -> QdrantClient:
    """Return a Qdrant client (REST). Use grpc=True for higher throughput later."""
    return QdrantClient(
        host=cfg.qdrant_host,
        port=cfg.qdrant_port,
        timeout=120,                # seconds — large upserts can take a while
        check_compatibility=False,  # silences the 1.18 vs 1.11 warning
    )


def ensure_collection(client: QdrantClient, cfg: Config) -> None:
    """Create the emails collection + payload indexes if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if cfg.qdrant_collection in existing:
        return

    log("🗄️  [Qdrant]", f"Creating collection '{cfg.qdrant_collection}'", "yellow")
    client.create_collection(
        collection_name=cfg.qdrant_collection,
        vectors_config=qmodels.VectorParams(
            size=cfg.embedding_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )
    # Payload indexes make metadata filtering fast.
    for field, schema in [
        ("sender_email", qmodels.PayloadSchemaType.KEYWORD),
        ("sender_domain", qmodels.PayloadSchemaType.KEYWORD),
        ("received_at_ts", qmodels.PayloadSchemaType.INTEGER),
        ("uid", qmodels.PayloadSchemaType.INTEGER),
        ("date_str", qmodels.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=cfg.qdrant_collection,
            field_name=field,
            field_schema=schema,
        )
    log("🗄️  [Qdrant]", "Collection ready.", "yellow")


# ─── Email parsing ────────────────────────────────────────────────────────────
@dataclass
class ParsedEmail:
    uid: int
    message_id: str
    sender_name: str
    sender_email: str
    sender_domain: str
    subject: str
    received_at: datetime          # always UTC
    body: str                      # plain-text body (HTML stripped, attachments listed)
    attachment_names: list[str]


def _decode(value) -> str:
    """Safely decode email header bytes to str."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return value.decode("latin-1", errors="replace")
    return str(value)


def _header(msg: Message, name: str) -> str:
    """Get a header, decoding RFC 2047 encoded-word forms."""
    raw = msg.get(name, "")
    if not raw:
        return ""
    decoded_parts = email.header.decode_header(raw)
    pieces: list[str] = []
    for text, charset in decoded_parts:
        if isinstance(text, bytes):
            pieces.append(text.decode(charset or "utf-8", errors="replace"))
        else:
            pieces.append(text)
    return "".join(pieces).strip()


def _strip_html(html: str) -> str:
    """Convert HTML email body to clean plain text."""
    soup = BeautifulSoup(html, "lxml")
    # Remove script / style noise
    for tag in soup(["script", "style", "head", "meta", "title"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse runs of whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _extract_body_and_attachments(msg: Message) -> tuple[str, list[str]]:
    """Walk a (possibly multipart) message; return (body_text, attachment_filenames)."""
    plain_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[str] = []

    for part in msg.walk():
        ctype = part.get_content_type()
        disposition = (part.get("Content-Disposition") or "").lower()

        if "attachment" in disposition or part.get_filename():
            fn = part.get_filename()
            if fn:
                attachments.append(_decode(fn))
            continue

        if part.is_multipart():
            continue

        payload = part.get_payload(decode=True)
        if payload is None:
            continue

        charset = part.get_content_charset() or "utf-8"
        try:
            text = payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            text = payload.decode("utf-8", errors="replace")

        if ctype == "text/plain":
            plain_parts.append(text)
        elif ctype == "text/html":
            html_parts.append(_strip_html(text))

    # Prefer plain text, fall back to HTML-stripped — but include both for completeness.
    pieces = []
    if plain_parts:
        pieces.append("\n".join(plain_parts).strip())
    if html_parts:
        pieces.append("\n".join(html_parts).strip())

    body = "\n\n".join(p for p in pieces if p).strip()

    if attachments:
        body += "\n\n[Attachments: " + ", ".join(attachments) + "]"

    return body, attachments


def parse_email(raw_bytes: bytes, uid: int) -> ParsedEmail | None:
    """Parse raw IMAP message bytes into a ParsedEmail. Returns None on hard failure."""
    try:
        msg = email.message_from_bytes(raw_bytes)
    except Exception as e:
        log("⚠️  [Parse]", f"Failed to parse UID {uid}: {e}", "red")
        return None

    # --- sender ---
    from_raw = _header(msg, "From")
    addrs = getaddresses([from_raw])
    sender_name, sender_email = (addrs[0] if addrs else ("", ""))
    sender_email = sender_email.lower().strip()
    sender_domain = sender_email.split("@", 1)[1] if "@" in sender_email else ""

    # --- subject ---
    subject = _header(msg, "Subject") or "(no subject)"

    # --- date ---
    date_raw = msg.get("Date")
    received_at: datetime
    if date_raw:
        try:
            received_at = parsedate_to_datetime(date_raw)
            if received_at.tzinfo is None:
                received_at = received_at.replace(tzinfo=timezone.utc)
            else:
                received_at = received_at.astimezone(timezone.utc)
        except Exception:
            received_at = datetime.now(timezone.utc)
    else:
        received_at = datetime.now(timezone.utc)

    # --- body + attachments ---
    body, attachments = _extract_body_and_attachments(msg)

    return ParsedEmail(
        uid=uid,
        message_id=_header(msg, "Message-ID") or f"<uid-{uid}@local>",
        sender_name=sender_name,
        sender_email=sender_email,
        sender_domain=sender_domain,
        subject=subject,
        received_at=received_at,
        body=body,
        attachment_names=attachments,
    )


# ─── Text chunking ────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple word-aware sliding-window chunker.

    Why not LangChain's RecursiveCharacterTextSplitter?
    For short-to-medium emails this is plenty, has zero overhead, and we
    keep the dep surface small.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break on a paragraph or sentence boundary near `end`.
        if end < len(text):
            window = text[start:end]
            for sep in ["\n\n", ". ", "\n", " "]:
                idx = window.rfind(sep)
                if idx != -1 and idx > chunk_size // 2:
                    end = start + idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return [c for c in chunks if c]


# ─── Deterministic point IDs ──────────────────────────────────────────────────
def make_point_id(uid: int, chunk_index: int) -> int:
    """
    Generate a deterministic 63-bit integer ID per (uid, chunk).

    Qdrant accepts either UUIDs or unsigned 64-bit ints; we pick ints
    because they're cheaper and we can rebuild them at any time.
    """
    h = hashlib.blake2b(f"{uid}:{chunk_index}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << 63) - 1)


def build_chunk_payload(em: ParsedEmail, chunk_index: int, chunk_text_: str) -> dict:
    """Metadata payload stored alongside each vector."""
    return {
        "uid": em.uid,
        "message_id": em.message_id,
        "sender_name": em.sender_name,
        "sender_email": em.sender_email,
        "sender_domain": em.sender_domain,
        "subject": em.subject,
        "received_at_ts": int(em.received_at.timestamp()),
        "received_at_iso": em.received_at.isoformat(),
        "date_str": em.received_at.strftime("%Y-%m-%d"),
        "chunk_index": chunk_index,
        "text": chunk_text_,
        "attachments": em.attachment_names,
    }
