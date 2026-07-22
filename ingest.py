"""
ingest.py
─────────
Real-time Gmail ingestor for the Email RAG Agent.

What it does
============
  1. On first run: BACKFILLS every message from the entire mailbox into Qdrant.
  2. After backfill: enters IMAP IDLE mode and listens for new mail in real
     time. The moment Gmail pushes an EXISTS notification, we fetch the new
     UID(s), parse, embed, and upsert into Qdrant.

Run it in its own terminal:

    python ingest.py

Stop with Ctrl+C.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import Iterable

from imapclient import IMAPClient
from imapclient.exceptions import IMAPClientError
from qdrant_client.http import models as qmodels
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from common import (
    Config,
    ParsedEmail,
    build_chunk_payload,
    chunk_text,
    embed_texts,
    ensure_collection,
    get_embedder,
    get_qdrant,
    load_config,
    log,
    make_point_id,
    parse_email,
)

# IMAP IDLE must be refreshed periodically. Gmail closes idle sessions after
# ~29 minutes; we re-issue IDLE every 14 minutes to be safe.
IDLE_REFRESH_SECONDS = 14 * 60

# How many UIDs we fetch from the server per network round-trip during backfill.
BACKFILL_BATCH = 25


# ─── IMAP helpers ─────────────────────────────────────────────────────────────
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((IMAPClientError, OSError)),
)
def connect_imap(cfg: Config) -> IMAPClient:
    """Open a fresh authenticated IMAP connection with retry/backoff."""
    log("📬 [IMAP]", f"Connecting to {cfg.imap_host}:{cfg.imap_port} ...", "blue")
    client = IMAPClient(cfg.imap_host, port=cfg.imap_port, ssl=True, use_uid=True)
    client.login(cfg.email_address, cfg.email_password)
    client.select_folder(cfg.imap_mailbox, readonly=True)
    log("📬 [IMAP]", f"Logged in as {cfg.email_address}, folder={cfg.imap_mailbox}", "blue")
    return client


def get_already_ingested_uids(qclient, cfg: Config) -> set[int]:
    """Scroll Qdrant to find UIDs we've already indexed (so we never re-embed)."""
    seen: set[int] = set()
    next_offset = None
    while True:
        points, next_offset = qclient.scroll(
            collection_name=cfg.qdrant_collection,
            with_payload=["uid"],
            with_vectors=False,
            limit=1000,
            offset=next_offset,
        )
        for p in points:
            uid = p.payload.get("uid") if p.payload else None
            if uid is not None:
                seen.add(int(uid))
        if next_offset is None:
            break
    return seen


def fetch_and_index_uids(
    imap: IMAPClient, qclient, cfg: Config, uids: Iterable[int]
) -> int:
    """Fetch raw RFC822 bytes for a batch of UIDs, parse, embed, and upsert."""
    uids = list(uids)
    if not uids:
        return 0

    # FETCH the raw messages.
    fetched = imap.fetch(uids, [b"RFC822"])

    all_points: list[qmodels.PointStruct] = []
    indexed = 0

    for uid in uids:
        msg_data = fetched.get(uid)
        if not msg_data:
            continue
        raw = msg_data.get(b"RFC822")
        if not raw:
            continue

        parsed = parse_email(raw, uid)
        if parsed is None:
            continue

        # Build the document we actually embed (subject helps a lot).
        document = f"From: {parsed.sender_name} <{parsed.sender_email}>\nSubject: {parsed.subject}\n\n{parsed.body}"
        chunks = chunk_text(document, cfg.chunk_size, cfg.chunk_overlap)
        if not chunks:
            continue

        vectors = embed_texts(chunks, cfg.embedding_model)

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            all_points.append(
                qmodels.PointStruct(
                    id=make_point_id(parsed.uid, i),
                    vector=vec,
                    payload=build_chunk_payload(parsed, i, chunk),
                )
            )

        indexed += 1
        log(
            "✉️  [Ingest]",
            f"UID {uid} | {parsed.sender_email} | {parsed.subject[:60]}",
            "green",
        )

    if all_points:
        # Upsert in small slices so a single batch never trips the HTTP timeout.
        # Retry once on transient timeouts before giving up on the batch.
        UPSERT_CHUNK = 64
        for i in range(0, len(all_points), UPSERT_CHUNK):
            slice_ = all_points[i : i + UPSERT_CHUNK]
            for attempt in range(3):
                try:
                    qclient.upsert(
                        collection_name=cfg.qdrant_collection,
                        points=slice_,
                        wait=True,
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        log("⚠️  [Qdrant]", f"Upsert failed after 3 attempts: {e}", "red")
                        raise
                    log("⚠️  [Qdrant]", f"Upsert timeout — retrying ({attempt + 1}/3) ...", "yellow")
                    time.sleep(2 * (attempt + 1))

    return indexed


# ─── Backfill ─────────────────────────────────────────────────────────────────
def backfill(imap: IMAPClient, qclient, cfg: Config) -> None:
    """
    Walk the entire mailbox once and index any UID not already present in Qdrant.
    """
    log("🚚 [Backfill]", "Listing all UIDs in mailbox ...", "yellow")
    all_uids = imap.search(["ALL"])
    log("🚚 [Backfill]", f"Mailbox contains {len(all_uids):,} messages.", "yellow")

    already = get_already_ingested_uids(qclient, cfg)
    todo = [u for u in all_uids if u not in already]

    if not todo:
        log("🚚 [Backfill]", "Nothing new to backfill — all caught up.", "yellow")
        return

    log("🚚 [Backfill]", f"{len(todo):,} new messages to index.", "yellow")

    total_indexed = 0
    for start in range(0, len(todo), BACKFILL_BATCH):
        batch = todo[start : start + BACKFILL_BATCH]
        total_indexed += fetch_and_index_uids(imap, qclient, cfg, batch)
        log(
            "🚚 [Backfill]",
            f"Progress: {min(start + BACKFILL_BATCH, len(todo))}/{len(todo)}",
            "yellow",
        )

    log("🚚 [Backfill]", f"Done. Indexed {total_indexed:,} messages.", "yellow")


# ─── IDLE listener ────────────────────────────────────────────────────────────
def listen_idle(imap: IMAPClient, qclient, cfg: Config) -> None:
    """
    Enter IMAP IDLE and react to new EXISTS notifications in real time.

    Gmail pushes a response of the form (msg_id, b'EXISTS') whenever a new
    message lands. We compare the new highest UID against the last seen
    UID and fetch only the delta.
    """
    log("👂 [IDLE]", "Entering real-time listener ...", "cyan")

    # Anchor the "last seen UID" so we only react to genuinely new mail.
    last_uids = imap.search(["ALL"])
    last_max_uid = max(last_uids) if last_uids else 0
    log("👂 [IDLE]", f"Anchor UID = {last_max_uid}", "cyan")

    while True:
        try:
            imap.idle()
            log("👂 [IDLE]", "Listening for new mail ... (refresh every 14 min)", "cyan")

            # Block until something happens, or until the refresh timer fires.
            responses = imap.idle_check(timeout=IDLE_REFRESH_SECONDS)
            imap.idle_done()

            if not responses:
                # Timeout — just loop and re-IDLE to keep the connection alive.
                log("👂 [IDLE]", "Refreshing IDLE session ...", "dim cyan")
                continue

            # New mail signaled by an EXISTS response.
            saw_new_mail = any(
                isinstance(r, tuple) and len(r) >= 2 and r[1] == b"EXISTS"
                for r in responses
            )
            if not saw_new_mail:
                continue

            # Find the new UIDs (anything strictly greater than our anchor).
            current = imap.search(["UID", f"{last_max_uid + 1}:*"])
            new_uids = [u for u in current if u > last_max_uid]
            if not new_uids:
                continue

            log("⚡ [IDLE]", f"New mail! UIDs: {new_uids}", "bold green")
            fetch_and_index_uids(imap, qclient, cfg, new_uids)
            last_max_uid = max(last_max_uid, *new_uids)

        except (IMAPClientError, OSError) as e:
            log("⚠️  [IDLE]", f"Connection blip ({e}). Reconnecting in 5s ...", "red")
            try:
                imap.logout()
            except Exception:
                pass
            time.sleep(5)
            imap = connect_imap(cfg)
            last_uids = imap.search(["ALL"])
            last_max_uid = max(last_uids) if last_uids else 0


# ─── Entrypoint ───────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config()

    # Pre-load the embedder so we don't pay the cost mid-IDLE.
    get_embedder(cfg.embedding_model)

    qclient = get_qdrant(cfg)
    ensure_collection(qclient, cfg)

    imap = connect_imap(cfg)

    # Graceful Ctrl+C
    def _shutdown(signum, frame):
        log("👋 [Ingest]", "Shutting down ...", "magenta")
        try:
            imap.logout()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    backfill(imap, qclient, cfg)
    listen_idle(imap, qclient, cfg)


if __name__ == "__main__":
    main()
