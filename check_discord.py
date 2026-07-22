"""Find every Discord chunk in the index, no matter where it is."""
from common import load_config, get_qdrant
from qdrant_client.http import models as qmodels

cfg = load_config()
q = get_qdrant(cfg)

# Use a server-side filter so we don't scan every point manually.
# This catches anything whose sender_domain contains "discord".
discord_filter = qmodels.Filter(
    must=[
        qmodels.FieldCondition(
            key="sender_domain",
            match=qmodels.MatchText(text="discord"),
        )
    ]
)

all_hits = []
next_offset = None
while True:
    pts, next_offset = q.scroll(
        collection_name=cfg.qdrant_collection,
        scroll_filter=discord_filter,
        limit=200,
        offset=next_offset,
        with_payload=True,
    )
    all_hits.extend(pts)
    if next_offset is None:
        break

print(f"\nTotal Discord chunks indexed: {len(all_hits)}\n")

# Deduplicate by (sender + subject + date)
seen = set()
unique = []
for p in all_hits:
    payload = p.payload or {}
    key = (payload.get("sender_email"), payload.get("subject"), payload.get("date_str"))
    if key in seen:
        continue
    seen.add(key)
    unique.append(payload)

# Sort newest-first
unique.sort(key=lambda p: p.get("received_at_ts", 0), reverse=True)

print(f"Unique Discord emails: {len(unique)}\n")
print("Most recent 15:")
print("-" * 80)
for p in unique[:15]:
    print(f"  date={p.get('date_str')} | from={p.get('sender_email')}")
    print(f"  subject: {(p.get('subject') or '')[:80]}")
    print()