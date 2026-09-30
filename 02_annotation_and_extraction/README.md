# Chatbot extraction — input and output

Everything lives in `C:\Users\HP\Documents\Claude\news-tkg-chat\`.

## INPUT — what you paste into the chatbot

| # | File | When |
|---|---|---|
| 1 | `PROMPT.md` | **once**, at the start of a chat session |
| 2 | `batches\batch-001.txt` … `batch-086.txt` | **one per turn**, in order |

86 batches, 15 articles each, 1,280 articles total. Ordered so the articles most
likely to create cross-document links come first — stopping early still leaves
you with the useful half.

## OUTPUT — where you save what the chatbot replies

| Save to | As |
|---|---|
| `responses\batch-001.json` | the reply to batch-001, and so on |

Save the reply **verbatim**. Leading prose and ``` fences are stripped
automatically; you do not need to clean anything up. The filename number must
match the batch number — that is how missing articles are detected.

## Then run

```bash
cd "C:/Users/HP/Documents/Claude/news-tkg-chat"
python ingest_chat_output.py
```

Reads everything in `responses\`, validates it, and writes one checkpoint per
article to `checkpoints-chat\`. Safe to re-run at any time; it reprocesses all
responses present and overwrites cleanly.

## Directory map

```
news-tkg-chat\
├── PROMPT.md              INPUT  — paste once
├── batches\               INPUT  — paste one file per turn
│   ├── batch-001.txt
│   └── … batch-086.txt
├── responses\             OUTPUT — you save chatbot replies here
│   ├── batch-001.json
│   └── …
├── checkpoints-chat\      GENERATED — one JSON per article, pipeline format
├── batch-index.json       GENERATED — which article ids are in which batch
├── ingest-report.json     GENERATED — validation results after each ingest
├── build_chat_batches.py  regenerates batches\
└── ingest_chat_output.py  responses\ → checkpoints-chat\
```

You only ever touch two things: **read** from `batches\`, **write** to
`responses\`. Everything else is generated.

## What the ingester rejects

A chatbot drifts from a schema in ways a local runner does not, so nothing is
trusted:

| Check | On failure |
|---|---|
| evidence quote is an exact substring of the article body | article rejected |
| no event date later than the publication date | article rejected |
| event_kind agrees with event_subtype | event dropped, counted |
| relation slots resolve; policy relations carry a policy_id | relation dropped, counted |
| article ids match the batch that was sent | missing/extra flagged |

Check `ingest-report.json` after the first batch before committing to all 86. A
high rejection rate means the prompt needs adjusting, not that the articles are
bad.

## A caveat worth keeping in mind

This is a different extraction regime from the 955 articles already done with
local qwen3:8b — different model, and the prompt adds two fields the local run
never had (`east_java_basis`, `granularity_basis`). Every checkpoint is tagged
`extraction_route: chatbot` and every downstream edge carries
`extraction_method`, so the two can be held apart. **Do not pool them when
reporting results.**
