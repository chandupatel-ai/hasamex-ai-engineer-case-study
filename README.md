# Transcript Insight Explorer — Hasamex AI Engineer Case Study

A small Streamlit app that analyses 3 expert-call transcripts (robotic surgery adoption
in France, Germany, and the UK) and:

1. Answers the interview-guide questions for each expert, with an exact quote + timestamp
2. Extracts common themes and disagreements across all 3 experts
3. Lets you ask free-form questions across all transcripts, with citations

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then, in the sidebar, pick a provider and paste in the matching API key.

**This submission was built and tested with Groq** (free tier, no credit card, very fast —
runs OpenAI's open-weight GPT-OSS models). Get a free key at
[console.groq.com/keys](https://console.groq.com/keys), select "Groq (free)" in the
sidebar, and paste it in.

The app also supports two other providers as drop-in alternatives, useful if Groq's free-tier
rate limits are ever hit:

- **Google Gemini (free)** — genuine free tier, no credit card. Get a key at
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
- **OpenAI (paid)** — requires billing credit on your OpenAI account. Get a key at
  [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

You can also set `GEMINI_API_KEY`, `GROQ_API_KEY`, or `OPENAI_API_KEY` as an environment
variable before launching instead of typing it into the sidebar.

The app ships with the 3 sample transcripts and the interview guide already loaded — no
upload needed to try it, but you can upload your own `.txt` transcripts in the same
`[MM:SS] Speaker: text` style to replace them.

## Architecture

- **Single-file Streamlit app** (`app.py`) — no database, no backend server. Transcripts
  are plain `.txt` files parsed into `(timestamp, speaker, text)` turns.
- **No retrieval / vector DB for this scale.** With only 3 short transcripts (~1,500 words
  total), the entire transcript text fits comfortably in a single LLM prompt. Adding a
  retrieval layer here would add complexity and a new failure mode (missed chunks) without
  any real benefit. See "Scaling to 30+ transcripts" below for how this changes at scale.
- **Model**: `gemini-2.5-flash` by default (free tier, fast, accurate enough for this
  extraction task). Groq (`openai/gpt-oss-120b`) and OpenAI (`gpt-4o-mini`, `gpt-4o`,
  `gpt-4.1-mini`) are also supported via a provider switch in the sidebar. Groq is called
  through its OpenAI-compatible endpoint, so the same code path (and JSON schema) handles
  both Groq and OpenAI.
- **Structured outputs**: every LLM call uses `response_format={"type": "json_object"}`
  and a strict schema (answer/quote/timestamp, or theme/evidence), so the UI can render
  results reliably without fragile text parsing.

## How hallucination is reduced

1. **Grounding**: the model is only ever given the raw transcript text — never asked to
   use outside/general knowledge. System prompts explicitly say "if the transcript does
   not answer this, say so" rather than guess.
2. **Verbatim quote requirement**: the model must return the *exact* substring it is
   citing, not a paraphrase.
3. **Automatic quote verification**: after the model responds, the app checks (in code,
   not by trusting the model) that the returned quote actually appears in the source
   transcript text (`verify_quote()` in `app.py`, whitespace/case-normalized substring
   match). Verified quotes get a ✅, anything that doesn't match gets a ⚠️ so a
   fabricated or mangled quote is visibly flagged rather than silently trusted.
4. **Temperature 0** for all analysis calls, to minimize variance/invention.

## How citations/timestamps work

Each transcript line is parsed with its preceding `MM:SS` timestamp. The model is asked
to return the timestamp that immediately precedes the quoted line, so every answer,
theme, and chat response can point back to the exact moment in the exact transcript it
came from.

## Scaling from 3 transcripts to 30+

At 3 transcripts everything fits in one prompt. At 30+ transcripts (or longer calls),
the approach would change to a standard RAG pipeline:

- **Chunk** each transcript (e.g. by timestamped turn or ~200-token windows with overlap),
  keeping `(transcript_id, timestamp, speaker, text)` metadata on every chunk.
- **Embed** chunks (e.g. `text-embedding-3-small`) into a vector store (pgvector, Chroma,
  or a managed vector DB).
- **Retrieve** the top-k relevant chunks per question (guide question or free-form user
  question) instead of sending full transcripts, then run the same "answer + verbatim
  quote + timestamp" prompt over just the retrieved chunks.
- **Cross-transcript themes at scale** would move from "paste all transcripts into one
  prompt" to a map-reduce pattern: extract per-transcript summaries/positions first, then
  a second pass clusters/compares those summaries into themes and disagreements, since
  30+ full transcripts won't fit in one context window.
- Quote verification stays the same (still checked in code against the source text) —
  that check is what keeps the system honest regardless of scale.

## Submission checklist

- [x] Working app / run instructions — see "Quick start" above
- [x] Source code — `app.py`
- [x] Short README — this file
