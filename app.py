"""
Hasamex AI Engineer - Technical Case
Transcript Insight Explorer

An app that analyses 3 expert-call transcripts and:
  1. Answers the interview-guide questions per expert, with exact quotes + timestamps
  2. Identifies common themes and disagreements across all experts
  3. Lets the user ask free-form questions across all transcripts

Design goals (see README.md for full write-up):
  - No hallucination: the model is only ever given the transcript text itself,
    and every quote returned by the model is verified against the source
    transcript before being shown. Unverifiable quotes are flagged, not hidden.
  - Every answer is traceable to a transcript + timestamp + expert name.
  - Simple, single-file Streamlit app; easy to read and to demo.
"""

import os
import re
import json
import glob
import streamlit as st
from openai import OpenAI

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

# --------------------------------------------------------------------------
# Config / constants
# --------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GUIDE_PATH = os.path.join(DATA_DIR, "interview_guide.txt")

PROVIDER_MODELS = {
    "Groq (free)": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    "Google Gemini (free)": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "OpenAI (paid)": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

st.set_page_config(page_title="Transcript Insight Explorer", layout="wide")


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

def parse_transcript(raw_text: str):
    """Parse a transcript file into header info + a list of timestamped turns.

    Expected format (as given in the case pack):
        Expert 1 - Dr. Jean Martin
        Role: Head of Urology
        Market: France

        00:00
        Interviewer: ...

        00:18
        Dr. Martin: ...
    """
    lines = raw_text.strip().splitlines()

    header = {"name": "Unknown", "role": "", "market": ""}
    body_start = 0
    for i, line in enumerate(lines[:6]):
        if line.lower().startswith("expert"):
            # "Expert 1 - Dr. Jean Martin"
            parts = line.split("-", 1)
            if len(parts) == 2:
                header["name"] = parts[1].strip()
        elif line.lower().startswith("role:"):
            header["role"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("market:"):
            header["market"] = line.split(":", 1)[1].strip()
        if line.strip() == "" and i > 0:
            body_start = i
            break

    turns = []
    ts_re = re.compile(r"^\d{1,2}:\d{2}$")
    i = body_start
    current_ts = None
    while i < len(lines):
        line = lines[i].strip()
        if ts_re.match(line):
            current_ts = line
        elif line and ":" in line and current_ts is not None:
            speaker, _, text = line.partition(":")
            turns.append({
                "timestamp": current_ts,
                "speaker": speaker.strip(),
                "text": text.strip(),
            })
            current_ts = None
        i += 1

    return header, turns


def load_default_transcripts():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "transcript_*.txt")))
    transcripts = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
        header, turns = parse_transcript(raw)
        transcripts.append({
            "filename": os.path.basename(p),
            "header": header,
            "turns": turns,
            "raw": raw,
        })
    return transcripts


def load_guide():
    with open(GUIDE_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    questions = re.findall(r"^\s*\d+\.\s*(.+)$", raw, flags=re.MULTILINE)
    return raw, questions


def transcript_for_prompt(t):
    """Render a transcript back to plain text for use in a prompt."""
    lines = [f"Expert: {t['header']['name']}",
             f"Role: {t['header']['role']}",
             f"Market: {t['header']['market']}", ""]
    for turn in t["turns"]:
        lines.append(f"[{turn['timestamp']}] {turn['speaker']}: {turn['text']}")
    return "\n".join(lines)


def verify_quote(quote: str, transcript_raw: str) -> bool:
    """Check that a quote the model produced actually appears in the transcript."""
    if not quote:
        return False
    norm_quote = re.sub(r"\s+", " ", quote).strip().lower()
    norm_source = re.sub(r"\s+", " ", transcript_raw).strip().lower()
    return norm_quote in norm_source


def get_client():
    """Returns (provider, client_or_none). client is None if no key is set yet."""
    provider = st.session_state.get("provider", "Google Gemini (free)")
    env_var = {
        "Google Gemini (free)": "GEMINI_API_KEY",
        "Groq (free)": "GROQ_API_KEY",
        "OpenAI (paid)": "OPENAI_API_KEY",
    }[provider]
    api_key = st.session_state.get("api_key") or os.environ.get(env_var)
    if not api_key:
        return provider, None
    if provider.startswith("Google"):
        if genai is None:
            st.error("google-genai is not installed. Run: pip install google-genai")
            return provider, None
        return provider, genai.Client(api_key=api_key)
    if provider.startswith("Groq"):
        return provider, OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return provider, OpenAI(api_key=api_key)


def call_json(provider, client, model, system, user):
    if provider.startswith("Google"):
        resp = client.models.generate_content(
            model=model,
            contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return json.loads(resp.text)
    else:
        # Both Groq and OpenAI use the OpenAI-compatible chat completions API
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

GUIDE_ANSWER_SYSTEM = """You are an analyst answering interview-guide questions using ONLY the \
provided call transcript. Never use outside knowledge and never invent information that is not \
in the transcript. If the transcript does not clearly answer a question, say so explicitly.

For each question, return:
- "answer": a concise 1-3 sentence answer grounded in the transcript
- "quote": the SHORTEST exact, verbatim substring from the transcript that best supports the answer \
(copy it exactly, do not paraphrase, do not add ellipses)
- "timestamp": the timestamp shown immediately before that quoted line in the transcript

Respond ONLY with a JSON object of the form:
{"answers": [{"question": "...", "answer": "...", "quote": "...", "timestamp": "MM:SS"}, ...]}
"""

THEMES_SYSTEM = """You are an analyst comparing three expert-call transcripts about the same topic. \
Using ONLY the transcripts provided, identify:
- "common_themes": points where 2 or more experts broadly agree
- "disagreements": points where experts give notably different views, numbers, or emphasis

For every theme or disagreement, include which experts (by name) said it, and one short supporting \
quote with timestamp per expert cited. Do not invent anything not present in the transcripts.

Respond ONLY with a JSON object of the form:
{
  "common_themes": [{"theme": "...", "evidence": [{"expert": "...", "quote": "...", "timestamp": "MM:SS"}]}],
  "disagreements": [{"topic": "...", "evidence": [{"expert": "...", "quote": "...", "timestamp": "MM:SS"}]}]
}
"""

QA_SYSTEM = """You are answering a user's question using ONLY the three provided call transcripts. \
Never use outside knowledge. If the transcripts do not contain the answer, say so clearly instead \
of guessing.

Respond ONLY with a JSON object of the form:
{"answer": "...", "citations": [{"expert": "...", "quote": "...", "timestamp": "MM:SS"}]}
Include one citation per transcript that supports your answer, where relevant. Quotes must be exact \
verbatim substrings from the transcripts.
"""


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Settings")
st.session_state["provider"] = st.sidebar.selectbox(
    "AI provider", list(PROVIDER_MODELS.keys()), index=0,
    help="Gemini and Groq both have genuine free tiers. OpenAI requires paid credits.",
)
provider_name = st.session_state["provider"]
KEY_INFO = {
    "Google Gemini (free)": ("Gemini API key", "GEMINI_API_KEY",
                              "Get a free key at aistudio.google.com/apikey — no credit card needed."),
    "Groq (free)": ("Groq API key", "GROQ_API_KEY",
                     "Get a free key at console.groq.com/keys — no credit card needed."),
    "OpenAI (paid)": ("OpenAI API key", "OPENAI_API_KEY",
                       "Requires billing credit at platform.openai.com."),
}
key_label, key_env, key_hint = KEY_INFO[provider_name]

st.session_state["api_key"] = st.sidebar.text_input(
    key_label, type="password",
    value=st.session_state.get("api_key", os.environ.get(key_env, "")),
    help="Your key is only kept in this session, never stored.",
)
st.sidebar.caption(key_hint)
model = st.sidebar.selectbox("Model", PROVIDER_MODELS[provider_name], index=0)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Bundled with 3 sample transcripts (France, Germany, UK) and the interview guide "
    "from the case pack. Upload your own .txt transcripts below to replace them."
)
uploaded = st.sidebar.file_uploader(
    "Upload transcripts (.txt)", type=["txt"], accept_multiple_files=True
)

if "transcripts" not in st.session_state:
    st.session_state["transcripts"] = load_default_transcripts()
if "guide_questions" not in st.session_state:
    _, st.session_state["guide_questions"] = load_guide()

if uploaded:
    new_transcripts = []
    for f in uploaded:
        raw = f.read().decode("utf-8")
        header, turns = parse_transcript(raw)
        new_transcripts.append({"filename": f.name, "header": header, "turns": turns, "raw": raw})
    st.session_state["transcripts"] = new_transcripts

transcripts = st.session_state["transcripts"]
guide_questions = st.session_state["guide_questions"]

st.title("🔎 Transcript Insight Explorer")
st.caption("Hasamex AI Engineer case study — analyses 3 expert-call transcripts using AI, "
           "grounded strictly in the source text.")

with st.expander("Loaded transcripts", expanded=False):
    for t in transcripts:
        st.markdown(f"**{t['header']['name']}** — {t['header']['role']} ({t['header']['market']}) "
                    f"· {len(t['turns'])} turns · `{t['filename']}`")

provider, client = get_client()
if client is None:
    st.warning(f"Enter your {key_label} in the sidebar to run the analysis.", icon="🔑")

tab1, tab2, tab3 = st.tabs(["📋 Guide Answers", "🧭 Themes & Disagreements", "💬 Ask a Question"])

# --------------------------------------------------------------------------
# Tab 1: Guide Answers
# --------------------------------------------------------------------------
with tab1:
    st.subheader("Interview guide answers, per expert")
    st.caption("Each answer shows the exact quote and timestamp it was drawn from. "
               "Quotes are automatically checked against the transcript; unverified quotes are flagged.")

    if st.button("Generate guide answers", disabled=client is None):
        results = {}
        with st.spinner("Analysing transcripts..."):
            for t in transcripts:
                user_prompt = (
                    f"TRANSCRIPT:\n{transcript_for_prompt(t)}\n\n"
                    f"QUESTIONS:\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(guide_questions))
                )
                try:
                    data = call_json(provider, client, model, GUIDE_ANSWER_SYSTEM, user_prompt)
                    results[t["filename"]] = data.get("answers", [])
                except Exception as e:
                    results[t["filename"]] = {"error": str(e)}
        st.session_state["guide_results"] = results

    results = st.session_state.get("guide_results")
    if results:
        for t in transcripts:
            st.markdown(f"### {t['header']['name']} ({t['header']['market']})")
            answers = results.get(t["filename"])
            if isinstance(answers, dict) and "error" in answers:
                st.error(answers["error"])
                continue
            for a in answers or []:
                verified = verify_quote(a.get("quote", ""), t["raw"])
                badge = "✅ verified" if verified else "⚠️ could not verify quote"
                st.markdown(f"**Q: {a.get('question')}**")
                st.write(a.get("answer"))
                st.markdown(
                    f"> \"{a.get('quote')}\" — *{t['header']['name']}, {a.get('timestamp')}* &nbsp; `{badge}`"
                )
            st.markdown("---")

# --------------------------------------------------------------------------
# Tab 2: Themes & Disagreements
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Common themes and disagreements across all 3 experts")

    if st.button("Analyse themes & disagreements", disabled=client is None):
        all_transcripts_text = "\n\n---\n\n".join(transcript_for_prompt(t) for t in transcripts)
        user_prompt = f"TRANSCRIPTS:\n{all_transcripts_text}"
        with st.spinner("Comparing experts..."):
            try:
                data = call_json(provider, client, model, THEMES_SYSTEM, user_prompt)
                st.session_state["themes_result"] = data
            except Exception as e:
                st.session_state["themes_result"] = {"error": str(e)}

    themes_result = st.session_state.get("themes_result")
    if themes_result:
        if "error" in themes_result:
            st.error(themes_result["error"])
        else:
            st.markdown("#### 🤝 Common themes")
            for theme in themes_result.get("common_themes", []):
                st.markdown(f"**{theme.get('theme')}**")
                for ev in theme.get("evidence", []):
                    ok = any(verify_quote(ev.get("quote", ""), t["raw"]) for t in transcripts)
                    badge = "✅" if ok else "⚠️"
                    st.markdown(f"- {badge} *{ev.get('expert')}*, {ev.get('timestamp')}: \"{ev.get('quote')}\"")
                st.write("")

            st.markdown("#### ⚖️ Disagreements")
            for dis in themes_result.get("disagreements", []):
                st.markdown(f"**{dis.get('topic')}**")
                for ev in dis.get("evidence", []):
                    ok = any(verify_quote(ev.get("quote", ""), t["raw"]) for t in transcripts)
                    badge = "✅" if ok else "⚠️"
                    st.markdown(f"- {badge} *{ev.get('expert')}*, {ev.get('timestamp')}: \"{ev.get('quote')}\"")
                st.write("")

# --------------------------------------------------------------------------
# Tab 3: Ask a Question
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Ask a question across all transcripts")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, content in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(content)

    question = st.chat_input("Ask something about adoption, barriers, ROI, training, timelines...",
                              disabled=client is None)
    if question:
        st.session_state["chat_history"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        all_transcripts_text = "\n\n---\n\n".join(transcript_for_prompt(t) for t in transcripts)
        user_prompt = f"TRANSCRIPTS:\n{all_transcripts_text}\n\nQUESTION:\n{question}"
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    data = call_json(provider, client, model, QA_SYSTEM, user_prompt)
                    answer = data.get("answer", "")
                    citations = data.get("citations", [])
                    st.markdown(answer)
                    for c in citations:
                        ok = any(verify_quote(c.get("quote", ""), t["raw"]) for t in transcripts)
                        badge = "✅" if ok else "⚠️"
                        st.markdown(f"- {badge} *{c.get('expert')}*, {c.get('timestamp')}: \"{c.get('quote')}\"")
                    full_reply = answer + "\n\n" + "\n".join(
                        f"- {c.get('expert')}, {c.get('timestamp')}: \"{c.get('quote')}\"" for c in citations
                    )
                    st.session_state["chat_history"].append(("assistant", full_reply))
                except Exception as e:
                    st.error(str(e))
