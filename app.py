"""AI-Generated Text Region Detection — demo app (Member 4).

Labels every region of an article as human-written or AI-generated.
Run:  streamlit run app.py
"""
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pipeline import (
    AI,
    DEFAULT_MIN_LEN,
    DEFAULT_THRESHOLD,
    HUMAN,
    _load_segmenter,
    analyze_article,
    component_info,
    load_model,
)
from segmenter import _runs, char_labels_from_segments, true_char_labels

st.set_page_config(page_title="AI Text Region Detector", page_icon="🔍", layout="wide")

SAMPLES_DIR = Path(__file__).parent / "samples"
COLORS = {HUMAN: "#d7f5dd", AI: "#ffd6d6", None: "#eeeeee"}
NAME = {HUMAN: "Human", AI: "AI"}
VERDICT = {"human": "Human only", "ai": "AI only", "mixed": "Mixed", "empty": "—"}
ISFAKE_TO_VERDICT = {0: "human", 1: "mixed", 2: "ai"}
SAMPLE_LABELS = {"1_human": "Human-written", "2_ai": "AI-generated", "3_mixed_a": "Mixed (A)", "4_mixed_b": "Mixed (B)"}


# ---------------------------------------------------------------------------
# Samples & ground truth
# ---------------------------------------------------------------------------
def load_samples():
    if not SAMPLES_DIR.exists():
        return {}
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(SAMPLES_DIR.glob("*.txt"))}


def find_truth(text: str, samples: dict):
    """If `text` is an unedited sample article, return its ground truth from answers.csv."""
    ans_path = SAMPLES_DIR / "answers.csv"
    if not ans_path.exists():
        return None
    ans = pd.read_csv(ans_path, dtype={"id": str}).set_index("sample")
    for name, body in samples.items():
        if body == text and name in ans.index:
            row = ans.loc[name]
            isfake = int(row.isfake)
            labels = true_char_labels(len(text), isfake, row.nchar_real)
            return {
                "verdict": ISFAKE_TO_VERDICT[isfake],
                "char_labels": labels,
                "regions": [{"start": s, "end": e, "label": l} for s, e, l in _runs(labels)],
            }
    return None


def chunk_truth(start, end, truth):
    """Chunk label from character truth; None if the chunk spans both (Member 1's rule)."""
    vals = set(truth["char_labels"][start:end].tolist())
    return vals.pop() if len(vals) == 1 else None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_regions(text, regions) -> str:
    spans = "".join(
        f'<span style="background:{COLORS[r["label"]]};color:#111;">'
        f'{html.escape(text[r["start"]:r["end"]])}</span>'
        for r in regions
    )
    return (
        '<div style="line-height:2;white-space:pre-wrap;border:1px solid #ccc;'
        f'border-radius:8px;padding:1rem;">{spans}</div>'
    )


def render_chunk_strip(pred, truth_labels=None) -> str:
    """One box per chunk. Top row = prediction, bottom row = ground truth."""

    def row(title, labels, other=None):
        cells = []
        for i, lab in enumerate(labels):
            wrong = other is not None and other[i] is not None and other[i] != lab
            border = "3px solid #e8a400" if wrong else "1px solid #999"
            txt = "–" if lab is None else ("AI" if lab == AI else "H")
            cells.append(
                f'<div title="chunk {i}" style="width:34px;height:26px;background:{COLORS[lab]};'
                f"border:{border};border-radius:4px;color:#111;font-size:11px;"
                f'display:flex;align-items:center;justify-content:center;">{txt}</div>'
            )
        return (
            '<div style="display:flex;gap:4px;align-items:center;margin:4px 0;flex-wrap:wrap;">'
            f'<div style="width:70px;font-size:12px;">{title}</div>{"".join(cells)}</div>'
        )

    out = row("Predicted", pred, truth_labels)
    if truth_labels is not None:
        out += row("Truth", truth_labels)
    return f'<div style="margin-bottom:1rem;">{out}</div>'


LEGEND = (
    f'<span style="background:{COLORS[HUMAN]};color:#111;padding:2px 6px">Human</span> '
    f'<span style="background:{COLORS[AI]};color:#111;padding:2px 6px">AI</span>'
)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "AI threshold", 0.30, 0.70, DEFAULT_THRESHOLD, 0.05,
        help="A character is AI if its average P(AI) is at least this value.",
    )
    min_len = st.slider(
        "Minimum region length", 0, 300, DEFAULT_MIN_LEN, 50,
        help="Shorter regions are treated as noise and merged into their neighbours. 0 = off.",
    )
    try:
        info = component_info()
        st.caption(f"Model: {info['model']}  \nSegmenter: {info['segmenter']}")
    except Exception as e:  # pragma: no cover
        st.warning(str(e))

# ---------------- Main ----------------
st.title("🔍 AI Text Region Detector")
st.caption(
    "Finds which parts of a Japanese news article were written by a human and which by AI (GPT-2). "
    "Not validated on modern LLMs such as ChatGPT."
)

samples = load_samples()
choice = st.selectbox(
    "Try a sample (test set, with answer)",
    ["—"] + list(samples),
    format_func=lambda k: SAMPLE_LABELS.get(k, k),
)
text = st.text_area("Article", value=samples.get(choice, ""), height=180, key=f"txt_{choice}")

# Keep showing the result while the sliders move (re-analyse on every rerun).
if st.button("Analyze", type="primary"):
    st.session_state["analyzed_text"] = text
if st.session_state.get("analyzed_text") is not None and st.session_state["analyzed_text"] == text:
    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    res = analyze_article(text, threshold=threshold, min_len=min_len)
    for w in res.warnings:
        st.warning(w)
    if res.verdict == "empty":
        st.stop()

    truth = find_truth(text, samples)
    pred_chars = char_labels_from_segments(res.regions, len(text))

    # ---------- Summary ----------
    st.divider()
    cols = st.columns(3)
    cols[0].metric("Verdict", VERDICT[res.verdict])
    cols[1].metric("AI share", f"{res.ai_chars / len(text):.0%}")
    if truth:
        acc = float((pred_chars == truth["char_labels"]).mean())
        cols[2].metric("Character accuracy", f"{acc:.0%}")
        ok = truth["verdict"] == res.verdict
        st.markdown(
            f"{'✅' if ok else '❌'} Answer: **{VERDICT[truth['verdict']]}**"
            + ("" if ok else f" (predicted {VERDICT[res.verdict]})")
        )

    # ---------- Highlighted text ----------
    if truth:
        left, right = st.columns(2)
        left.markdown("**Predicted**")
        left.markdown(render_regions(text, res.regions), unsafe_allow_html=True)
        right.markdown("**Answer**")
        right.markdown(render_regions(text, truth["regions"]), unsafe_allow_html=True)
    else:
        st.markdown(render_regions(text, res.regions), unsafe_allow_html=True)
    st.markdown(LEGEND, unsafe_allow_html=True)

    # ---------- Why ----------
    st.markdown("#### Why")
    for r in res.regions:
        st.markdown(
            f"- chars {r['start']}–{r['end']}: **{NAME[r['label']]}** (avg P(AI) {r['score']:.2f})"
        )
    if truth:
        wrong = [(s, e) for s, e, w in _runs((pred_chars != truth["char_labels"]).astype(int)) if w]
        for s, e in wrong:
            st.markdown(
                f"- ⚠️ chars {s}–{e} were **{NAME[int(truth['char_labels'][s])]}**, "
                f"predicted **{NAME[int(pred_chars[s])]}**"
            )

    # ---------- Details ----------
    with st.expander("Details"):
        segment_article, _ = _load_segmenter()
        raw = segment_article(text, model, threshold=threshold, min_len=0)
        st.markdown(
            f"1. Split into {len(res.chunks)} chunks of 100 characters (stride 50) and scored each with the classifier.\n"
            f"2. Each character's score = mean P(AI) of the chunks that contain it; ≥ {threshold:.2f} → AI.\n"
            f"3. Regions shorter than {min_len} characters were merged into neighbours "
            f"({len(raw)} → {len(res.regions)} regions)."
        )
        truth_chunks = [chunk_truth(s, e, truth) for s, e, _ in res.chunks] if truth else None
        st.markdown(render_chunk_strip(res.chunk_labels, truth_chunks), unsafe_allow_html=True)
        if truth:
            st.caption("Orange border = chunk disagrees with the answer; – = chunk spans the true boundary.")
        df = pd.DataFrame(
            {
                "start": [c[0] for c in res.chunks],
                "end": [c[1] for c in res.chunks],
                "P(AI)": np.round(res.ai_proba, 3),
                "predicted": [NAME[l] for l in res.chunk_labels],
            }
        )
        if truth_chunks is not None:
            df["answer"] = ["–" if l is None else NAME[l] for l in truth_chunks]
        df["text"] = [c[2] for c in res.chunks]
        st.line_chart(df.set_index("start")["P(AI)"], height=180)
        st.dataframe(df, width="stretch", hide_index=True)
