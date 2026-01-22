# app/app.py
from __future__ import annotations

import re
from datetime import datetime
from typing import List

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


# -----------------------------
# Page + styling
# -----------------------------
st.set_page_config(
    page_title="AI Job Matcher",
    page_icon="🧩",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2.0rem; }

.h1-center {
  text-align: center;
  font-size: 3.0rem;
  font-weight: 800;
  margin: 0.2rem 0 0.2rem 0;
}
.subtitle-center {
  text-align: center;
  color: rgba(255,255,255,0.72);
  margin-bottom: 1.25rem;
  font-size: 1.05rem;
}

div[data-testid="stButton"] > button {
  border-radius: 14px !important;
  padding: 0.7rem 1.2rem !important;
  font-weight: 700 !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.06) !important;
}
div[data-testid="stButton"] > button:hover {
  border-color: rgba(255,255,255,0.25) !important;
  background: rgba(255,255,255,0.08) !important;
}

.primary-btn div[data-testid="stButton"] > button {
  background: linear-gradient(90deg, rgba(255,88,88,0.95), rgba(255,140,0,0.95)) !important;
  border: 0 !important;
  color: white !important;
}
.primary-btn div[data-testid="stButton"] > button:hover {
  filter: brightness(1.05);
}

.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 1rem 1.1rem;
}
.card-title {
  color: rgba(255,255,255,0.72);
  font-weight: 700;
  font-size: 0.95rem;
  margin-bottom: 0.2rem;
}
.card-value {
  font-size: 2.2rem;
  font-weight: 900;
  margin: 0;
}

.chip-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}
.chip-green {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(60, 255, 160, 0.35);
  background: rgba(60, 255, 160, 0.10);
  color: rgba(200, 255, 225, 0.95);
  font-size: 0.92rem;
  font-weight: 650;
}
.chip-yellow {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255, 205, 90, 0.35);
  background: rgba(255, 205, 90, 0.10);
  color: rgba(255, 240, 205, 0.95);
  font-size: 0.92rem;
  font-weight: 650;
}

.section-title {
  font-size: 2.0rem;
  font-weight: 900;
  margin: 0.4rem 0 0.8rem 0;
}

.hr {
  height: 1px;
  background: rgba(255,255,255,0.10);
  border: none;
  margin: 1.4rem 0 1.4rem 0;
}

.small-muted { color: rgba(255,255,255,0.60); font-size: 0.92rem; }
.spacer-8 { height: 8px; }
.spacer-16 { height: 16px; }

/* Center download button row */
.center-row {
  display: flex;
  justify-content: center;
  margin-top: 10px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Session state helpers
# -----------------------------
def _init_state():
    st.session_state.setdefault("resume_text", "")
    st.session_state.setdefault("jd_text", "")
    st.session_state.setdefault("last_results", None)


def clear_inputs():
    st.session_state["resume_text"] = ""
    st.session_state["jd_text"] = ""
    st.session_state["last_results"] = None
    st.rerun()


_init_state()


# -----------------------------
# Model load (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# -----------------------------
# Skills / matching logic
# -----------------------------
SKILL_KEYWORDS = [
    "python", "sql", "excel", "power bi", "tableau", "pandas", "numpy", "scikit-learn",
    "statistics", "data visualization", "data cleaning", "feature engineering",
    "kpis", "stakeholder", "a/b testing", "reporting",
    "git", "github", "streamlit", "fastapi",
    "machine learning",
]

ALIASES = {
    "ab testing": "a/b testing",
    "a b testing": "a/b testing",
    "powerbi": "power bi",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "data viz": "data visualization",
    "kpi": "kpis",
    "kpi's": "kpis",
}

def normalize_text(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("\u2019", "'")
    t = re.sub(r"\s+", " ", t).strip()
    for k, v in ALIASES.items():
        t = t.replace(k, v)
    return t

def detect_skills(text: str) -> List[str]:
    t = normalize_text(text)
    found = []
    for skill in SKILL_KEYWORDS:
        s = normalize_text(skill)
        pattern = r"(?<!\w)" + re.escape(s) + r"(?!\w)"
        if re.search(pattern, t):
            found.append(skill)
    # stable de-dupe
    out, seen = [], set()
    for x in found:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def semantic_match_score(resume_text: str, jd_text: str) -> float:
    r = (resume_text or "").strip()
    j = (jd_text or "").strip()
    if not r or not j:
        return 0.0
    emb = model.encode([r, j], normalize_embeddings=True)
    score = cosine_similarity(emb[0], emb[1])
    score01 = max(0.0, min(1.0, (score + 1.0) / 2.0))
    return score01 * 100.0

def skills_coverage_score(resume_skills: List[str], jd_skills: List[str]) -> float:
    if not jd_skills:
        return 0.0
    inter = set(resume_skills).intersection(set(jd_skills))
    return (len(inter) / len(set(jd_skills))) * 100.0


# -----------------------------
# Minimal PDF generator (no dependencies)
# Creates a simple "text-only PDF" by writing PDF syntax directly.
# Works fine for a small report (our case).
# -----------------------------
def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

def text_report_to_simple_pdf_bytes(report_text: str) -> bytes:
    # Very small PDF builder: one page, Helvetica, lines of text.
    lines = report_text.splitlines()
    max_lines = 55  # keep to one page; report is short
    lines = lines[:max_lines]

    y_start = 760
    y_step = 13

    content_lines = ["BT", "/F1 10 Tf", "54 0 0 54 54 780 Tm"]  # not used, keep simple
    # We'll position each line with Td
    content_lines = ["BT", "/F1 10 Tf", "54 760 Td"]

    first = True
    y = y_start
    for i, line in enumerate(lines):
        safe = _pdf_escape(line)
        if first:
            content_lines.append(f"({safe}) Tj")
            first = False
        else:
            content_lines.append(f"0 -{y_step} Td")
            content_lines.append(f"({safe}) Tj")
        y -= y_step

    content_lines.append("ET")
    content_stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    # PDF objects
    objects = []

    def obj(n: int, data: bytes) -> None:
        objects.append((n, data))

    # 1: Catalog
    obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2: Pages
    obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # 3: Page
    obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
    # 4: Contents stream
    obj(4, b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content_stream), content_stream))
    # 5: Font
    obj(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Build xref
    out = bytearray()
    out.extend(b"%PDF-1.4\n")
    offsets = {0: 0}
    for n, data in objects:
        offsets[n] = len(out)
        out.extend(f"{n} 0 obj\n".encode("ascii"))
        out.extend(data)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(b"xref\n")
    out.extend(f"0 {len(objects)+1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for i in range(1, len(objects) + 1):
        off = offsets[i]
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    out.extend(b"trailer\n")
    out.extend(f"<< /Size {len(objects)+1} /Root 1 0 R >>\n".encode("ascii"))
    out.extend(b"startxref\n")
    out.extend(f"{xref_pos}\n".encode("ascii"))
    out.extend(b"%%EOF\n")
    return bytes(out)


def build_report(
    semantic_pct: float,
    coverage_pct: float,
    matched: List[str],
    missing: List[str],
    resume_skills: List[str],
    jd_skills: List[str],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("AI Job Matcher — Results Report")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append(f"Semantic Match Score: {semantic_pct:.1f}%")
    lines.append(f"Skills Coverage:      {coverage_pct:.1f}%")
    lines.append("")
    lines.append("Skills matched (in both):")
    lines += [f"  - {s}" for s in matched] if matched else ["  - (none)"]
    lines.append("")
    lines.append("Missing skills (in JD but not in resume):")
    lines += [f"  - {s}" for s in missing] if missing else ["  - (none)"]
    lines.append("")
    lines.append("Detected in JD:")
    lines.append(", ".join(jd_skills) if jd_skills else "(none)")
    lines.append("")
    lines.append("Detected in Resume:")
    lines.append(", ".join(resume_skills) if resume_skills else "(none)")
    lines.append("")
    lines.append("Note: Semantic score uses SentenceTransformers embeddings; skills are keyword-based (expandable list).")
    return "\n".join(lines)


# -----------------------------
# UI
# -----------------------------
st.markdown('<div class="h1-center">🧩 AI Job Matcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-center">Compare a resume with a job description to get a <b>semantic match</b>, <b>skills coverage</b>, and <b>ATS keyword suggestions</b>.</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="hr" />', unsafe_allow_html=True)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Resume")
    st.text_area(
        "Paste your resume text",
        key="resume_text",
        height=220,
        placeholder="Paste your resume text here...",
        label_visibility="collapsed",
    )

with right:
    st.subheader("Job Description")
    st.text_area(
        "Paste the job description text",
        key="jd_text",
        height=220,
        placeholder="Paste the job description here...",
        label_visibility="collapsed",
    )

st.markdown('<div class="spacer-16"></div>', unsafe_allow_html=True)

# Centered buttons: Match + Clear
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
with c2:
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    run_clicked = st.button("Match 🚀", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.button("Clear", use_container_width=True, on_click=clear_inputs)

st.markdown('<hr class="hr" />', unsafe_allow_html=True)


# Run matching
if run_clicked:
    resume_text = st.session_state["resume_text"]
    jd_text = st.session_state["jd_text"]

    sem = semantic_match_score(resume_text, jd_text)
    res_sk = detect_skills(resume_text)
    jd_sk = detect_skills(jd_text)

    matched = sorted(set(res_sk).intersection(set(jd_sk)))
    missing = sorted(set(jd_sk) - set(res_sk))
    coverage = skills_coverage_score(res_sk, jd_sk)

    st.session_state["last_results"] = {
        "semantic": sem,
        "coverage": coverage,
        "matched": matched,
        "missing": missing,
        "resume_skills": res_sk,
        "jd_skills": jd_sk,
    }


# Display results
results = st.session_state.get("last_results")
if results:
    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

    sem = float(results["semantic"])
    cov = float(results["coverage"])
    matched = list(results["matched"])
    missing = list(results["missing"])
    res_sk = list(results["resume_skills"])
    jd_sk = list(results["jd_skills"])

    m1, m2, m3 = st.columns([1, 1, 2], gap="large")

    with m1:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Semantic Match</div>
              <p class="card-value">{sem:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, max(0.0, sem / 100.0)))

    with m2:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Skills Coverage</div>
              <p class="card-value">{cov:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, max(0.0, cov / 100.0)))

    with m3:
        st.markdown('<div class="card"><div class="card-title">📝 Suggested keywords to add (ATS-friendly)</div>', unsafe_allow_html=True)
        if missing:
            chips_html = '<div class="chip-wrap">' + "".join(
                f'<span class="chip-yellow">{s}</span>' for s in missing
            ) + "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-muted">No missing skills detected from the current keyword list.</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted" style="margin-top:8px;">Tip: add only what you can genuinely justify in an interview.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="spacer-16"></div>', unsafe_allow_html=True)

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.subheader("✅ Skills matched")
        if matched:
            chips_html = '<div class="chip-wrap">' + "".join(
                f'<span class="chip-green">{s}</span>' for s in matched
            ) + "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.info("No matched skills detected (from the current keyword list).")

        st.markdown('<div class="spacer-16"></div>', unsafe_allow_html=True)

    with s2:
        st.subheader("⚠️ Missing skills")
        if missing:
            chips_html = '<div class="chip-wrap">' + "".join(
                f'<span class="chip-yellow">{s}</span>' for s in missing
            ) + "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.success("No missing skills detected 🎉")

        st.markdown('<div class="spacer-16"></div>', unsafe_allow_html=True)

    with st.expander("Show detected skills (debug / transparency)"):
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("**Detected in JD:**")
            st.write(sorted(set(jd_sk)) if jd_sk else "(none)")
        with b:
            st.markdown("**Detected in Resume:**")
            st.write(sorted(set(res_sk)) if res_sk else "(none)")

    st.markdown('<div class="spacer-16"></div>', unsafe_allow_html=True)

    # --- Centered PDF download only ---
    report_text = build_report(
        semantic_pct=sem,
        coverage_pct=cov,
        matched=matched,
        missing=missing,
        resume_skills=res_sk,
        jd_skills=jd_sk,
    )
    pdf_bytes = text_report_to_simple_pdf_bytes(report_text)

    # Create 5 columns and place download in the center column
    d1, d2, d3, d4, d5 = st.columns([1, 1, 1.3, 1, 1])
    with d3:
        st.download_button(
            "⬇️ Download PDF report",
            data=pdf_bytes,
            file_name="ai_job_matcher_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown('<div class="spacer-8"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-muted">Match score uses SentenceTransformers embeddings. Skills are detected via a curated keyword list (expandable).</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="small-muted">Paste your resume + job description, then click <b>Match 🚀</b>.</div>',
        unsafe_allow_html=True,
    )
