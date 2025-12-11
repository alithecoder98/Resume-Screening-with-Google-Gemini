# app.py
# Premium Resume Screening App — Complete Updated Code

from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import re
import tempfile
import json
import traceback
import fitz                    # PyMuPDF
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai
from difflib import SequenceMatcher
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# -----------------------
# Config
# -----------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")

genai.configure(api_key=API_KEY)

# Embedding model candidates (try in order)
EMBED_MODELS = ["models/text-embedding-004", "gemini-embedding-001"]
# Generation model
GEN_MODEL = "gemini-1.5-flash"

app = Flask(__name__)
app.config["REQUIRED_SKILLS"] = []
app.config["RESULTS"] = []

# -----------------------
# Helpers: PDF extraction
# -----------------------
def extract_text_from_pdf(file_storage):
    """Extract text from uploaded PDF file (PyMuPDF)."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_storage.read())
            tmp_path = tmp.name

        text_parts = []
        with fitz.open(tmp_path) as doc:
            for page in doc:
                text_parts.append(page.get_text("text") or "")

        os.remove(tmp_path)
        text = "\n".join(text_parts).strip()
        # debug:
        print(f"[debug] Extracted {len(text)} chars from {getattr(file_storage,'filename', 'upload')}")
        return text
    except Exception:
        traceback.print_exc()
        return ""

# -----------------------
# Regex extraction helpers
# -----------------------
_email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_phone_re = re.compile(r"(\+?\d[\d\s\-\(\)]{6,}\d)")
_name_re = re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})")

def regex_extract_email(text):
    m = _email_re.search(text or "")
    return m.group(0) if m else ""

def regex_extract_phone(text):
    m = _phone_re.search(text or "")
    return m.group(0) if m else ""

def regex_extract_name_header(text):
    """Fallback simple header name extraction - scans top lines."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:10]:
        # skip lines that look like emails/phones
        if _email_re.search(ln) or _phone_re.search(ln):
            continue
        # prefer lines with capitalized words
        m = _name_re.search(ln)
        if m:
            return m.group(0)
    # fallback: try email local-part
    email = regex_extract_email(text)
    if email:
        local = email.split("@")[0]
        # from 'michael.ramirez' -> Michael Ramirez
        parts = re.split(r"[._\-\d]+", local)
        parts = [p for p in parts if p]
        if len(parts) >= 1:
            return " ".join([p.capitalize() for p in parts[:3]])
    return ""

def regex_extract_skills(text, top_n=60):
    """Heuristic skill token extraction from resume/JD text."""
    if not text:
        return []
    tokens = re.split(r"[\n•\-\u2022,;:()]+", text)
    candidates = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if len(t.split()) > 7:
            continue
        cleaned = re.sub(r"[^A-Za-z0-9\+\#\.\-/ ]", "", t).strip()
        if 2 <= len(cleaned) <= 70:
            candidates.append(cleaned)
    # dedupe preserve order
    out = []
    seen = set()
    for c in candidates:
        low = c.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(c)
    return out[:top_n]

# -----------------------
# Text cleaning for JD
# -----------------------
def _clean_jd_text(raw):
    if not raw:
        return ""
    s = raw.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"Page\s*\d+|\d{4}|Approved|Revised|TITLE|CLASSIFICATION|SALARY|RANGE|SCOPE", " ", s, flags=re.I)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    fixed = []
    for w in s.split():
        if len(w) > 20:
            parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", w)
            fixed.extend(parts)
        else:
            fixed.append(w)
    s = " ".join(fixed)
    s = re.sub(r"[^A-Za-z0-9 ,\-/]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

# -----------------------
# Gemini wrapper helpers
# -----------------------
def call_gemini(prompt):
    """Call Gemini generation model and return raw text (best-effort)."""
    try:
        model = genai.GenerativeModel(GEN_MODEL)
        # some environments support .generate_content(prompt) or .generate_content(contents=[...])
        try:
            resp = model.generate_content(prompt)
        except TypeError:
            resp = model.generate_content(contents=[prompt])
        raw = getattr(resp, "text", None)
        if raw is None:
            raw = str(resp)
        return raw
    except Exception:
        traceback.print_exc()
        return ""

def extract_json_block(raw):
    """Extract the first {...} JSON block from LLM messy output."""
    if not raw:
        return None
    s = raw.find("{")
    if s == -1:
        return None
    depth = 0
    for i in range(s, len(raw)):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[s:i+1]
    return None

def extract_json_array_block(raw):
    """Extract first [...] JSON array block."""
    if not raw:
        return None
    s = raw.find("[")
    if s == -1:
        return None
    depth = 0
    for i in range(s, len(raw)):
        ch = raw[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return raw[s:i+1]
    return None

# -----------------------
# Advanced name extraction (ultra-strong)
# -----------------------
CITY_WORDS = {
    "los angeles","san diego","san jose","oakland","sacramento","san francisco","long beach",
    "fresno","riverside","stockton","new york","chicago","houston","phoenix","dallas","austin",
    "miami","orlando","seattle","boston","denver","atlanta"
}
JOB_TITLE_WORDS = {
    "manager","coordinator","analyst","specialist","consultant","assistant","associate",
    "director","hr","recruitment","analytics","administration","support","clerk","engineer",
    "supervisor","superintendent","officer"
}

def titlecase_name(s):
    s = s.strip()
    if not s:
        return ""
    parts = []
    for part in s.split():
        if len(part) <= 2:
            parts.append(part.upper())
        else:
            parts.append(part.capitalize())
    return " ".join(parts)

def looks_like_person_name(s):
    if not s:
        return False
    s = s.strip()
    low = s.lower()
    # reject city equal lines
    if low in CITY_WORDS:
        return False
    # reject job title lines
    if any(w in low for w in JOB_TITLE_WORDS):
        return False
    # reject lines with digits, emails or many punctuation
    if re.search(r"[\d@/\\]", s):
        return False
    # overly long lines are not names
    if len(s.split()) > 4:
        return False
    # if all caps: accept only two-word lines (JANE DOE)
    if s.isupper():
        if len(s.split()) != 2:
            return False
    # basic name pattern: Titlecase tokens
    if re.match(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,3}$", s):
        return True
    # also allow single capitalized + last allcaps (rare)
    if re.match(r"^[A-Z][a-z]+(\s+[A-Z]{2,})?$", s):
        return True
    return False

def extract_name_improved(gemini_name, resume_text, filename_fallback="Candidate"):
    # 1) use gemini name if valid
    if gemini_name and looks_like_person_name(gemini_name):
        return titlecase_name(gemini_name)
    # 2) look for name above email/phone
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    email = regex_extract_email(resume_text)
    phone = regex_extract_phone(resume_text)
    for i, ln in enumerate(lines):
        if email and email in ln or phone and phone in ln:
            for j in range(max(0, i-3), i):
                candidate = lines[j].strip()
                if looks_like_person_name(candidate):
                    return titlecase_name(candidate)
            # try pick uppercase words in previous line
            if i-1 >= 0:
                prev = lines[i-1].strip()
                words = prev.split()
                up = [w for w in words if w.isupper() and len(w) > 1]
                if len(up) >= 2:
                    return titlecase_name(" ".join(up[:2]))
    # 3) scan top lines
    for ln in lines[:12]:
        if looks_like_person_name(ln):
            return titlecase_name(ln)
    # 4) email local-part fallback
    if email:
        local = email.split("@")[0]
        parts = re.split(r"[._\-\d]+", local)
        parts = [p for p in parts if p]
        if parts:
            return " ".join([p.capitalize() for p in parts[:3]])
    # 5) filename fallback
    fallback = os.path.splitext(filename_fallback)[0]
    fallback = fallback.replace("_", " ").replace("-", " ")
    return titlecase_name(fallback)

# -----------------------
# Candidate JSON extraction using Gemini
# -----------------------
def extract_candidate_details(resume_text):
    """
    Ask Gemini to return a JSON object, fallback to regex heuristics.
    Returns dict with keys: full_name, city, email, phone, skills, experience_years, certifications
    """
    prompt = f"""
Extract a JSON object from the resume text with the exact keys:
{{"full_name": "", "city": "", "email": "", "phone": "", "skills": [], "experience_years": 0, "certifications": []}}
Return ONLY the JSON object (no extra text).

Resume:
{resume_text}
    """
    raw = call_gemini(prompt)
    block = extract_json_block(raw)
    parsed = {}
    if block:
        try:
            parsed = json.loads(block)
        except Exception:
            parsed = {}
    # fallback extraction
    full_name = parsed.get("full_name") if isinstance(parsed.get("full_name"), str) else None
    city = parsed.get("city") or ""
    email = parsed.get("email") or regex_extract_email(resume_text)
    phone = parsed.get("phone") or regex_extract_phone(resume_text)
    skills = parsed.get("skills") or regex_extract_skills(resume_text, top_n=80)
    certs = parsed.get("certifications") or []
    exp = parsed.get("experience_years") or 0
    # normalize
    try:
        exp = float(exp)
    except Exception:
        m = re.findall(r"[\d\.]+", str(exp))
        exp = float(m[0]) if m else 0.0
    if not isinstance(skills, list):
        if isinstance(skills, str):
            skills = regex_extract_skills(skills, top_n=80)
        else:
            skills = []
    if not isinstance(certs, list):
        certs = [certs] if certs else []
    # ensure strings
    full_name = full_name or ""
    city = city or ""
    email = email or ""
    phone = phone or ""
    return {
        "full_name": full_name,
        "city": city,
        "email": email,
        "phone": phone,
        "skills": skills,
        "experience_years": round(float(exp), 2),
        "certifications": certs
    }

# -----------------------
# JD skills extraction
# -----------------------
def extract_required_skills_from_jd(jd_text):
    cleaned = _clean_jd_text(jd_text)
    prompt = f"""
You are an assistant that extracts REQUIRED technical and role-related skills from a job description.
Return ONLY a JSON array like ["skill1","skill2"] (no explanation).
Job Description:
{cleaned}
"""
    raw = call_gemini(prompt)
    arr = None
    try:
        arr = json.loads(raw)
    except Exception:
        jb = extract_json_array_block(raw)
        if jb:
            try:
                arr = json.loads(jb)
            except Exception:
                arr = None
    if isinstance(arr, list):
        final = []
        seen = set()
        for it in arr:
            s = str(it).strip()
            if not s:
                continue
            if len(s.split()) > 5:
                continue
            low = s.lower()
            if low in seen:
                continue
            seen.add(low)
            final.append(s)
        return final[:30]
    # fallback heuristic
    fallback = regex_extract_skills(cleaned, top_n=80)
    # take only short tokens <=4 words and likely HR terms
    out = []
    seen = set()
    for tok in fallback:
        if len(tok.split()) > 4:
            continue
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(tok)
        if len(out) >= 20:
            break
    return out

# -----------------------
# Embeddings wrapper
# -----------------------
def get_embeddings_for_docs(docs):
    last_err = None
    for model_name in EMBED_MODELS:
        try:
            emb = GoogleGenerativeAIEmbeddings(model=model_name)
            vecs = emb.embed_documents(docs)
            # ensure we have lists/arrays
            if vecs and isinstance(vecs[0], (list, tuple, np.ndarray)):
                return vecs
        except Exception as e:
            print(f"[embed] model {model_name} failed:", e)
            last_err = e
    raise RuntimeError("Embedding failed for all models") from last_err

# -----------------------
# Skill matching (fuzzy)
# -----------------------
def skill_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def match_skills(required_skills, candidate_skills, threshold=0.65):
    req = [r.lower().strip() for r in required_skills]
    cand = [c.lower().strip() for c in candidate_skills]
    matched = set()
    for r in req:
        for c in cand:
            if not r or not c:
                continue
            if r == c:
                matched.add(r); break
            if r in c or c in r:
                matched.add(r); break
            if skill_similarity(r, c) >= threshold:
                matched.add(r); break
    pct = round((len(matched) / max(1, len(req))) * 100, 2) if req else 0.0
    # return original-cased matched list from required_skills for display
    matched_display = [s for s in required_skills if s.lower() in matched]
    return matched_display, pct

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def upload_page():
    if request.method == "POST":
        jd_file = request.files.get("jd_file")
        resumes = request.files.getlist("resume_files")
        if not jd_file or not resumes:
            return render_template("upload.html", error="Upload job description and at least one resume.")

        # Extract JD text and required skills
        jd_text = extract_text_from_pdf(jd_file)
        req_skills = extract_required_skills_from_jd(jd_text)
        print("[debug] Required skills extracted:", req_skills)

        # Extract resume texts
        resume_texts = []
        filenames = []
        for f in resumes:
            t = extract_text_from_pdf(f)
            if t:
                resume_texts.append(t)
                filenames.append(getattr(f, "filename", "candidate.pdf"))

        if not resume_texts:
            return render_template("upload.html", error="No resume text extracted from files.")

        # Get embeddings (resumes + JD)
        try:
            resume_vecs = get_embeddings_for_docs(resume_texts)
            jd_vec = get_embeddings_for_docs([jd_text])[0]
        except Exception as e:
            print("[error] Embedding error:", e)
            traceback.print_exc()
            return render_template("upload.html", error="Embedding service failed. See server logs.")

        # Build FAISS index
        dim = len(resume_vecs[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(resume_vecs, dtype=np.float32))
        k = min(10, len(resume_vecs))
        D, I = index.search(np.array([jd_vec], dtype=np.float32), k)
        ranked_indices = [int(x) for x in I[0] if x >= 0]

        results = []
        # For each candidate, extract details and compute scores
        for pos, idx in enumerate(ranked_indices, start=1):
            text = resume_texts[idx]
            details = extract_candidate_details(text)
            # Attempt to get a strong name: use details['full_name'] (from Gemini) + improved extractor
            gemini_name = details.get("full_name") or ""
            name = extract_name_improved(gemini_name, text, filename_fallback=filenames[idx])
            # Match skills
            matched_skills, skill_match_pct = match_skills(req_skills, details.get("skills", []), threshold=0.65)
            cert_match_pct = 100.0 if details.get("certifications") else 0.0
            score = round((details.get("experience_years", 0) * 0.35) + (skill_match_pct * 0.50) + (cert_match_pct * 0.15), 2)
            results.append({
                "rank": pos,
                "filename": filenames[idx],
                "name": name,
                "city": details.get("city", ""),
                "email": details.get("email", ""),
                "phone": details.get("phone", ""),
                "skills": details.get("skills", []),
                "experience_years": details.get("experience_years", 0),
                "certifications": details.get("certifications", []),
                "skill_match_pct": round(skill_match_pct, 2),
                "cert_match_pct": round(cert_match_pct, 2),
                "score": score,
                "matched_skills": matched_skills
            })

        # sort by score desc and re-rank
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results, start=1):
            r["rank"] = i

        app.config["REQUIRED_SKILLS"] = req_skills
        app.config["RESULTS"] = results
        return redirect(url_for("results_page"))

    # GET
    return render_template("upload.html")

@app.route("/results")
def results_page():
    return render_template("results.html", required_skills=app.config.get("REQUIRED_SKILLS", []), resumes=app.config.get("RESULTS", []))

# useful debug route to download sample resume (if you add sample files in samples/)
@app.route("/download-sample/<path:filename>")
def download_sample(filename):
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    fp = os.path.join(samples_dir, filename)
    if os.path.exists(fp):
        return send_file(fp, as_attachment=True)
    return "Not found", 404

# -----------------------
# Start server
# -----------------------
if __name__ == "__main__":
    print("🚀 Starting Resume Screening App (Updated)")
    # disable reloader because libs sometimes trigger reloads; set debug=False if you prefer
    app.run(debug=False, use_reloader=False)

