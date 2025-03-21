from flask import Flask, render_template, request, redirect, url_for
import os
import fitz  # PyMuPDF for PDF text extraction
import faiss
import tempfile
from dotenv import load_dotenv
import numpy as np
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import json

# ✅ Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY is missing! Please check your .env file.")

genai.configure(api_key=api_key)

# ✅ Initialize Flask
app = Flask(__name__)

resume_rankings = []  # Store rankings globally

# ✅ Extract Text from PDF
def extract_text_from_pdf(file):
    """Extracts text from PDFs and logs text size."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file.read())
        temp_pdf_path = temp_pdf.name

    with fitz.open(temp_pdf_path) as doc:
        print(f"📄 Processing PDF: {file.filename} | Pages: {len(doc)}")  # ✅ Log page count
        for page in doc:
            text += page.get_text("text") + "\n"

    os.remove(temp_pdf_path)  # ✅ Cleanup temp file
    print("\n✅ Extracted Resume Text (First 1000 chars):\n", text[:1000], "...\n")  # ✅ Debugging Log
    return text.strip()

# ✅ Extract Certifications from Resume
def calculate_certification_match(resume_text):
    """Checks for known certifications."""
    CERTIFICATIONS_LIST = [
        "AWS Certified", "Azure Certified", "Google Cloud Certified",
        "Kubernetes Certified", "CCNA", "PMP", "Scrum Master"
    ]
    match_count = sum(1 for cert in CERTIFICATIONS_LIST if cert.lower() in resume_text.lower())
    return round((match_count / len(CERTIFICATIONS_LIST)) * 100, 2)

# ✅ Extract Skills Match
def calculate_skill_match(resume_text, jd_text):
    """Compares job description keywords with resume."""
    jd_keywords = set(jd_text.lower().split())
    resume_keywords = set(resume_text.lower().split())
    return round((len(jd_keywords.intersection(resume_keywords)) / len(jd_keywords)) * 100, 2)

# ✅ Extract Experience, Skills, and Tools from LLM
def extract_resume_details_llm(resume_text):
    """Uses Google Gemini API to extract structured JSON resume details."""
    prompt = f"""
    Extract structured JSON data from the following resume:
    {{
      "experience_years": "Extract total years of experience.",
      "companies": ["List companies worked at."],
      "skills": ["List the technical skills mentioned."],
      "tools": ["List tools and technologies used."]
    }}

    Resume:
    {resume_text}

    IMPORTANT: Return JSON format ONLY.
    """

    try:
        response = genai.generate_content(model="gemini-pro", contents=[prompt])
        json_output = json.loads(response.text)
        return json_output.get("experience_years", 0), json_output.get("companies", []), json_output.get("skills", []), json_output.get("tools", [])

    except json.JSONDecodeError:
        print("⚠️ Gemini API returned invalid JSON!")
        return 0, [], [], []
    
    except Exception as e:
        print(f"⚠️ Error in LLM extraction: {e}")
        return 0, [], [], []

@app.route("/", methods=["GET", "POST"])
def upload_page():
    global resume_rankings
    resume_rankings = []

    if request.method == "POST":
        resumes = request.files.getlist("resume_files")
        jd_file = request.files.get("jd_file")

        if resumes and jd_file:
            try:
                jd_text = extract_text_from_pdf(jd_file)  # ✅ Extract JD text
                resume_texts = [extract_text_from_pdf(res) for res in resumes]  # ✅ Extract Resume text

                # ✅ Fix FAISS Multiple Resume Issue
                merged_resumes = ["\n".join(text) for text in resume_texts]  # ✅ Ensure 1 document per resume

                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                resume_vectors = embeddings.embed_documents(merged_resumes)  # ✅ Generate 1 vector per resume
                jd_embedding = embeddings.embed_documents([jd_text])[0]  # ✅ Generate JD embedding

                # ✅ FAISS Index Setup (Ensure 1 entry per resume)
                index = faiss.IndexFlatL2(len(resume_vectors[0]))

                print(f"\n⚡ FAISS Index Before Adding Resumes: {index.ntotal} entries")  # ✅ Debugging log
                index.add(np.array(resume_vectors, dtype=np.float32))
                print(f"✅ FAISS Index After Adding Resumes: {index.ntotal} entries\n")  # ✅ Debugging log

                # ✅ Retrieve Top Matches
                distances, indices = index.search(np.array([jd_embedding], dtype=np.float32), k=min(5, len(resume_vectors)))
                best_matches = [merged_resumes[i] for i in indices[0]]

                for i, content in enumerate(best_matches):
                    years_experience, companies, skills, tools = extract_resume_details_llm(content)
                    skill_match = calculate_skill_match(content, jd_text)
                    cert_match = calculate_certification_match(content)

                    total_score = (years_experience * 0.4) + (skill_match * 0.4) + (cert_match * 0.2)

                    resume_rankings.append({
                        "rank": i + 1,
                        "content": content[:500] + "...",
                        "years_experience": years_experience,
                        "companies": companies,
                        "skills": skills,
                        "tools": tools,
                        "skill_match": skill_match,
                        "cert_match": cert_match,
                        "total_similarity": round(total_score, 2)
                    })

                # ✅ Sort results in descending order by similarity score
                resume_rankings.sort(key=lambda x: x["total_similarity"], reverse=True)

            except Exception as e:
                print(f"⚠️ Error processing files: {e}")

            return redirect(url_for("results_page"))

    return render_template("upload.html")

@app.route("/results")
def results_page():
    return render_template("results.html", resumes=resume_rankings)

if __name__ == "__main__":
    app.run(debug=True)
