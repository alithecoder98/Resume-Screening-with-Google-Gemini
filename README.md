# Resume Screening Application

## 📌 Overview
This project is a **Flask-based web application** that leverages **Google Gemini AI** to analyze and rank resumes based on their similarity to a given job description. The application extracts text from resumes, identifies key skills, experience, and certifications, and ranks candidates using **FAISS for vector-based similarity matching**.

## 🚀 Features
- 📂 **Upload Multiple Resumes & Job Descriptions (PDF format)**
- 🤖 **AI-powered Resume Analysis using Google Gemini**
- 🔍 **Skill & Certification Matching**
- 📊 **Ranking & Scoring based on AI analysis**
- ⚡ **FAISS-based Fast Similarity Search**
- 📜 **Results Displayed in Web & Downloadable as CSV**

## 🏗️ Tech Stack
- **Backend:** Flask, FAISS, Google Generative AI (Gemini)
- **Frontend:** HTML, Bootstrap, Jinja2
- **Processing:** PyMuPDF (fitz), NumPy, Langchain
- **Results Visualization:** Streamlit, Pandas

## 📂 Project Structure
```
|-- app.py              # Flask application for uploading & processing resumes
|-- results.py          # Streamlit dashboard to display ranked results
|-- templates/
|   |-- upload.html     # Upload page for resumes and job description
|   |-- results.html    # Display results of resume screening
|-- .env                # Environment file containing API keys
|-- requirements.txt    # Python dependencies
```

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/resume-screening.git
cd resume-screening
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file and add your **Google API Key**:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 5️⃣ Run the Application
```bash
python app.py
```

The application will be available at **http://127.0.0.1:5000/**.

## 📊 Running Streamlit Dashboard
To view results interactively:
```bash
streamlit run results.py
```

## 📌 How It Works
1. **Upload resumes** (multiple PDFs) and **job description** (PDF).
2. AI extracts **text, skills, experience, and certifications** from resumes.
3. FAISS-based similarity search finds the best matches.
4. Results are displayed in **web UI & Streamlit dashboard** with ranking.
5. Download ranked results as a **CSV file**.

## 🏆 Contributions
Feel free to contribute by submitting PRs or reporting issues!


