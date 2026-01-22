# 🧩 AI Job Matcher (Resume vs Job Description)

A Streamlit app that compares a resume with a job description to compute:
- **Semantic Match Score** (SentenceTransformers embeddings)
- **Skills Coverage** + **Missing Skills**
- **ATS-friendly keyword suggestions**
- **Downloadable PDF report**

## Demo (Local)
```bash
cd ai-job-matcher
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
