# Automated Resume Screening System Using Natural Language Processing and Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18+-blue?logo=react)](https://reactjs.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green?logo=node.js)](https://nodejs.org/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Testing & Results](#testing--results)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)
- [References](#references)

---

## Overview

Recruitment workflows increasingly rely on Applicant Tracking Systems (ATS) to efficiently filter resumes. However, traditional ATS solutions often rely on rigid keyword matching, causing unfair rejections and missing qualified candidates due to minor terminology differences. 

**This project introduces a Hybrid Resume Screening System** that leverages both deterministic scoring and semantic similarity (using SBERT embeddings) to deliver transparent, explainable, and reproducible candidate evaluations. The system parses resumes and job descriptions in various formats (PDF, DOCX, TXT), normalizes skills, computes a multi-factor ATS score, and delivers actionable recommendations through an interactive ReactJS dashboard.

---

## Features

- **Hybrid Scoring:** Combines deterministic keyword matching with semantic similarity for fair and robust evaluation.
- **Multi-format Parsing:** Supports resumes and JDs in PDF, DOCX, and TXT formats.
- **Skill Normalization:** Synonym mapping and fuzzy matching prevent unfair rejections due to minor terminology differences.
- **Component-wise Scoring:** Evaluates candidates based on five weighted components: keyword match, semantic similarity, section coverage, experience alignment, and certifications.
- **Actionable Recommendations:** Suggests improvements to maximize resume–JD alignment (e.g., missing skills, keyword optimization).
- **Transparent & Reproducible:** JSON-based outputs, clear scoring breakdown, and minimal score variation (<2%) between repeated runs.
- **Fallback Robustness:** Works both with and without Azure OpenAI APIs, using regex-based parsing as a fallback.
- **User Dashboard:** ReactJS-based UI for uploading, analyzing, and tracking resumes and job descriptions.

---

## System Architecture

The system is built using a **three-tier architecture** for modularity and scalability:

1. **Frontend (ReactJS):**
   - Handles user authentication and file uploads.
   - Displays detailed scoring results, history, and actionable feedback.
2. **Orchestration Layer (Node.js):**
   - Acts as a mediator, handling JSON I/O and executing Python scripts asynchronously.
3. **Backend (Python):**
   - Performs NLP-heavy tasks: parsing, preprocessing, embeddings, scoring, and recommendations.

**Workflow:**
- User uploads resume and JD → Node.js triggers Python backend → Backend parses, scores, and generates recommendations → Node.js returns structured JSON → Frontend displays results.

---

## Tech Stack

| Layer       | Technologies                                      |
|-------------|---------------------------------------------------|
| Backend     | Python, Sentence-BERT (SBERT), Regex, Azure OpenAI|
| Middleware  | Node.js, child_process.spawn, JSON I/O            |
| Frontend    | ReactJS, TailwindCSS, Material-UI, npm            |
| Data        | JSON (structured outputs), MongoDB (persistence)  |
| Dev/Ops     | Git, GitHub, Postman                              |

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ & npm 9+
- [SBERT](https://www.sbert.net/)
- MongoDB (for persistence)
- (Optional) Azure OpenAI API credentials

### 1. Clone the repository

```sh
git clone https://github.com/Raheelkhan-05/ResumeScreener.git
cd ResumeScreener
```

### 2. Backend Setup (Python)

```sh
cd server
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install SBERT and other dependencies if not included
pip install sentence-transformers scikit-learn pandas pdfplumber python-docx
```

### 3. Frontend Setup (React)

```sh
cd ../client
npm install
```

### 4. Orchestration Layer (Node.js)

```sh
cd ../server
npm install
```

### 5. MongoDB Setup

- Ensure MongoDB is running locally or provide a connection URI in the environment config.

### 6. Configure Environment

- Update `.env` files as needed for API keys and DB URIs in backend and orchestrator directories.

---

## Usage

### 1. Start the Backend

```sh
cd server
node index.js
```

### 2. Start the Frontend

```sh
cd ../client
npm start
```

### 3. Access Dashboard

- Open [http://localhost:3000](http://localhost:3000) in your browser.
- Register or log in.
- Upload your resume and job description.
- View results: ATS score, section breakdown, recommendations, and history.

---

## Testing & Results

- Tested on 20+ real-world resume-JD pairs across domains: Software Development, Data Science, IT Admin, Marketing, etc.
- Average improvement in ATS score after applying recommendations: **~14%**.
- Deterministic scoring ensures <2% variation on repeated runs.
- Robust to varied resume formats and terminologies.
- See the [project report](./PROJECT_REPORT.pdf) for detailed test cases, methodologies, and results.

<details>
<summary>Sample Test Case Results</summary>

| TC | Resume Type | Initial ATS | Final ATS | Improvement |
|----|-------------|-------------|-----------|-------------|
| 1  | Software Dev| 82          | 92        | +12%        |
| 2  | Data Science| 75          | 88        | +17%        |
| ...| ...         | ...         | ...       | ...         |

</details>

---

## Limitations

- **Azure OpenAI Dependency:** Highest accuracy with Azure OpenAI; regex fallback is less robust for highly complex formats.
- **Embedding Cache:** Partially implemented, limiting efficiency gains in large-scale use.
- **Parsing Heuristics:** Regex-based extraction may miss nuanced sections in creative layouts.
- **Language Support:** Currently supports English only.
- **Domain Customization:** Not fully optimized for highly niche industry jargon.

---

## Future Enhancements

| Enhancement                 | Description                                                      |
|-----------------------------|------------------------------------------------------------------|
| Multi-language Support      | Add multilingual embeddings for global resume support.           |
| Cloud/SaaS Deployment       | Deploy on Azure/AWS/GCP for scalability and API access.          |
| Recruiter Portal            | Role-based access for recruiters, candidates, hiring managers.   |
| Real-Time API               | Integrate with HR portals and job boards.                        |
| Industry-specific Tuning    | Fine-tune models for domains like finance, healthcare, etc.      |
| Advanced Recommendation     | Use domain-trained LLMs for phrasing, structure, and formatting. |

---

## Contributors

- **Aditya Bhalsod** (aaditya.bhalsod115957@marwadiuniversity.ac.in)
- **Raheelkhan Lohani** (raheelkhan.lohani116039@marwadiuniversity.ac.in)
- **Khush Aghera** (khush.aghera116157@marwadiuniversity.ac.in)

Project developed under the guidance of:
- Prof. Pratikkumar Chauhan, Asst. Professor, Department of Computer Engineering
- Prof. & Head Dr. Krunal Vaghela, Professor & Head, Department of Computer Engineering

---

## References

1. Suresh & Thomas, "Automated resume screening using NLP," IJCA, 2020.
2. Mehta & Majumdar, "Hybrid AI approaches in recruitment," JAI Research in Business, 2021.
3. Reimers & Gurevych, "Sentence-BERT," EMNLP-IJCNLP, 2019.
4. Devlin et al., "BERT: Pre-training of deep bidirectional transformers," NAACL-HLT, 2019.
5. Deshmukh & Raut, "Enhanced resume screening using SBERT embeddings," The Computer Science Journal, 2024.
6. Verma & Gupta, "Advanced AI-based resume screening," IJISIE, 2025.
...

See [project report](./PROJECT_REPORT.pdf) for the complete bibliography.

---

