````markdown
# 🤖 AI Resume Screener & Ranker

A lightweight, interactive web app built with **Streamlit** that uses **GPT-4o via Azure OpenAI** to evaluate and rank multiple resumes against a provided job description (JD). This tool leverages **LangChain** to generate semantic similarity-based scoring using LLMs.

---

## 🎯 Features

- 📄 Upload **one job description** in PDF format.
- 📑 Upload **up to 10 resumes** (PDFs).
- ⚙️ Uses **GPT-4o via Azure OpenAI** to calculate how well each resume matches the JD.
- 🔢 Generates a **numeric score (0–100)** per resume.
- 📊 Ranks resumes in descending order of compatibility.
- ⚡ Real-time scoring feedback via a minimal **Streamlit UI**.

---

## 🧠 How It Works

1. Users upload a **job description PDF** and **multiple resume PDFs**.
2. The app extracts raw text from the files using **PyPDF2**.
3. For each resume, it uses **LangChain's LLMChain** with a custom prompt to query GPT-4o.
4. The LLM is asked to score the resume's fit for the JD from 0 to 100.
5. Results are sorted and displayed in ranked order.

---

## 🛠 Tech Stack

| Component         | Technology                            |
|------------------|----------------------------------------|
| Frontend UI      | Streamlit                              |
| PDF Text Parsing | PyPDF2                                 |
| LLM Integration  | Azure OpenAI (`gpt-4o`) + LangChain    |
| Prompting Engine | LangChain LLMChain + ChatPromptTemplate |
| Config Management| Python `dotenv` + OS Environment Vars  |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/resume-screener.git
cd resume-screener
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Sample `requirements.txt`:

```txt
streamlit
PyPDF2
langchain
python-dotenv
langchain-openai
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory and add your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_BASE=https://<your-endpoint>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_API_NAME=gpt-4o-deployment-name
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 💡 Prompt Design

```txt
Given the job description below:
{jd}
And the resume below:
{resume}
Score the resume for how well it fits the job description on a scale of 0 to 100.
Only respond with a numeric score.
```

---

## 🚧 Limitations

* Only supports **PDF** files.
* Designed for **up to 10 resumes** per batch.
* **LLM inference latency** may increase with multiple large files.
* Assumes LLM can parse and understand resume structure accurately without fine-tuning.

---

## 📌 Possible Improvements

* Add **detailed reasoning/explanation** with score.
* Support **DOCX** or **plain text** resumes.
* Enable **feedback loop** to refine scoring based on user input.
* Persist results to a database for audit/review.
* Batch download of ranked reports.

---

## 👨‍💻 Author

* [Raheelkhan Lohani](https://github.com/Raheelkhan-05)

---

## 📜 License

This project is for educational and evaluation purposes only. All rights reserved.

---
