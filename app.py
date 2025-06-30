import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_API_NAME"),
    model_name="gpt-4o",
    temperature=0.9,
    top_p=0.9,
    max_tokens=100
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """Given the job description below:
    {jd}
    And the resume below:
    {resume}
    Score the resume for how well it fits the job description on a scale of 0 to 100. Only respond with a numeric score.
"""
)

st.title("Resume Ranker")

jd_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")
resume_files = st.file_uploader("Upload Resumes (max 10 PDFs)", type="pdf", accept_multiple_files=True)


def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


if jd_file and resume_files:
    if len(resume_files) > 10:
        st.warning("Please upload up to 10 resumes only.")
    else:
        jd_text = extract_pdf_text(jd_file)
        resume_texts = []
        for res in resume_files:
            reader = PdfReader(res)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            resume_texts.append((res.name, text))

        st.info("Scoring resumes, please wait...")

        results = []
        for name, text in resume_texts:            
            llmchain = LLMChain(
                llm=llm,
                prompt=prompt
            )
            score_raw = llmchain.invoke({"jd": jd_text, "resume": text})
            score = int(score_raw['text'])
            results.append((name, score))

        st.subheader("Ranked Resumes")
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        for i, (name, score) in enumerate(sorted_results, start=1):
            st.write(f"{i}. {name} — Score: {score}/100")
