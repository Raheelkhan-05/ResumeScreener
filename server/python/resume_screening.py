import os
import sys
import json
import argparse
import re
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import docx
import difflib
import openai

from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Determinism & seeds
# ---------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------
# Azure OpenAI config (env)
# ---------------------------
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_API_NAME")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")

if not (AZURE_API_KEY and AZURE_API_BASE and AZURE_DEPLOYMENT):
    print("ERROR: Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, AZURE_OPENAI_API_NAME in environment.", file=sys.stderr)
    # we don't exit here; user might call only embedding part. But warn.
    
# Configure openai (azure mode)
if AZURE_API_KEY and AZURE_API_BASE and AZURE_DEPLOYMENT:
    openai.api_type = "azure"
    openai.api_key = AZURE_API_KEY
    openai.api_base = AZURE_API_BASE
    openai.api_version = AZURE_API_VERSION

# ---------------------------
# Settings (can be tuned)
# ---------------------------
SBERT_MODEL = "all-mpnet-base-v2"   # user-selected
EMBEDDING_CACHE_DIR = Path(".embed_cache")
EMBEDDING_CACHE_DIR.mkdir(exist_ok=True)
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 2000

# ---------------------------
# Helper functions: text extraction
# ---------------------------

def read_txt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def extract_text_from_pdf(path: Path) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
    return "\n".join(text_parts)

def extract_text_from_docx(path: Path) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def extract_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(p)
    if suffix in {".docx", ".doc"}:
        return extract_text_from_docx(p)
    if suffix in {".txt"}:
        return read_txt_file(p)
    # fallback: try reading as text
    return read_txt_file(p)

# ---------------------------
# Simple normalizers, canonicalization
# ---------------------------

CANONICAL_SKILL_MAP = {
    # common normalizations
    "js": "javascript",
    "nodejs": "node.js",
    "node": "node.js",
    "py": "python",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "aws": "aws",
    "gcp": "gcp",
    "sql": "sql",
    "oop": "object oriented programming",
    "oops": "object oriented programming",
    "communication": "communication skills",
    "cognitive": "cognitive skills",
    "problem solving": "problem solving",
    "sdlc": "software development life cycle",
    "api": "rest api",
    "rest": "rest api",
    "git": "version control",
    "github": "version control",
    "mysql": "database",
    "mongodb": "database",
    "postgresql": "database",
    "firebase": "database",
    "react": "react.js",
    "reactjs": "react.js",
    "express": "express.js",
    "expressjs": "express.js",
    "tensorflow": "machine learning",
    "pytorch": "machine learning",
    "scikit-learn": "machine learning",
    "sklearn": "machine learning",
    "opencv": "computer vision",
    "cv": "computer vision",
    "nlp": "natural language processing",
    "transformers": "natural language processing",
    "bert": "natural language processing",
    "distilbert": "natural language processing",
}

STOPWORDS = set(["a","an","the","in","of","on","for","and","or","with","to","by","at","from","as"])

def normalize_token(t: str) -> str:
    s = t.lower().strip()
    s = re.sub(r"[^\w\.\+\-#]", " ", s)  # keep + - # . for names like C++ / C# / node.js
    s = " ".join(s.split())
    if s in CANONICAL_SKILL_MAP:
        s = CANONICAL_SKILL_MAP[s]
    # strip stopwords tokens if wholly stopword
    if s in STOPWORDS:
        return ""
    return s

def canonicalize_skill(s: str) -> str:
    return normalize_token(s)

# ---------------------------
# Embeddings (SBERT) with caching
# ---------------------------

model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer(SBERT_MODEL)
        # ensure deterministic pooling behavior; sentence-transformers uses deterministic pooled outputs
    return model

def text_hash_key(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h

def cache_embedding(text: str) -> np.ndarray:
    key = text_hash_key(text)
    cache_file = EMBEDDING_CACHE_DIR / f"{key}.npy"
    if cache_file.exists():
        arr = np.load(cache_file)
        return arr
    m = get_model()
    emb = m.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
    # np.save(cache_file, emb)
    return emb

# ---------------------------
# Azure OpenAI LLM helpers
# ---------------------------

SYSTEM_PROMPT_BASE = (
    "You are an exact, strict JSON-returning parser and evaluator used in an automated ATS pipeline. "
    "Always respond with valid JSON only (no commentary). If a field cannot be found, use empty string, empty list or null. "
    "Do not hallucinate; rely only on the provided text. Normalize dates to YYYY-MM when possible. "
    "Lowercase canonical skill names in skills_canonical. Keep responses deterministic."
)

def call_azure_chat(messages: List[Dict[str,str]], max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE) -> str:
    """
    Calls Azure OpenAI Chat Completions. Expects environment variables configured.
    Returns assistant content (string).
    """
    if not (AZURE_API_KEY and AZURE_API_BASE and AZURE_DEPLOYMENT):
        raise RuntimeError("Azure OpenAI environment variables not set (AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, AZURE_OPENAI_API_NAME).")
    # Using openai.ChatCompletion.create in azure mode:
    resp = openai.ChatCompletion.create(
        engine=AZURE_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1
    )
    content = resp.choices[0].message["content"]
    return content

# ---------------------------
# Prompt templates (shortened, strict)
# ---------------------------

PARSE_RESUME_PROMPT = (
    SYSTEM_PROMPT_BASE
    + "\n\n"
    + "TASK: Parse the following resume text into JSON according to the schema 'parsed_resume' below. "
    "Extract ALL technical skills mentioned anywhere in the resume. Include programming languages, frameworks, tools, technologies, methodologies, and soft skills. "
    "Produce only JSON that matches the schema.\n\n"
    "SCHEMA parsed_resume:\n"
    "{"
    "\"name\":\"\", \"email\":\"\", \"phone\":\"\", \"location\":\"\", \"summary\":\"\", "
    "\"sections\": [{\"label\":\"experience\",\"text\":\"\"}, {\"label\":\"skills\",\"text\":\"\"}, {\"label\":\"education\",\"text\":\"\"}, {\"label\":\"projects\",\"text\":\"\"}], "
    "\"experience_years\": 0.0, "
    "\"education\": [{\"degree\":\"\",\"institution\":\"\",\"start\":\"YYYY-MM\",\"end\":\"YYYY-MM\",\"grade\":\"\"}], "
    "\"skills_canonical\": [], \"skills_tokens\": [], "
    "\"projects\": [], \"certifications\": [] "
    "}\n\n"
    "For skills extraction, look for:\n"
    "- Programming languages (Python, Java, C++, JavaScript, etc.)\n"
    "- Frameworks and libraries (React, Flask, TensorFlow, etc.)\n"
    "- Tools and technologies (Git, Docker, AWS, etc.)\n"
    "- Methodologies (SDLC, Agile, etc.)\n"
    "- Soft skills (communication, problem-solving, etc.)\n"
    "- Any technical competencies mentioned\n\n"
    "INPUT_RESUME:\n"
)

PARSE_JD_PROMPT = (
    SYSTEM_PROMPT_BASE
    + "\n\n"
    + "TASK: Parse the following Job Description into JSON according to 'parsed_jd' schema. "
    "Extract ALL required skills from skills section, job roles, and responsibilities. "
    "Produce only JSON.\n\n"
    "SCHEMA parsed_jd:\n"
    "{"
    "\"title\":\"\", \"company\":\"\", \"location\":\"\", \"seniority\":\"\", \"min_experience_years\":0, "
    "\"required_skills\":[{\"skill\":\"\",\"importance\":\"required|preferred|nice-to-have\"}], "
    "\"responsibilities\":[], \"hard_constraints\": {\"degree\":\"\",\"visa_sponsorship\":\"yes|no|unknown\"}, "
    "\"priority_skills\":[]"
    "}\n\n"
    "For skills extraction, include:\n"
    "- Technical skills mentioned in skills section\n"
    "- Skills implied by job roles (e.g., Python Developer implies Python skills)\n"
    "- Skills mentioned in responsibilities\n"
    "- Soft skills mentioned\n"
    "- Educational requirements as skills\n\n"
    "INPUT_JD:\n"
)

SCORING_PROMPT_PREFIX = (
    SYSTEM_PROMPT_BASE + "\n\n"
    "TASK: You will NOT compute the score as LLM. The scoring will be computed deterministically by Python using the recipe below. "
    "This prompt exists if you later want LLM-based calibration. For now Python scoring will use the exact recipe embedded in code.\n"
)

SUMMARY_RECOMMEND_PROMPT = (
    SYSTEM_PROMPT_BASE
    + "\n\n"
    + "TASK: Given parsed_resume (JSON), parsed_jd (JSON) and scoring (JSON), produce ONLY JSON of the form: "
    "{\"summary\":\"<two-line summary>\", \"recommendations\": [\"...\", \"...\"] }. "
    "Recommendations should be specific, actionable suggestions that will improve ATS score. "
    "Focus on missing skills, keywords, and specific improvements. "
)

# ---------------------------
# LLM parsing wrappers
# ---------------------------

def run_parse_resume_llm(raw_text: str) -> Dict[str,Any]:
    msg = [
        {"role":"system", "content": PARSE_RESUME_PROMPT},
        {"role":"user", "content": raw_text}
    ]
    content = call_azure_chat(msg)
    # Expect JSON - try load
    try:
        return json.loads(content)
    except Exception as e:
        # attempt to extract JSON substring
        m = re.search(r"(\{[\s\S]*\})", content)
        if m:
            return json.loads(m.group(1))
        raise

def run_parse_jd_llm(raw_text: str) -> Dict[str,Any]:
    msg = [
        {"role":"system", "content": PARSE_JD_PROMPT},
        {"role":"user", "content": raw_text}
    ]
    content = call_azure_chat(msg)
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"(\{[\s\S]*\})", content)
        if m:
            return json.loads(m.group(1))
        raise

def run_summary_recommend_llm(parsed_resume: dict, parsed_jd: dict, scoring: dict) -> Dict[str,Any]:
    payload = {
        "parsed_resume": parsed_resume,
        "parsed_jd": parsed_jd,
        "scoring": scoring
    }
    msg = [
        {"role":"system", "content": SUMMARY_RECOMMEND_PROMPT},
        {"role":"user", "content": json.dumps(payload)}
    ]
    content = call_azure_chat(msg)
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"(\{[\s\S]*\})", content)
        if m:
            return json.loads(m.group(1))
        raise

def is_safe_skill_replacement(resume_skill: str, jd_skill: str) -> bool:
    """
    Determines if suggesting to replace resume_skill with jd_skill is safe and makes sense.
    Prevents dangerous suggestions like replacing Java with JavaScript.
    """
    
    # Convert to lowercase for comparison
    resume_lower = resume_skill.lower().strip()
    jd_lower = jd_skill.lower().strip()
    
    # Dangerous replacements to avoid
    dangerous_pairs = {
        # Programming Languages - completely different
        "java": ["javascript", "js"],
        "javascript": ["java"],
        "python": ["php", "perl"],
        "php": ["python"],
        "c++": ["c#", "objective-c"],
        "c#": ["c++", "c"],
        "go": ["golang"],  # Actually same, but avoid confusion
        
        # Frameworks/Libraries - different ecosystems
        "react": ["react native", "angular", "vue"],
        "react native": ["react", "flutter"],
        "angular": ["react", "vue"],
        "django": ["flask", "fastapi"],
        "flask": ["django"],
        "tensorflow": ["pytorch", "keras"],
        "pytorch": ["tensorflow"],
        
        # Databases - different types
        "mysql": ["mongodb", "postgresql", "redis"],
        "mongodb": ["mysql", "postgresql"],
        "postgresql": ["mysql", "mongodb"],
        "redis": ["mysql", "mongodb"],
        
        # Infrastructure/Tools - different purposes
        "docker": ["kubernetes", "jenkins"],
        "kubernetes": ["docker"],
        "jenkins": ["docker", "kubernetes"],
        "aws": ["azure", "gcp"],
        "azure": ["aws", "gcp"],
        "gcp": ["aws", "azure"],
        
        # Methodologies - different approaches
        "agile": ["waterfall", "scrum"],
        "waterfall": ["agile"],
        "scrum": ["kanban"],
        
        # Domain-specific terms that shouldn't be confused
        "machine learning": ["deep learning", "data mining"],
        "deep learning": ["machine learning"],
        "data engineering": ["data science", "feature engineering"],
        "data science": ["data engineering"],
        "feature engineering": ["data engineering"],
        
        # Technical concepts - different meanings
        "api": ["rest api", "graphql"],
        "rest api": ["graphql", "soap"],
        "graphql": ["rest api"],
        
        # Specific technology names that are often confused
        "u2net": [".net", "dotnet"],  # U2Net is a neural network, .NET is Microsoft framework
        "bert": ["robotics", "automation"],  # BERT is NLP model
        "opencv": ["computer vision"],  # OpenCV is library, computer vision is field
        "nlp": ["natural language processing"],  # Actually same, but NLP is abbreviation
        
        # Testing - different types
        "unit testing": ["integration testing", "e2e testing"],
        "integration testing": ["unit testing"],
        
        # Mobile development - different platforms
        "android": ["ios", "flutter"],
        "ios": ["android", "flutter"],
        "flutter": ["react native", "android", "ios"],
    }
    
    # Check if this would be a dangerous replacement
    if resume_lower in dangerous_pairs:
        if any(jd_lower in dangerous or dangerous in jd_lower for dangerous in dangerous_pairs[resume_lower]):
            return False
    
    # Additional semantic checks
    
    # Don't replace specific library/framework names with general concepts
    specific_tools = ["tensorflow", "pytorch", "opencv", "pandas", "numpy", "scikit-learn", "keras", "u2net", "bert", "distilbert"]
    general_concepts = ["machine learning", "deep learning", "computer vision", "data analysis", "artificial intelligence", "natural language processing"]
    
    if resume_lower in specific_tools and any(concept in jd_lower for concept in general_concepts):
        return False
    
    # Don't replace programming languages with similar-sounding but different ones
    programming_languages = ["python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "rust", "swift", "kotlin"]
    if resume_lower in programming_languages and jd_lower in programming_languages and resume_lower != jd_lower:
        # Only allow very close matches (like python -> python3)
        if difflib.SequenceMatcher(None, resume_lower, jd_lower).ratio() < 0.8:
            return False
    
    # Don't replace if they're completely different domains
    ml_terms = ["machine learning", "deep learning", "neural networks", "tensorflow", "pytorch", "scikit-learn", "ai", "ml", "nlp", "computer vision", "opencv", "bert", "transformers"]
    web_terms = ["html", "css", "javascript", "react", "angular", "vue", "node.js", "express", "django", "flask", "php", "laravel"]
    mobile_terms = ["android", "ios", "flutter", "react native", "swift", "kotlin", "dart"]
    devops_terms = ["docker", "kubernetes", "jenkins", "ci/cd", "aws", "azure", "gcp", "terraform", "ansible"]
    
    domains = [ml_terms, web_terms, mobile_terms, devops_terms]
    
    resume_domains = [i for i, domain in enumerate(domains) if any(term in resume_lower for term in domain)]
    jd_domains = [i for i, domain in enumerate(domains) if any(term in jd_lower for term in domain)]
    
    # If they're in completely different domains, don't suggest replacement
    if resume_domains and jd_domains and not set(resume_domains).intersection(set(jd_domains)):
        return False
    
    return True

def find_relevant_skill_additions(resume_skills: set, jd_skills: list, parsed_resume: dict) -> List[str]:
    """
    Find JD skills that are genuinely missing and would be relevant to add.
    """
    relevant_additions = []
    
    # Get resume text for context
    resume_text = json.dumps(parsed_resume, ensure_ascii=False).lower()
    
    for jd_skill in jd_skills:
        jd_skill_canonical = canonicalize_skill(jd_skill).lower()
        
        # Skip if already present
        if jd_skill_canonical in resume_skills:
            continue
            
        # Check if it's a reasonable addition based on resume content
        is_relevant = False
        
        # For soft skills, always suggest if missing
        soft_skills = ["communication", "cognitive", "analytical", "problem solving", "teamwork", "leadership"]
        if any(soft in jd_skill.lower() for soft in soft_skills):
            is_relevant = True
        
        # For technical skills, check if candidate has related experience
        elif "programming" in jd_skill.lower() or "oop" in jd_skill.lower():
            # If they have any programming languages, they likely know OOP
            programming_langs = ["python", "java", "javascript", "c++", "c#"]
            if any(lang in resume_text for lang in programming_langs):
                is_relevant = True
                
        elif "sdlc" in jd_skill.lower() or "software development" in jd_skill.lower():
            # If they have development experience, SDLC is relevant
            dev_indicators = ["developer", "development", "programming", "coding", "software"]
            if any(indicator in resume_text for indicator in dev_indicators):
                is_relevant = True
                
        elif "testing" in jd_skill.lower() and "qa" not in jd_skill.lower():
            # General testing skills are relevant if they have development experience
            if any(term in resume_text for term in ["development", "programming", "project"]):
                is_relevant = True
                
        # For role-specific skills, only suggest if they're applying for that role
        job_title = parsed_resume.get("sections", [])
        job_roles = []
        for section in job_title:
            if section.get("label", "").lower() in ["experience", "summary"]:
                job_roles.extend(section.get("text", "").lower().split())
        
        # Only suggest role-specific technologies if they have some experience in that area
        role_technologies = {
            "flutter": ["mobile", "android", "ios", "app"],
            "react native": ["mobile", "react", "javascript"],
            "devops": ["deployment", "server", "cloud", "infrastructure"],
            ".net": ["c#", "microsoft", "windows"],
            "php": ["web", "server", "backend"],
        }
        
        for tech, indicators in role_technologies.items():
            if tech in jd_skill.lower():
                if any(indicator in resume_text for indicator in indicators):
                    is_relevant = True
                break
        
        if is_relevant:
            relevant_additions.append(jd_skill)
            
    return relevant_additions

def generate_specific_recommendations(parsed_resume: dict, parsed_jd: dict, scoring: dict) -> List[str]:
    """
    Generates safe, specific, actionable recommendations that will directly improve ATS score.
    """
    suggestions = []
    
    # Get skills from both sides
    resume_skills_raw = parsed_resume.get("skills_canonical", []) + parsed_resume.get("skills_tokens", [])
    resume_skills = {canonicalize_skill(s).lower() for s in resume_skills_raw if canonicalize_skill(s)}
    
    jd_skills = []
    for skill_obj in parsed_jd.get("required_skills", []):
        if isinstance(skill_obj, dict):
            jd_skills.append(skill_obj.get("skill", ""))
        else:
            jd_skills.append(str(skill_obj))
    
    # Only suggest safe skill modifications
    safe_mapping_suggestions = {}
    
    for jd_skill in jd_skills:
        jd_skill_canonical = canonicalize_skill(jd_skill).lower()
        if jd_skill_canonical and jd_skill_canonical not in resume_skills:
            # Look for very close matches that are safe to suggest
            for resume_skill in resume_skills_raw:
                ratio = difflib.SequenceMatcher(None, jd_skill.lower(), resume_skill.lower()).ratio()
                # Higher threshold and safety check for replacements
                if ratio > 0.85 and is_safe_skill_replacement(resume_skill, jd_skill):
                    safe_mapping_suggestions[resume_skill] = jd_skill
                    break
    
    # Safe keyword enhancement suggestions (not replacements)
    if safe_mapping_suggestions:
        for old_term, new_term in safe_mapping_suggestions.items():
            suggestions.append(f"Consider adding '{new_term}' alongside '{old_term}' to better match job requirements.")
    
    # Find genuinely missing but relevant skills
    relevant_missing_skills = find_relevant_skill_additions(resume_skills, jd_skills, parsed_resume)
    
    # Handle critical soft skills
    critical_soft_skills = {
        "communication": ["Excellent Communication Skills", "Strong Communication Abilities", "Verbal and Written Communication"],
        "cognitive": ["Analytical Thinking", "Problem-Solving", "Critical Thinking", "Logical Reasoning"],
        "oop": ["Object-Oriented Programming (OOP)", "OOPS Concepts", "OOP Fundamentals", "Object-Oriented Design"]
    }
    
    for missing_skill in relevant_missing_skills:
        skill_lower = missing_skill.lower()
        added_suggestion = False
        
        for key, alternatives in critical_soft_skills.items():
            if key in skill_lower:
                suggestions.append(f"Add {' or '.join(alternatives[:2])} to highlight {key} abilities.")
                added_suggestion = True
                break
        
        if not added_suggestion and len(missing_skill) > 2:
            # Only suggest adding skills that are clearly missing and relevant
            suggestions.append(f"Add '{missing_skill}' to your skills section as it appears to be required for this role.")
    
    # Role-specific skill suggestions based on job titles
    job_title = parsed_jd.get("title", "").lower()
    
    safe_role_suggestions = {
        "python developer": ["Python", "Django", "Flask"],
        "react developer": ["React.js", "JavaScript", "HTML", "CSS"],
        "node.js developer": ["Node.js", "Express.js", "JavaScript"],
        "ai/ml developer": ["Machine Learning", "Python", "Data Analysis"],
        "qa developer": ["Testing", "Quality Assurance", "Test Automation"]
    }
    
    for role, skills in safe_role_suggestions.items():
        if role in job_title:
            missing_role_skills = [skill for skill in skills 
                                 if canonicalize_skill(skill).lower() not in resume_skills]
            if missing_role_skills:
                suggestions.append(f"For {role.replace('developer', 'Developer').title()}, consider highlighting: {', '.join(missing_role_skills[:2])}.")
    
    # Section and formatting improvements
    sections = {s.get("label", "").lower(): s.get("text", "") for s in parsed_resume.get("sections", [])}
    
    if not sections.get("skills") or len(sections.get("skills", "")) < 50:
        suggestions.append("Add a dedicated 'Technical Skills' section with comprehensive skill listing.")
    
    # SDLC and methodology suggestions
    experience_text = sections.get("experience", "")
    if not any(keyword in experience_text.lower() for keyword in ["sdlc", "agile", "development lifecycle"]):
        suggestions.append("Mention 'Software Development Life Cycle (SDLC)' or 'Agile methodology' in your experience descriptions.")
    
    # Safe formatting suggestions
    suggestions.append("Use exact terminology from the job description in your resume to improve keyword matching.")
    suggestions.append("Include skill variations and acronyms (e.g., 'Machine Learning (ML)', 'Object-Oriented Programming (OOP)') to capture different search terms.")
    
    # Project suggestions
    if not sections.get("projects") and not parsed_resume.get("projects"):
        suggestions.append("Add a 'Projects' section showcasing technical projects that demonstrate the required skills.")
    
    return suggestions

# ---------------------------
# Scoring functions (deterministic)
# ---------------------------

def cosine_map_to_0_100(cosine_val: float) -> float:
    # map [-1,1] to [0,100], clamp
    v = max(0.0, (cosine_val + 1.0) / 2.0 * 100.0)
    return max(0.0, min(100.0, v))

def fuzzy_match_score(skill_a: str, skill_b: str) -> float:
    # exact -> 1.0 ; fuzzy -> 0.8 if close; partial -> 0.5
    a = normalize_token(skill_a)
    b = normalize_token(skill_b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # try sequence matcher ratio
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if ratio >= 0.90:
        return 1.0
    if ratio >= 0.75:
        return 0.85
    if ratio >= 0.60:
        return 0.7
    if ratio >= 0.45:
        return 0.5
    # Check if one is contained in the other
    if a in b or b in a:
        return 0.6
    return 0.0

# Enhanced skill synonyms
skill_synonyms = {
    "database systems": ["firebase", "mysql", "postgresql", "mongodb", "database"],
    "analytical": ["analytical problem-solving", "problem solving", "problem-solving", "analytical thinking"],
    "big data": ["hadoop", "spark", "apache spark"],
    "communication skills": ["communication", "good communication", "strong communication"],
    "cognitive skills": ["cognitive", "analytical thinking", "problem solving", "critical thinking"],
    "object oriented programming": ["oop", "oops", "object-oriented", "oop fundamentals", "oops concepts"],
    "software development life cycle": ["sdlc", "development lifecycle", "software development"],
    "machine learning": ["ml", "tensorflow", "pytorch", "scikit-learn", "sklearn"],
    "natural language processing": ["nlp", "transformers", "bert", "distilbert"],
    "version control": ["git", "github", "version control system"],
    "rest api": ["api", "rest", "restful", "web api"],
    "javascript": ["js", "node.js", "react.js", "express.js"],
    "python": ["python programming", "python development"],
}

def compute_keyword_match_score(
    jd_required_skills: List[Dict[str, str]],
    resume_skills: List[str],
    parsed_resume: Dict[str, Any],
    skill_synonyms: Dict[str, List[str]] = None
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Enhanced keyword matching with better fuzzy matching and synonym support
    """
    if not jd_required_skills:
        return 100.0, []

    total_score = 0.0
    matched = []
    num = len(jd_required_skills)

    # Create comprehensive text for searching
    resume_text = json.dumps(parsed_resume, ensure_ascii=False).lower()
    resume_skills_lower = [s.lower() for s in resume_skills]
    
    # Also search in sections text specifically
    sections_text = " ".join([s.get("text", "") for s in parsed_resume.get("sections", [])]).lower()
    all_text = resume_text + " " + sections_text

    for jd_item in jd_required_skills:
        skill = jd_item.get("skill", "") if isinstance(jd_item, dict) else str(jd_item)
        skill_lower = skill.lower()
        importance = jd_item.get("importance", "required") if isinstance(jd_item, dict) else "required"
        importance_mul = {"required": 1.0, "preferred": 0.8, "nice-to-have": 0.5}.get(importance, 1.0)

        best = 0.0
        best_type = "none"

        # 1. Exact match in skills list
        for rsk in resume_skills:
            s = fuzzy_match_score(skill, rsk)
            if s > best:
                best = s
                best_type = "exact" if s >= 0.95 else ("fuzzy" if s >= 0.7 else ("partial" if s >= 0.4 else "none"))
                if best >= 1.0:
                    break

        # 2. Search in all resume text (more comprehensive)
        if best < 1.0:
            # Direct substring match
            if skill_lower in all_text:
                best = max(best, 0.9)
                best_type = "exact" if best >= 0.95 else "fuzzy"
            
            # Word boundary match (whole word)
            import re
            if re.search(r'\b' + re.escape(skill_lower) + r'\b', all_text):
                best = max(best, 0.95)
                best_type = "exact"

        # 3. Enhanced synonym matching
        if best < 0.8 and skill_synonyms:
            canonical_skill = canonicalize_skill(skill_lower)
            
            # Check direct synonym mapping
            if canonical_skill in skill_synonyms:
                for syn in skill_synonyms[canonical_skill]:
                    syn_lower = syn.lower()
                    if syn_lower in all_text or any(fuzzy_match_score(syn, rsk) >= 0.8 for rsk in resume_skills_lower):
                        best = max(best, 0.85)
                        best_type = "fuzzy"
                        break
            
            # Check reverse mapping (if skill is a synonym)
            for main_skill, synonyms in skill_synonyms.items():
                if skill_lower in [s.lower() for s in synonyms]:
                    if main_skill in all_text or any(fuzzy_match_score(main_skill, rsk) >= 0.8 for rsk in resume_skills_lower):
                        best = max(best, 0.85)
                        best_type = "fuzzy"
                        break

        # 4. Partial word matching for compound skills
        if best < 0.5:
            skill_words = skill_lower.split()
            if len(skill_words) > 1:
                word_matches = 0
                for word in skill_words:
                    if len(word) > 2 and word not in STOPWORDS:
                        if word in all_text:
                            word_matches += 1
                
                if word_matches >= len(skill_words) * 0.6:  # 60% of words match
                    best = max(best, 0.5)
                    best_type = "partial"

        weighted = best * importance_mul
        total_score += weighted

        if best > 0:
            matched.append({
                "skill": skill, 
                "resume_value": "present", 
                "jd_value": importance, 
                "match_type": best_type,
                "score": round(best, 2)
            })
        else:
            matched.append({
                "skill": skill, 
                "resume_value": "missing", 
                "jd_value": importance, 
                "match_type": "none",
                "score": 0.0
            })

    avg = (total_score / num) * 100.0 if num > 0 else 100.0
    return max(0.0, min(100.0, avg)), matched

def compute_section_coverage(parsed_resume: dict) -> float:
    # Enhanced section coverage scoring
    sections = {s.get("label","").lower(): s.get("text","") for s in parsed_resume.get("sections", [])}
    
    points = 0
    total = 4
    
    # Experience section
    if sections.get("experience","") and len(sections.get("experience","")) > 50:
        points += 1
    elif parsed_resume.get("experience_years", 0) > 0:
        points += 0.5  # partial credit
    
    # Skills section (most important for ATS)
    skills_text = sections.get("skills", "")
    skills_list = parsed_resume.get("skills_canonical", []) + parsed_resume.get("skills_tokens", [])
    if skills_text and len(skills_text) > 30:
        points += 1
    elif skills_list and len(skills_list) > 3:
        points += 0.8  # high partial credit
    
    # Education section
    if sections.get("education","") or parsed_resume.get("education"):
        points += 1
    
    # Projects/Certifications section
    if (sections.get("projects","") or parsed_resume.get("projects") or 
        sections.get("certifications","") or parsed_resume.get("certifications")):
        points += 1
    
    return (points / total) * 100.0

def compute_experience_alignment(candidate_years: float, jd_min_years: float) -> float:
    if jd_min_years <= 0:
        return 100.0
    if candidate_years >= jd_min_years:
        return 100.0
    ratio = candidate_years / jd_min_years
    # More forgiving for close matches
    if ratio >= 0.8:
        return 90.0
    elif ratio >= 0.5:
        return 75.0
    else:
        return max(0.0, ratio * 100.0)

def compute_certifications_score(parsed_resume: dict, parsed_jd: dict) -> float:
    resume_certs = []

    # Extract certifications from structured field
    cert_list = parsed_resume.get("certifications", [])
    for cert in cert_list:
        if isinstance(cert, dict):
            cert_name = cert.get("name", "")
        else:
            cert_name = str(cert)
        if cert_name.strip():
            resume_certs.append(cert_name.lower())

    # Also scan free-text sections for any certifications
    sections = parsed_resume.get("sections", [])
    for section in sections:
        if "cert" in section.get("label", "").lower():
            cert_text = section.get("text", "")
            cert_matches = re.findall(
                r'([A-Z0-9][A-Za-z0-9\s\-\–\&]+?(?:Certified|Certificate|Certification))',
                cert_text
            )
            resume_certs.extend([cert.lower().strip() for cert in cert_matches])

    # Get certifications mentioned in JD (priority_skills + required_skills)
    jd_priority = [s.lower() for s in parsed_jd.get("priority_skills", [])]
    jd_skills = [
        skill.get("skill", "").lower() if isinstance(skill, dict) else str(skill).lower()
        for skill in parsed_jd.get("required_skills", [])
    ]
    all_jd_requirements = jd_priority + jd_skills

    # Filter JD requirements to only ones that mention certification
    jd_cert_requirements = [
        req for req in all_jd_requirements
        if any(keyword in req for keyword in ["certified", "certificate", "certification"])
    ]

    # CASE 1: No certifications mentioned in JD → full marks
    if not jd_cert_requirements:
        return 100.0

    # CASE 2: JD requires certifications, but resume has none
    if not resume_certs:
        return 0.0

    # CASE 3: JD requires certifications → match them
    matches = 0
    for cert in resume_certs:
        cert_tokens = set(cert.replace("-", "-").split())
        for req in jd_cert_requirements:
            req_tokens = set(req.replace("-", "-").split())
            if cert_tokens & req_tokens or difflib.SequenceMatcher(None, cert, req).ratio() >= 0.7:
                matches += 1
                break

    score = (matches / len(jd_cert_requirements)) * 100.0
    return min(100.0, max(score, 0.0))

def compute_semantic_similarity_score(parsed_resume: dict, parsed_jd: dict) -> float:
    # Enhanced semantic similarity with better text aggregation
    resume_parts = [
        parsed_resume.get("summary", ""),
        " ".join([s.get("text", "") for s in parsed_resume.get("sections", [])]),
        " ".join(parsed_resume.get("skills_canonical", [])),
        " ".join(parsed_resume.get("skills_tokens", []))
    ]
    resume_text = " ".join([part for part in resume_parts if part.strip()])
    
    jd_parts = [
        parsed_jd.get("title", ""),
        " ".join(parsed_jd.get("responsibilities", [])),
        " ".join([s.get("skill", "") if isinstance(s, dict) else str(s) for s in parsed_jd.get("required_skills", [])]),
        " ".join(parsed_jd.get("priority_skills", []))
    ]
    jd_text = " ".join([part for part in jd_parts if part.strip()])
    
    if not resume_text.strip() or not jd_text.strip():
        return 50.0, 0.0
    
    emb_r = cache_embedding(resume_text)
    emb_j = cache_embedding(jd_text)
    cos = float(cosine_similarity([emb_r], [emb_j])[0, 0])
    
    # Enhanced mapping for better score distribution
    score = cosine_map_to_0_100(cos)
    # Boost score slightly for better alignment with keyword matching
    boosted_score = min(100.0, score * 1.1)
    
    return boosted_score, cos

# ---------------------------
# Top-level scoring orchestrator
# ---------------------------

def score_resume_against_jd(parsed_resume: dict, parsed_jd: dict) -> Dict[str,Any]:
    # Adjusted weights for better balance
    weights = {
        "semantic_similarity": 25,    
        "keyword_match": 45,          
        "section_coverage": 15,       
        "experience_alignment": 10,   
        "certifications_licenses": 5
    }
    
    # compute components
    sem_score, sem_cos = compute_semantic_similarity_score(parsed_resume, parsed_jd)
    jd_required_skills = parsed_jd.get("required_skills", [])
    resume_skills = parsed_resume.get("skills_tokens", []) or parsed_resume.get("skills_canonical", [])
    kw_score, matched_skills = compute_keyword_match_score(jd_required_skills, resume_skills, parsed_resume, skill_synonyms)
    section_score = compute_section_coverage(parsed_resume)
    candidate_years = float(parsed_resume.get("experience_years") or 0.0)
    jd_min_years = float(parsed_jd.get("min_experience_years") or 0.0)
    exp_score = compute_experience_alignment(candidate_years, jd_min_years)
    cert_score = compute_certifications_score(parsed_resume, parsed_jd)
    
    # breakdown entries
    breakdown = [
        {"component_name":"semantic_similarity","weight":weights["semantic_similarity"],"score": round(sem_score,2),
         "explanation": f"SBERT cosine {sem_cos:.4f} mapped to {sem_score:.1f}. Measures overall content similarity."},
        {"component_name":"keyword_match","weight":weights["keyword_match"],"score": round(kw_score,2),
         "explanation": f"Keyword matching score with enhanced fuzzy matching and synonym support."},
        {"component_name":"section_coverage","weight":weights["section_coverage"],"score": round(section_score,2),
         "explanation": "Presence and quality of core sections: experience, skills, education, projects/certs."},
        {"component_name":"experience_alignment","weight":weights["experience_alignment"],"score": round(exp_score,2),
         "explanation": f"Candidate {candidate_years}y vs JD min {jd_min_years}y requirement."},
        {"component_name":"certifications_licenses","weight":weights["certifications_licenses"],"score": round(cert_score,2),
         "explanation": "Relevant certifications and technical qualifications alignment."}
    ]
    
    # combine
    total = 0.0
    for comp in breakdown:
        total += comp["weight"] * comp["score"] / 100.0
    total_score = int(round(total))
    
    # missing high priority skills
    missing = [m["skill"] for m in matched_skills if m["match_type"] == "none" and m.get("jd_value") == "required"]
    
    return {
        "total_score": total_score,
        "deterministic": True,
        "breakdown": breakdown,
        "matched_skills": matched_skills,
        "missing_high_priority_skills": missing,
        "recommendations": []  # filled after LLM recommendations if needed
    }

# ---------------------------
# Enhanced local parser fallback
# ---------------------------

def extract_skills_from_text(text: str) -> List[str]:
    """Enhanced skill extraction for fallback parsing"""
    skills = set()
    
    # Common skill patterns
    skill_patterns = [
        r'\b(?:Python|Java|JavaScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin|Scala)\b',
        r'\b(?:React|Angular|Vue|Flask|Django|Express|Spring|Laravel)\b',
        r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|SQLite|Oracle)\b',
        r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins)\b',
        r'\b(?:Git|GitHub|GitLab|SVN)\b',
        r'\b(?:HTML|CSS|SASS|LESS|Bootstrap|Tailwind)\b',
        r'\b(?:TensorFlow|PyTorch|Scikit-learn|OpenCV|NLTK)\b',
        r'\b(?:REST|API|GraphQL|JSON|XML)\b',
        r'\b(?:Agile|Scrum|DevOps|CI/CD|SDLC)\b'
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        skills.update(matches)
    
    # Extract from skills sections
    skills_sections = re.findall(r'(?:skills|technologies|tools)[:\-\s]+((?:[^\n]|\n(?!\s*[A-Z]))*)', text, re.IGNORECASE | re.MULTILINE)
    
    for section in skills_sections:
        # Split by common delimiters
        section_skills = re.split(r'[,;|•\n\t]+', section)
        for skill in section_skills:
            skill = skill.strip()
            if skill and len(skill) > 1 and len(skill) < 30:
                normalized = canonicalize_skill(skill)
                if normalized:
                    skills.add(normalized)
    
    return list(skills)

def simple_local_parse_resume(text: str) -> Dict[str,Any]:
    # Enhanced fallback parser
    email = ""
    phone = ""
    name = ""
    location = ""
    
    # Extract contact info
    email_match = re.search(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', text)
    if email_match:
        email = email_match.group(1)
    
    phone_match = re.search(r'(\+?\d[\d\s\-\(\)]{7,}\d)', text)
    if phone_match:
        phone = phone_match.group(1)
    
    # Extract name (usually first line or near contact info)
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line and len(line) > 3 and len(line) < 50:
            # Simple heuristic: if it's not an email/phone and looks like a name
            if '@' not in line and not re.search(r'\d{3,}', line) and len(line.split()) <= 4:
                name = line
                break
    
    # Extract location
    location_patterns = [
        r'([A-Za-z\s]+,\s*[A-Za-z\s]+,?\s*[A-Za-z]*)',  # City, State format
        r'\b([A-Za-z]+,\s*[A-Z]{2,3})\b',  # City, State abbreviation
    ]
    for pattern in location_patterns:
        match = re.search(pattern, text)
        if match:
            potential_location = match.group(1)
            if 'university' not in potential_location.lower() and 'college' not in potential_location.lower():
                location = potential_location
                break
    
    # Enhanced skill extraction
    skills_list = extract_skills_from_text(text)
    
    # Extract experience years
    years = 0.0
    year_patterns = [
        r'(\d+)[\+]?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)',
        r'(?:experience|exp).*?(\d+)[\+]?\s*(?:years?|yrs?)',
    ]
    for pattern in year_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            years = float(match.group(1))
            break
    
    # Extract education info
    education = []
    edu_pattern = r'(B\.?Tech|B\.?E\.?|M\.?Tech|M\.?E\.?|Bachelor|Master|PhD).*?(?:in|of)?\s*([A-Za-z\s]+).*?(\d{4})?'
    edu_matches = re.findall(edu_pattern, text, re.IGNORECASE)
    for match in edu_matches:
        degree, field, year = match
        education.append({
            "degree": f"{degree.strip()} {field.strip()}".strip(),
            "institution": "",
            "start": "",
            "end": year if year else "",
            "grade": ""
        })
    
    # Create sections
    sections = []
    
    # Skills section
    if skills_list:
        sections.append({"label": "skills", "text": ", ".join(skills_list)})
    
    # Experience section (look for experience keywords)
    exp_keywords = ['experience', 'work', 'internship', 'position', 'role']
    exp_sections = []
    for keyword in exp_keywords:
        # pattern = f'{keyword}[:\-\s]+((?:[^\n]|\n(?!\s*[A-Z]))*)'
        pattern = f'{keyword}[:\\-\\s]+((?:[^\\n]|\\n(?!\\s*[A-Z]))*)'
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        exp_sections.extend(matches)
    
    if exp_sections:
        sections.append({"label": "experience", "text": " ".join(exp_sections)})
    
    # Education section
    if education:
        edu_text = "; ".join([f"{ed['degree']} {ed.get('institution', '')}".strip() for ed in education])
        sections.append({"label": "education", "text": edu_text})
    
    # Projects section
    project_pattern = r'(?:projects?|portfolio)[:\-\s]+((?:[^\n]|\n(?!\s*[A-Z]))*)'
    project_matches = re.findall(project_pattern, text, re.IGNORECASE | re.MULTILINE)
    if project_matches:
        sections.append({"label": "projects", "text": " ".join(project_matches)})
    
    parsed = {
        "name": name,
        "email": email,
        "phone": phone,
        "location": location,
        "summary": "",
        "sections": sections,
        "experience_years": years,
        "education": education,
        "skills_canonical": skills_list,
        "skills_tokens": skills_list,
        "projects": [],
        "certifications": []
    }
    return parsed

def enhanced_jd_parse(text: str) -> Dict[str, Any]:
    """Enhanced JD parsing for fallback"""
    
    # Extract basic info
    title_match = re.search(r'(?:job\s+profile|position|role)[:\-\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else ""
    
    company_match = re.search(r'(?:company\s+name)[:\-\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
    company = company_match.group(1).strip() if company_match else ""
    
    location_match = re.search(r'(?:work\s+location|location)[:\-\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
    location = location_match.group(1).strip() if location_match else ""
    
    # Extract skills from multiple sections
    skills = []
    
    # From skills required section
    skills_pattern = r'(?:skills?\s+required|technical\s+skills|required\s+skills)[:\-\s]+((?:[^\n]|\n(?!\s*[A-Z][^a-z]))*)'
    skills_matches = re.findall(skills_pattern, text, re.IGNORECASE | re.MULTILINE)
    
    for match in skills_matches:
        # Split by various delimiters
        skill_items = re.split(r'[,;•\n\t]+', match)
        for item in skill_items:
            item = item.strip().strip('- ').strip()
            if item and len(item) > 2 and len(item) < 50:
                skills.append({"skill": item, "importance": "required"})
    
    # From job roles/profiles
    if "developer" in title.lower():
        role_skills = []
        if "python" in title.lower():
            role_skills.extend(["Python", "Django", "Flask"])
        if "react" in title.lower():
            role_skills.extend(["React.js", "JavaScript", "HTML", "CSS"])
        if "node" in title.lower():
            role_skills.extend(["Node.js", "Express.js", "JavaScript"])
        if "devops" in title.lower():
            role_skills.extend(["Docker", "Kubernetes", "CI/CD"])
        if ".net" in title.lower():
            role_skills.extend([".NET", "C#", "ASP.NET"])
        if "flutter" in title.lower():
            role_skills.extend(["Flutter", "Dart", "Mobile Development"])
        if "qa" in title.lower():
            role_skills.extend(["Testing", "Quality Assurance", "Test Automation"])
        if "ai" in title.lower() or "ml" in title.lower():
            role_skills.extend(["Machine Learning", "AI", "Python", "TensorFlow"])
        if "data engineer" in title.lower():
            role_skills.extend(["Data Engineering", "ETL", "SQL", "Python"])
        if "php" in title.lower():
            role_skills.extend(["PHP", "Laravel", "MySQL"])
        
        for skill in role_skills:
            if not any(s["skill"].lower() == skill.lower() for s in skills):
                skills.append({"skill": skill, "importance": "required"})
    
    # Extract responsibilities
    resp_pattern = r'(?:basic\s+role|responsibilities|duties)[:\-\s]+((?:[^\n]|\n(?!\s*[A-Z][^a-z]))*)'
    resp_matches = re.findall(resp_pattern, text, re.IGNORECASE | re.MULTILINE)
    responsibilities = []
    for match in resp_matches:
        resp_items = re.split(r'[•\n]+', match)
        for item in resp_items:
            item = item.strip().strip('- ').strip()
            if item and len(item) > 5:
                responsibilities.append(item)
    
    return {
        "title": title,
        "company": company,
        "location": location,
        "seniority": "",
        "min_experience_years": 0,
        "required_skills": skills,
        "responsibilities": responsibilities,
        "hard_constraints": {"degree": "", "visa_sponsorship": "unknown"},
        "priority_skills": []
    }

# ---------------------------
# Main CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Enhanced Deterministic Resume Screening (LLM parse + SBERT score)")
    parser.add_argument("--resume", required=True, help="path to resume file (pdf/docx/txt)")
    parser.add_argument("--jd", required=True, help="path to job description file (txt/pdf/docx)")
    parser.add_argument("--outdir", default="./out", help="output directory")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM parsing and use local fallback parsing")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_resume = extract_text(args.resume)
    raw_jd = extract_text(args.jd)

    if args.no_llm:
        parsed_resume = simple_local_parse_resume(raw_resume)
        parsed_jd = enhanced_jd_parse(raw_jd)
    else:
        # call LLM to parse
        try:
            parsed_resume = run_parse_resume_llm(raw_resume)
        except Exception as e:
            print("LLM resume parse failed, falling back to local parse:", e, file=sys.stderr)
            parsed_resume = simple_local_parse_resume(raw_resume)
        try:
            parsed_jd = run_parse_jd_llm(raw_jd)
        except Exception as e:
            print("LLM JD parse failed, falling back to enhanced local parse:", e, file=sys.stderr)
            parsed_jd = enhanced_jd_parse(raw_jd)

    # Enhanced skill normalization
    sk_tokens = parsed_resume.get("skills_tokens") or []
    sk_canon = parsed_resume.get("skills_canonical") or []
    
    if not sk_canon and sk_tokens:
        sk_canon = [canonicalize_skill(s) for s in sk_tokens if canonicalize_skill(s)]
    if not sk_tokens and sk_canon:
        sk_tokens = sk_canon
    
    # Ensure comprehensive skill extraction
    all_skills = set()
    all_skills.update([s for s in sk_tokens if s])
    all_skills.update([s for s in sk_canon if s])
    
    # Extract additional skills from resume text
    additional_skills = extract_skills_from_text(raw_resume)
    all_skills.update(additional_skills)
    
    final_skills = list(all_skills)
    parsed_resume["skills_tokens"] = final_skills
    parsed_resume["skills_canonical"] = final_skills

    # Score
    scoring = score_resume_against_jd(parsed_resume, parsed_jd)

    # Generate specific recommendations
    specific_recs = generate_specific_recommendations(parsed_resume, parsed_jd, scoring)

    # Always generate recommendations (enhanced for all scores)
    try:
        sr = run_summary_recommend_llm(parsed_resume, parsed_jd, scoring)
        llm_recs = sr.get("recommendations", [])
        combined_recs = specific_recs + llm_recs
        scoring["recommendations"] = combined_recs
        summary_json = {
            "summary": sr.get("summary", "Resume analysis completed with actionable recommendations."),
            "recommendations": combined_recs
        }
    except Exception as e:
        print("LLM summary/recommendation failed, using specific recommendations:", e, file=sys.stderr)
        scoring["recommendations"] = specific_recs
        summary_json = {
            "summary": f"Resume scored {scoring['total_score']}/100. Key improvements needed in keyword matching and skill alignment.",
            "recommendations": specific_recs
        }

    # Save outputs
    (outdir / "parsed_resume.json").write_text(json.dumps(parsed_resume, indent=2, ensure_ascii=False), encoding='utf-8')
    (outdir / "parsed_jd.json").write_text(json.dumps(parsed_jd, indent=2, ensure_ascii=False), encoding='utf-8')
    (outdir / "scoring.json").write_text(json.dumps(scoring, indent=2, ensure_ascii=False), encoding='utf-8')
    (outdir / "summary_recommendations.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding='utf-8')
    
    print(f"Enhanced Analysis Complete!")
    print(f"Final Score: {scoring['total_score']}/100")
    print(f"Results saved to: {outdir}")
    print("\nScore Breakdown:")
    for comp in scoring['breakdown']:
        print(f"  {comp['component_name']}: {comp['score']:.1f}/100 (weight: {comp['weight']}%)")
    
    print(f"\nFiles generated:")
    print("  - parsed_resume.json (enhanced skill extraction)")
    print("  - parsed_jd.json (comprehensive requirement parsing)")  
    print("  - scoring.json (detailed scoring breakdown)")
    print("  - summary_recommendations.json (actionable improvements)")
    
    if scoring['total_score'] < 70:
        print(f"\nScore is below 70. Apply the recommendations to improve ATS matching!")
    else:
        print(f"\nGood score! Minor optimizations available in recommendations.")

if __name__ == "__main__":
    main()