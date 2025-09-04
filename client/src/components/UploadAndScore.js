import { useState, useEffect } from "react";
import api from "../api";
import ResultCard from "./ResultCard.js";

export default function UploadAndScore({ onNewResult }) {
  const [resumeFile, setResumeFile] = useState(null);
  const [jdFile, setJdFile] = useState(null);
  const [jdText, setJdText] = useState("");
  const [jdProfile, setJdProfile] = useState(""); 
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [result, setResult] = useState(null);

  const acceptTypes = ".pdf,.doc,.docx,.txt";

  // Default JD texts for fresher profiles
  const PROFILE_JDS = {
  "Software Engineer": `
Job Title: Software Engineer (Fresher)  
Company Overview:  
We are a fast-growing technology company dedicated to building scalable software systems and delivering impactful digital solutions across industries. Our team values innovation, collaboration, and a passion for solving real-world problems with code.  

Role Summary:  
As a Fresher Software Engineer, you will contribute to the design, development, testing, and deployment of web and backend applications. You will work with senior engineers and product teams to translate requirements into clean, efficient, and maintainable code. This is an excellent opportunity to kickstart your career by working on production-grade projects.  

Key Responsibilities:  
- Write, test, and maintain clean, efficient code in languages such as Java, Python, or JavaScript.  
- Collaborate with cross-functional teams including product managers, designers, and QA engineers.  
- Assist in debugging, performance optimization, and code reviews.  
- Learn and apply best practices in software design, version control, and deployment pipelines.  
- Contribute to API development, database design, and cloud-based deployments.  

Qualifications:  
- Bachelor’s degree in Computer Science, IT, or related field (or equivalent practical experience).  
- Strong understanding of Data Structures, Algorithms, and Object-Oriented Programming.  
- Familiarity with databases (SQL/NoSQL) and web technologies (HTML, CSS, JavaScript).  
- Problem-solving mindset with eagerness to learn new tools and frameworks.  

Preferred Skills:  
- Exposure to version control (Git/GitHub).  
- Basic knowledge of cloud platforms (AWS, GCP, or Azure).  
- Understanding of Agile/Scrum methodology.  

Why Join Us?  
- Work on live projects with mentorship from experienced engineers.  
- Gain exposure to cutting-edge technologies.  
- Supportive environment with career growth opportunities.  
`,

  "Data Scientist": `
Job Title: Data Scientist (Fresher)  
Company Overview:  
We are an analytics-driven company helping organizations unlock value from their data. Our mission is to combine data science, AI, and business insights to drive better decision-making and innovation.  

Role Summary:  
As a Fresher Data Scientist, you will support the design and development of data models, machine learning algorithms, and visualization dashboards. You’ll work with real-world datasets to uncover patterns, solve business problems, and present actionable insights.  

Key Responsibilities:  
- Clean, preprocess, and analyze structured and unstructured datasets.  
- Build and validate basic machine learning models (classification, regression, clustering).  
- Assist in feature engineering, model evaluation, and reporting.  
- Create data visualizations and dashboards to communicate insights.  
- Collaborate with business analysts and engineers to translate requirements into models.  

Qualifications:  
- Bachelor’s degree in Computer Science, Statistics, Data Science, or related field.  
- Strong foundation in Probability, Statistics, and Linear Algebra.  
- Proficiency in Python with libraries such as Pandas, NumPy, Scikit-learn, and Matplotlib.  
- SQL knowledge for querying databases.  

Preferred Skills:  
- Familiarity with TensorFlow, PyTorch, or Keras.  
- Knowledge of big data frameworks (Spark, Hadoop) is a plus.  
- Experience with data visualization tools (Power BI, Tableau).  

Why Join Us?  
- Work on impactful projects with real business data.  
- Gain hands-on exposure to end-to-end ML pipelines.  
- Be mentored by experienced data scientists in a collaborative setting.  
`,

  "Database Administrator": `
Job Title: Database Administrator (Fresher)  
Company Overview:  
We provide robust and scalable data management solutions to enterprises across industries. Our focus is on ensuring high availability, security, and performance of mission-critical databases.  

Role Summary:  
As a Fresher Database Administrator (DBA), you will be responsible for assisting in the design, implementation, maintenance, and security of databases. You will ensure data integrity, optimize performance, and support business applications that depend on reliable data access.  

Key Responsibilities:  
- Install, configure, and maintain relational databases (MySQL, PostgreSQL, Oracle).  
- Monitor performance, troubleshoot issues, and tune databases for efficiency.  
- Assist in backup, recovery, and disaster recovery planning.  
- Implement data security policies and user access controls.  
- Work with developers to design normalized schemas and optimize queries.  

Qualifications:  
- Bachelor’s degree in Computer Science, IT, or related field.  
- Strong understanding of SQL and relational database concepts.  
- Knowledge of database design, normalization, and indexing.  
- Familiarity with backup and recovery strategies.  

Preferred Skills:  
- Exposure to NoSQL databases (MongoDB, Cassandra).  
- Understanding of cloud databases (AWS RDS, Azure SQL, GCP Cloud SQL).  
- Knowledge of database monitoring tools and scripting for automation.  

Why Join Us?  
- Hands-on learning with enterprise-grade databases.  
- Opportunity to grow into advanced DBA or Data Engineer roles.  
- Work in a supportive environment with structured career progression.  
`
};


  // --- POLLING ---
  useEffect(() => {
    let interval;
    async function checkStatus() {
      try {
        const res = await api.get("/ats/status/latest");
        if (res.data.active) {
          setLoading(true);
        } else {
          setLoading(false);
          if (res.data.status === "done") {
            const runRes = await api.get(`/ats/result/${res.data.runId}`);
            setResult(runRes.data);
            if (onNewResult) onNewResult(runRes.data.runId);
          }
          clearInterval(interval);
        }
      } catch (err) {
        console.error("Status check failed", err);
      }
    }
    checkStatus();
    interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, [onNewResult]);

  const submit = async (e) => {
    e.preventDefault();
    setErr("");
    setResult(null);

    if (!resumeFile) return setErr("Resume file is required (PDF, DOC, DOCX, or TXT)");
    if (!jdFile && !jdText.trim() && !jdProfile) return setErr("Please provide a job description");

    const fd = new FormData();
    fd.append("resumeFile", resumeFile);

    if (jdFile) fd.append("jdFile", jdFile);
    else if (jdText.trim()) fd.append("jdText", jdText);
    else if (jdProfile) fd.append("jdText", PROFILE_JDS[jdProfile]);

    setLoading(true);
    try {
      const { data } = await api.post("/ats/score", fd, { 
        headers: { "Content-Type": "multipart/form-data" }
      });
      
      const newResult = {
        totalScore: data.totalScore,
        summary: data.summary_recommendations?.summary,
        recommendations: data.summary_recommendations?.recommendations,
        scoring: data.scoring,
        runId: data.runId
      };

      setResult(newResult);
      if (onNewResult) onNewResult(data.runId);
    } catch (e) {
      setErr(e?.response?.data?.message || "Failed to analyze resume");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h3 className="card-title" style={{ fontVariant: 'small-caps' }}>Resume Analysis</h3>

      {err && <div className="error-message">{err}</div>}

      <form onSubmit={submit} className="upload-form">
        {/* Resume Upload */}
        <div>
          <label className="form-label">📄 Resume File *</label>
          <div className={`file-input-wrapper ${resumeFile ? 'has-file' : ''}`}>
            <input
              type="file"
              accept={acceptTypes}
              onChange={e => setResumeFile(e.target.files[0] || null)}
              required
              className="file-input"
            />
            <div className="file-input-label">
              {resumeFile ? (
                <div>
                  <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>✅</div>
                  <div>{resumeFile.name}</div>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📤</div>
                  <div>Click or drag to upload resume</div>
                  <div style={{ fontSize: '0.875rem', opacity: 0.7, marginTop: '0.5rem' }}>
                    PDF, DOC, DOCX, TXT supported
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* JD Section (all 3 options wrapped together) */}
        <div>
          <label className="form-label">📋 Job Description (choose one option)</label>
          <div className="jd-wrapper">
            
            {/* JD File Upload */}
            <div className={`file-input-wrapper ${jdFile ? 'has-file' : ''}`}>
              <input
                type="file"
                accept={acceptTypes}
                onChange={e => {
                  setJdFile(e.target.files[0] || null);
                  if (e.target.files[0]) {
                    setJdText("");
                    setJdProfile("");
                  }
                }}
                className="file-input"
              />
              <div className="file-input-label">
                {jdFile ? (
                  <div>
                    <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>✅</div>
                    <div>{jdFile.name}</div>
                  </div>
                ) : (
                  <div>
                    <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📎</div>
                    <div>Upload job description file</div>
                  </div>
                )}
              </div>
            </div>
            <div className="text-center" style={{maxHeight:'10px'}}>----------------OR----------------</div>

            {/* JD Text */}
            <textarea
              rows={6}
              placeholder="Paste the complete job description here..."
              value={jdText}
              onChange={e => { 
                setJdText(e.target.value);
                if (e.target.value) {
                  setJdFile(null);
                  setJdProfile("");
                }
              }}
              className="textarea"
            />
            <div className="text-center" style={{maxHeight:'10px'}}>----------------OR----------------</div>
            {/* JD Profile Selection */}
            <select
              value={jdProfile}
              onChange={e => {
                setJdProfile(e.target.value);
                if (e.target.value) {
                  setJdFile(null);
                  setJdText("");
                }
              }}
              className="form-input jd-select"
            >
              <option value="">-- Select a Profile --</option>
              <option value="Software Engineer">Software Engineer (Fresher)</option>
              <option value="Data Scientist">Data Scientist (Fresher)</option>
              <option value="Database Administrator">Database Administrator (Fresher)</option>
            </select>

          </div>
        </div>

        <button type="submit" disabled={loading} className="btn btn-primary" style={{transform:"scale(1.0)", fontVariant:'small-caps'}}>
          {loading ? (
            <>
              <div className="spinner"></div>
              Analyzing Resume...
            </>
          ) : (
            <>Analyze Resume</>
          )}
        </button>
      </form>

      {loading && (
        <div className="loading-container">
          <div className="spinner"></div>
          <div>
            <div style={{ fontWeight: '600', marginBottom: '0.5rem' }}>
              AI analysis in progress...
            </div>
            <div style={{ fontSize: '0.875rem', opacity: '0.8' }}>
              Processing resume against job requirements. This may take up to 30 seconds.
            </div>
          </div>
        </div>
      )}

      {result && <ResultCard result={result} />}
    </div>
  );
}
