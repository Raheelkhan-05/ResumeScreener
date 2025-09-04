import { useEffect, useState } from "react";
import api from "../api";

function downloadFile(runId, type, filename) {
  api.get(`/ats/download/${runId}/${type}`, { responseType: "blob" })
    .then(res => {
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
    })
    .catch(err => {
      alert("Failed to download file: " + (err?.response?.data?.message || err.message));
    });
}

export default function HistoryList({ refreshTrigger }) {
  const [runs, setRuns] = useState([]);
  const [err, setErr] = useState("");

  const fetchHistory = () => {
    api.get("/ats/history")
      .then(({data}) => setRuns(data))
      .catch(e => setErr(e?.response?.data?.message || "Failed to load history"));
  };

  useEffect(() => {
    fetchHistory();
  }, []); // initial load

  // refresh whenever refreshTrigger changes (new result)
  useEffect(() => {
    if (refreshTrigger) {
      console.log("🔄 Refreshing history due to new result:", refreshTrigger);
      fetchHistory();
    }
  }, [refreshTrigger]);

  return (
    <div className="card">
      <h3 className="card-title" style={{ fontVariant: 'small-caps' }}>Analysis History</h3>
      
      {err && <div className="error-message">{err}</div>}
      
      {!err && runs.length === 0 && (
        <div className="empty-state">
          <div className="empty-state-title">No previous analyses</div>
          <div className="empty-state-text">Your resume analysis history will appear here</div>
        </div>
      )}
      
      {runs.length > 0 && (
        <ul className="history-list">
          {runs.map(r => (
            <li key={r._id} className="history-item">
              <div className="history-info">
                <div className="history-date">
                  {new Date(r.createdAt).toLocaleDateString()} at {new Date(r.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
                <div className="history-score">
                  Score: {r.totalScore}/100
                </div>
                <div className="history-status">
                  {r.status}
                </div>
              </div>
              
              <div className="history-actions">
                {r.resumeOriginalName && (
                  <button 
                    onClick={() => downloadFile(r._id, "resume", r.resumeOriginalName)}
                    className="btn btn-ghost"
                    title={r.resumeOriginalName}
                  >
                    📄 Resume
                  </button>
                )}
                
                {(r.jdOriginalName || r.jdText) && (
                  <button 
                    onClick={() => downloadFile(r._id, "jd", r.jdOriginalName || "JobDescription.txt")}
                    className="btn btn-ghost"
                    title={r.jdOriginalName || "Job Description"}
                  >
                    📋 Job Des.
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}