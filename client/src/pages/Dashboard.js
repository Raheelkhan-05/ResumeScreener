import UploadAndScore from "../components/UploadAndScore.js";
import HistoryList from "../components/HistoryList.js";
import { useAuth } from "../context/AuthContext.js";
import { useState } from "react";
import { Link } from "react-router-dom";
import { scale } from "framer-motion";

export default function Dashboard() {
  const { logout } = useAuth();
  const [latestRunId, setLatestRunId] = useState(null);

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h2 className="dashboard-title" style={{ fontVariant: 'small-caps',letterSpacing:'1.5px' }}>
          Resume Screener Dashboard
        </h2>

        <div className="dashboard-actions">
          <Link to="/" className="btn home">Home</Link>
          <button onClick={logout} className="btn btn-danger">Logout</button>
        </div>
      </div>
      
      <div className="dashboard-grid">
        <UploadAndScore onNewResult={(runId) => setLatestRunId(runId)} />
        <HistoryList refreshTrigger={latestRunId} />
      </div>
    </div>
  );
}
