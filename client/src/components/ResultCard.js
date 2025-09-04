import { useEffect, useState } from "react";

export default function ResultCard({ result, onNewResult }) {
  const { 
    totalScore, 
    summary, 
    recommendations, 
    summary_recommendations 
  } = result || {};

  const [animatedScore, setAnimatedScore] = useState(0);
  const [showDetails, setShowDetails] = useState(false);

  const finalSummary = summary || summary_recommendations?.summary || "";
  const finalRecommendations = recommendations || summary_recommendations?.recommendations || [];

  // Notify parent
  useEffect(() => {
    if (result && onNewResult) {
      onNewResult(result._id);
    }
  }, [result, onNewResult]);

  // Animate score smoothly
  useEffect(() => {
    if (totalScore !== undefined) {
      let start = 0;
      const step = () => {
        start += 2;
        if (start <= totalScore) {
          setAnimatedScore(start);
          requestAnimationFrame(step);
        }
      };
      requestAnimationFrame(step);
    }
  }, [totalScore]);

  const scrollToDetails = () => {
    setShowDetails(true);
    setTimeout(() => {
      document.getElementById("details-section")?.scrollIntoView({ 
        behavior: "smooth",
        block: "start"
      });
    }, 150);
  };

  const getScoreColor = (score) => {
    if (score >= 80) return "#10b981"; // Green
    if (score >= 60) return "#f59e0b"; // Yellow
    return "#ef4444"; // Red
  };

  const getScoreMessage = (score) => {
    if (score >= 90) return "Excellent Match";
    if (score >= 80) return "Strong Match";
    if (score >= 60) return "Good Match";
    if (score >= 40) return "Fair Match";
    return "Needs Improvement ❌";
  };

  // Half-circle gauge values
  const radius = 90;
  const circumference = Math.PI * radius; // half circle
  const progress = (animatedScore / 100) * circumference;

  return (
    <div className="result-card card">
      <h3 className="card-title" style={{ fontVariant: 'small-caps' }}>Last Run</h3>
      <div className="score-gauge-container">
        {/* Half-circle Gauge */}
        <div className="gauge-wrapper">
          <svg className="gauge-svg" viewBox="0 0 240 100">
            <defs>
              <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#667eea" />
                <stop offset="50%" stopColor="#764ba2" />
                <stop offset="100%" stopColor="#f093fb" />
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge> 
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/> 
                </feMerge>
              </filter>
            </defs>

            {/* Background arc */}
            <path
              d="M 30 90 A 90 90 0 0 1 210 90"
              stroke="#e5e7eb"
              strokeWidth="12"
              fill="none"
            />

            {/* Progress arc */}
            <path
              d="M 30 90 A 90 90 0 0 1 210 90"
              stroke="url(#gaugeGradient)"
              strokeWidth="12"
              fill="none"
              strokeDasharray={`${progress} ${circumference}`}
              filter="url(#glow)"
              style={{
                transition: "stroke-dasharray 0.5s ease-out",
              }}
            />
          </svg>
          <div className="gauge-text">{animatedScore}/100</div>
        </div>
        
        <div className="score-status">
          <div className="score-label">ATS Compatibility Score</div>
          <div className="score-message text-center" style={{ color: getScoreColor(totalScore) }}>
            {getScoreMessage(totalScore)}
          </div>
        </div>
        
      </div>

      {/* Details */}
      <div id="details-section" className={`details-section ${showDetails ? "visible" : ""}`}>
        {/* Summary */}
        {finalSummary && (
          <div className="result-section">
            <h4 className="section-title">📝 Executive Summary</h4>
            <div className="summary-card">
              <p className="summary-text">{finalSummary}</p>
            </div>
          </div>
        )}

        {/* Recommendations */}
        {totalScore >= 90 ? (
          <div className="result-section">
            <h4 className="section-title">💡 Recommendations</h4>
            <div className="summary-card text-green-600 font-semibold">
              🚀 No major improvements needed — your resume looks job-ready!
            </div>
          </div>
        ) : (
          Array.isArray(finalRecommendations) &&
          finalRecommendations.length > 0 && (
            <div className="result-section">
              <h4 className="section-title">💡 Recommendations</h4>
              <ul className="recommendations-list">
                {finalRecommendations.map((r, i) => (
                  <li key={i} className="recommendation-item flex items-start gap-2">
                    <span className="recommendation-icon">👉 </span>
                    <span className="recommendation-text">{String(r)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )
        )}
      </div>
    </div>
  );
}
