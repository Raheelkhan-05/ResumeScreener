import { Link } from "react-router-dom";
import { useAuth } from "./context/AuthContext.js";

export default function App() {
  const { user, logout } = useAuth();
  
  return (
    <div className="landing-container">
      <div className="hero-content">
        
        {user ? (
          <p className="opacity-80 mb-3" style={{ color: 'rgba(255, 255, 255, 0.8)', fontWeight:'700', fontSize:'24px',  }}>
              {console.log(user)}
                Welcome back, <span className="font-semibold">{user.username}</span>
              </p>
        ) : (<p className="text-sm opacity-60" style={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                
              </p>)}
        <h1 className="hero-title" style={{fontVariant:'small-caps', letterSpacing:'5px'}}>Resume Screener</h1>
        <p className="hero-subtitle" style={{fontVariant:'small-caps'}}>
          Upload. Analyze. Improve. Get Instant ATS Scores And Personalized Tips To Land Your Dream Job.
        </p>
        
        {user ? (
          <div className="hero-actions">
            
            <Link to="/dashboard" className="btn btn-primary">
              Go to Dashboard
            </Link>
            <button onClick={logout} className="btn btn-secondary">
              Logout
            </button>
          </div>
        ) : (
          <div className="hero-actions">
            <Link to="/login" className="btn btn-primary">
              Get Started
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}