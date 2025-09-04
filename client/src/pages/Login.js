import { useState } from "react";
import { useAuth } from "../context/AuthContext.js";
import { useNavigate } from "react-router-dom";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";

export default function Login() {
  const nav = useNavigate();
  const { login, signup } = useAuth();

  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [mode, setMode] = useState("login");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const onSubmit = async (e) => {
    e.preventDefault();
    setErr(""); 
    setLoading(true);

    try {
      if (mode === "signup") {
        if (password !== confirmPassword) {
          setErr("Passwords do not match");
          return;
        }
        await signup(email, password, username);
      } else {
        await login(email, password);
      }
      nav("/dashboard");
    } catch (e) {
      setErr(e?.response?.data?.message || "Authentication failed");
    } finally { 
      setLoading(false); 
    }
  };

  return (
    <div className="login-container">
      <motion.div 
        className="login-card"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <h2 className="login-title" style={{ fontVariant: "small-caps" }}>
          {mode === "login" ? "Welcome Back" : "Create Account"}
        </h2>
        
        {err && <div className="error-message">{err}</div>}
        
        <form onSubmit={onSubmit} className="login-form">
          <AnimatePresence mode="wait">
            {mode === "signup" && (
              <motion.div
                key="username"
                className="form-group"
                initial={{ opacity: 0, y: 0 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, y: 20 }}
                transition={{ duration: 0.3 }}
              >
                <input
                  required
                  type="text"
                  placeholder="Username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="form-input"
                />
              </motion.div>
            )}
          </AnimatePresence>

          <div className="form-group">
            <input 
              required 
              type="email" 
              placeholder="Email address" 
              value={email} 
              onChange={(e) => setEmail(e.target.value)}
              className="form-input"
            />
          </div>
          
          <div className="form-group">
            <input 
              required 
              minLength={6} 
              type="password" 
              placeholder="Password (min 6 characters)" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)}
              className="form-input"
            />
          </div>

          <AnimatePresence mode="wait">
            {mode === "signup" && (
              <motion.div
                key="confirmPassword"
                className="form-group"
                initial={{ opacity: 0, y: 0 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
              >
                <input 
                  required 
                  minLength={6} 
                  type="password" 
                  placeholder="Confirm Password" 
                  value={confirmPassword} 
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="form-input"
                />
              </motion.div>
            )}
          </AnimatePresence>
          
          <button 
            disabled={loading} 
            type="submit"
            className="btn btn-primary"
          >
            {loading ? "Please wait..." : (mode === "login" ? "Sign In" : "Create Account")}
          </button>
        </form>
        
        <div className="text-center mt-3">
          <button 
            onClick={() => setMode(mode === "login" ? "signup" : "login")}
            className="btn-link"
          >
            {mode === "login" ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
          </button>
        </div>
        <div className="text-center mt-3">
        <Link to="/" className="btn btn-secondary" style={{maxHeight:'20px'}}>
              Go to Home
        </Link>
        </div>
      </motion.div>
    </div>
  );
}
