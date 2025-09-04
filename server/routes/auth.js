import express from "express";
import jwt from "jsonwebtoken";
import User from "../models/User.js";

const router = express.Router();

// --- SIGNUP ---
router.post("/signup", async (req, res) => {
  const { email, password, username } = req.body || {};
  
  if (!email || !password || !username) {
    return res.status(400).json({ message: "Email, username & password required" });
  }

  // Check for duplicate email or username
  const exists = await User.findOne({ $or: [{ email }, { username }] });
  if (exists) {
    return res.status(409).json({ message: "Email or username already in use" });
  }

  const user = await User.signup(email, password, username);
  const token = jwt.sign(
    { id: user._id },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN || "7d" }
  );

  res.json({ 
    token, 
    user: { id: user._id, email: user.email, username: user.username } 
  });
});

// --- LOGIN ---
router.post("/login", async (req, res) => {
  const { email, password } = req.body || {};
  const user = await User.findOne({ email });
  if (!user || !(await user.verifyPassword(password))) {
    return res.status(401).json({ message: "Invalid credentials" });
  }

  const token = jwt.sign(
    { id: user._id },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN || "7d" }
  );

  res.json({ 
    token, 
    user: { id: user._id, email: user.email, username: user.username } 
  });
});

export default router;
