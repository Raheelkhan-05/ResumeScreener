import "dotenv/config";
import express from "express";
import cors from "cors";
import { connectDB } from "./config/db.js";
import { initGridFS } from "./config/gridfs.js";
import authRoutes from "./routes/auth.js";
import atsRoutes from "./routes/ats.js";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// Health check endpoint (should be before auth routes)
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

app.use("/api/auth", authRoutes);
app.use("/api/ats", atsRoutes);

const PORT = process.env.PORT || 4000;

// Connect to MongoDB and initialize GridFS
connectDB(process.env.MONGO_URI).then(() => {
  // Initialize GridFS after DB connection
  initGridFS();
  
  app.listen(PORT, () => {
    console.log(`🚀 Server listening on :${PORT}`);
    console.log(`📦 GridFS initialized for file storage`);
  });
}).catch(err => {
  console.error("Failed to start server:", err);
  process.exit(1);
});