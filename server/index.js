import "dotenv/config";
import express from "express";
import cors from "cors";
import path from "path";
import { connectDB } from "./config/db.js";
import authRoutes from "./routes/auth.js";
import atsRoutes from "./routes/ats.js";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// static for uploaded files (demo-only)
app.use("/uploads", express.static(path.join(process.cwd(), "server", "uploads")));

app.use("/api/auth", authRoutes);
app.use("/api/ats", atsRoutes);

const PORT = process.env.PORT || 4000;
connectDB(process.env.MONGO_URI).then(() => {
  app.listen(PORT, () => console.log(`Server listening on :${PORT}`));
});
