import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs/promises";
import sanitize from "sanitize-filename";
import { authRequired } from "../middleware/auth.js";
import Run from "../models/Run.js";
import { runPipeline } from "../utils/runPython.js";
import { getGridFSBucket } from "../config/gridfs.js";
import { Readable } from "stream";

const router = express.Router();

// Use memory storage instead of disk
const MAX_MB = Number(process.env.MAX_FILE_MB || 8);
const upload = multer({
  storage: multer.memoryStorage(), // Store in memory
  limits: { fileSize: MAX_MB * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ok = ["application/pdf",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain"].includes(file.mimetype);
    cb(ok ? null : new Error("Only PDF/DOC/DOCX/TXT allowed"), ok);
  }
});

// Helper: Upload buffer to GridFS
const uploadToGridFS = async (buffer, filename, metadata = {}) => {
  const bucket = getGridFSBucket();
  const readableStream = Readable.from(buffer);
  
  return new Promise((resolve, reject) => {
    const uploadStream = bucket.openUploadStream(filename, { metadata });
    
    uploadStream.on("finish", () => {
      resolve(uploadStream.id);
    });
    
    uploadStream.on("error", reject);
    
    readableStream.pipe(uploadStream);
  });
};

// Helper: Download from GridFS to temp file
const downloadFromGridFS = async (fileId) => {
  const bucket = getGridFSBucket();
  const tempPath = path.join("/tmp", `${fileId}_${Date.now()}.tmp`);
  
  return new Promise((resolve, reject) => {
    const downloadStream = bucket.openDownloadStream(fileId);
    const writeStream = require("fs").createWriteStream(tempPath);
    
    downloadStream.on("error", reject);
    writeStream.on("error", reject);
    writeStream.on("finish", () => resolve(tempPath));
    
    downloadStream.pipe(writeStream);
  });
};

// Helper: Create temp file from text
const createTempFile = async (text, filename) => {
  const tempPath = path.join("/tmp", filename);
  await fs.writeFile(tempPath, text, "utf-8");
  return tempPath;
};

// POST /api/ats/score
router.post("/score", authRequired, upload.fields([
  { name: "resumeFile", maxCount: 1 },
  { name: "jdFile", maxCount: 1 }
]), async (req, res) => {
  let resumeTempPath = null;
  let jdTempPath = null;
  
  try {
    const uid = req.user.id;

    const resumeF = req.files?.resumeFile?.[0];
    const jdF = req.files?.jdFile?.[0];
    const jdText = (req.body?.jdText || "").trim();

    if (!resumeF) return res.status(400).json({ message: "Resume file is required" });
    if (!jdF && !jdText) return res.status(400).json({ message: "Either JD file or JD text required" });

    // Upload resume to GridFS
    const safeName = (f) => sanitize(f.originalname || "file");
    const resumeFilename = `resume_${Date.now()}_${safeName(resumeF)}`;
    const resumeFileId = await uploadToGridFS(
      resumeF.buffer, 
      resumeFilename,
      { 
        userId: uid, 
        originalName: resumeF.originalname,
        mimetype: resumeF.mimetype 
      }
    );

    // Handle JD (either file or text)
    let jdFileId = null;
    let jdFilename = null;
    
    if (jdF) {
      jdFilename = `jd_${Date.now()}_${safeName(jdF)}`;
      jdFileId = await uploadToGridFS(
        jdF.buffer,
        jdFilename,
        { 
          userId: uid, 
          originalName: jdF.originalname,
          mimetype: jdF.mimetype 
        }
      );
    }

    // Create DB run (status: processing)
    const run = await Run.create({
      userId: uid,
      resumeFileId,
      resumeOriginalName: resumeF.originalname,
      jdFileId: jdFileId || undefined,
      jdOriginalName: jdF ? jdF.originalname : undefined,
      jdText: jdText || undefined,
      status: "processing"
    });

    // Download files from GridFS to temp directory for Python processing
    resumeTempPath = await downloadFromGridFS(resumeFileId);
    
    if (jdFileId) {
      jdTempPath = await downloadFromGridFS(jdFileId);
    } else {
      // Create temp file from text
      jdTempPath = await createTempFile(jdText, `jd_${Date.now()}.txt`);
    }

    // Create temp output directory
    const outDir = path.join("/tmp", "out", String(run._id));
    await fs.mkdir(outDir, { recursive: true });

    // Run pipeline
    let results;
    try {
      results = await runPipeline({ 
        resumePath: resumeTempPath, 
        jdPath: jdTempPath, 
        outDir 
      });
    } catch (err) {
      await Run.findByIdAndUpdate(run._id, { 
        status: "error", 
        error: String(err) 
      });
      return res.status(500).json({ 
        message: "Scoring failed", 
        error: String(err) 
      });
    }

    const { parsed_resume, parsed_jd, scoring, summary_recommendations } = results;
    const totalScore = Number(scoring?.total_score || 0);

    // Update run with results
    await Run.findByIdAndUpdate(run._id, {
      status: "done",
      parsed_resume,
      parsed_jd,
      scoring,
      summary_recommendations,
      totalScore
    }, { new: true });

    // Cleanup temp files
    try {
      await fs.unlink(resumeTempPath);
      await fs.unlink(jdTempPath);
      await fs.rm(outDir, { recursive: true, force: true });
    } catch (cleanupErr) {
      console.error("Cleanup error:", cleanupErr);
    }

    // Respond with final result
    return res.json({
      runId: run._id,
      totalScore,
      summary: summary_recommendations?.summary || "",
      recommendations: scoring?.recommendations || [],
      parsed_resume,
      parsed_jd,
      scoring
    });
  } catch (e) {
    console.error(e);
    
    // Cleanup on error
    try {
      if (resumeTempPath) await fs.unlink(resumeTempPath);
      if (jdTempPath) await fs.unlink(jdTempPath);
    } catch (cleanupErr) {
      console.error("Cleanup error:", cleanupErr);
    }
    
    return res.status(500).json({ message: "Server error" });
  }
});

// GET /api/ats/history
router.get("/history", authRequired, async (req, res) => {
  const runs = await Run.find({ userId: req.user.id })
    .sort({ createdAt: -1 })
    .select("-__v -updatedAt");
  res.json(runs);
});

// GET /api/ats/download/:runId/:fileType
router.get("/download/:runId/:fileType", authRequired, async (req, res) => {
  try {
    const { runId, fileType } = req.params;

    const run = await Run.findById(runId);
    if (!run || String(run.userId) !== String(req.user.id)) {
      return res.status(404).json({ message: "Not found" });
    }

    let fileId, originalName;
    
    if (fileType === "resume") {
      fileId = run.resumeFileId;
      originalName = run.resumeOriginalName || "resume.pdf";
    } else if (fileType === "jd") {
      if (run.jdFileId) {
        fileId = run.jdFileId;
        originalName = run.jdOriginalName || "job_description.pdf";
      } else if (run.jdText) {
        // Serve JD text directly
        res.setHeader("Content-Disposition", "attachment; filename=job_description.txt");
        res.setHeader("Content-Type", "text/plain");
        return res.send(run.jdText);
      }
    }

    if (!fileId) {
      return res.status(404).json({ message: "File not found" });
    }

    // Stream file from GridFS
    const bucket = getGridFSBucket();
    const downloadStream = bucket.openDownloadStream(fileId);

    downloadStream.on("error", (err) => {
      console.error("GridFS download error:", err);
      if (!res.headersSent) {
        res.status(404).json({ message: "File not found in storage" });
      }
    });

    res.setHeader("Content-Disposition", `attachment; filename="${originalName}"`);
    res.setHeader("Content-Type", "application/octet-stream");
    
    downloadStream.pipe(res);
  } catch (e) {
    console.error(e);
    if (!res.headersSent) {
      res.status(500).json({ message: "Server error" });
    }
  }
});

// GET /api/ats/status/latest
router.get("/status/latest", authRequired, async (req, res) => {
  const run = await Run.findOne({ userId: req.user.id })
    .sort({ createdAt: -1 });

  if (!run) {
    return res.json({ active: false });
  }

  return res.json({
    active: run.status === "processing",
    status: run.status,
    runId: run._id,
    totalScore: run.totalScore || null
  });
});

// GET /api/ats/result/:id
router.get("/result/:id", authRequired, async (req, res) => {
  const run = await Run.findOne({ _id: req.params.id, userId: req.user.id });
  if (!run) return res.status(404).json({ message: "Not found" });
  return res.json(run);
});

export default router;