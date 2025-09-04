import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs/promises";
// import fs from "fs";
import sanitize from "sanitize-filename";
import { authRequired } from "../middleware/auth.js";
import Run from "../models/Run.js";
import { runPipeline } from "../utils/runPython.js";

const router = express.Router();

// Multer disk storage (demo). In prod: S3/GridFS/etc.
const MAX_MB = Number(process.env.MAX_FILE_MB || 8);
const upload = multer({
  dest: path.join(process.cwd(), "uploads"),
  limits: { fileSize: MAX_MB * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ok = ["application/pdf",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain"].includes(file.mimetype);
    cb(ok ? null : new Error("Only PDF/DOC/DOCX/TXT allowed"), ok);
  }
});

// POST /api/ats/score
// multipart/form-data:
// - resumeFile (pdf/doc/docx/txt) [required]
// - jdFile (pdf/doc/docx/txt) OR jdText (string) [one required]

router.post("/score", authRequired, upload.fields([
  { name: "resumeFile", maxCount: 1 },
  { name: "jdFile", maxCount: 1 }
]), async (req, res) => {
  try {
    const uid = req.user.id;

    const resumeF = req.files?.resumeFile?.[0];
    const jdF = req.files?.jdFile?.[0];
    const jdText = (req.body?.jdText || "").trim();

    if (!resumeF) return res.status(400).json({ message: "Resume file is required" });
    if (!jdF && !jdText) return res.status(400).json({ message: "Either JD file or JD text required" });

    // Normalized file names
    const safeName = (f) => sanitize(f.originalname || "file");
    const resumePath = path.join(resumeF.destination, `${Date.now()}_${safeName(resumeF)}`);
    await fs.rename(resumeF.path, resumePath);

    let jdPath = null;
    if (jdF) {
      jdPath = path.join(jdF.destination, `${Date.now()}_${safeName(jdF)}`);
      await fs.rename(jdF.path, jdPath);
    } else {
      // If JD text, write to a temp txt file
      const tempName = `jd_${Date.now()}.txt`;
      jdPath = path.join(process.cwd(), "uploads", tempName);
      await fs.writeFile(jdPath, jdText, "utf-8");
    }

    // Create DB run (status: processing)
    const run = await Run.create({
        userId: uid,
        resumeFilePath: resumePath,
        resumeOriginalName: resumeF.originalname,   
        jdFilePath: jdPath,
        jdOriginalName: jdF ? jdF.originalname : undefined,
        jdText: jdF ? undefined : jdText,
        status: "processing"
    });

    // Run pipeline (synchronously for simplicity)
    const outDir = path.join(process.cwd(), "out", String(run._id));
    await fs.mkdir(outDir, { recursive: true });

    let results;
    try {
      results = await runPipeline({ resumePath, jdPath, outDir });
    } catch (err) {
      await Run.findByIdAndUpdate(run._id, { status: "error", error: String(err) });
      return res.status(500).json({ message: "Scoring failed", error: String(err) });
    }

    const { parsed_resume, parsed_jd, scoring, summary_recommendations } = results;
    const totalScore = Number(scoring?.total_score || 0);

    await Run.findByIdAndUpdate(run._id, {
      status: "done",
      parsed_resume,
      parsed_jd,
      scoring,
      summary_recommendations,
      totalScore
    }, { new: true });

    // respond with final result
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
    return res.status(500).json({ message: "Server error" });
  }
});


// router.post(
//   "/score",
//   authRequired,
//   upload.fields([
//     { name: "resumeFile", maxCount: 1 },
//     { name: "jdFile", maxCount: 1 }
//   ]),
//   async (req, res) => {
//     try {
//       console.log("📥 Incoming /score request");

//       const uid = req.user.id;
//       console.log("➡️ User ID:", uid);

//       const resumeF = req.files?.resumeFile?.[0];
//       const jdF = req.files?.jdFile?.[0];
//       const jdText = (req.body?.jdText || "").trim();

//       if (!resumeF) {
//         console.warn("⚠️ No resume file provided");
//         return res.status(400).json({ message: "Resume file is required" });
//       }
//       if (!jdF && !jdText) {
//         console.warn("⚠️ No JD provided (neither file nor text)");
//         return res.status(400).json({ message: "Either JD file or JD text required" });
//       }

//       // Normalized file names
//       const safeName = (f) => sanitize(f.originalname || "file");

//       const resumePath = path.join(
//         resumeF.destination,
//         `${Date.now()}_${safeName(resumeF)}`
//       );
//       await fs.rename(resumeF.path, resumePath);
//       console.log("📄 Resume saved at:", resumePath);

//       let jdPath = null;
//       if (jdF) {
//         jdPath = path.join(jdF.destination, `${Date.now()}_${safeName(jdF)}`);
//         await fs.rename(jdF.path, jdPath);
//         console.log("📄 JD file saved at:", jdPath);
//       } else {
//         const tempName = `jd_${Date.now()}.txt`;
//         jdPath = path.join(process.cwd(), "server", "uploads", tempName);
//         await fs.writeFile(jdPath, jdText, "utf-8");
//         console.log("📝 JD text written to file:", jdPath);
//       }

//       // Create DB run (status: processing)
//       const run = await Run.create({
//         userId: uid,
//         resumeFilePath: resumePath,
//         jdFilePath: jdPath,
//         jdText: jdF ? undefined : jdText,
//         status: "processing"
//       });
//       console.log("✅ Run created in DB with ID:", run._id);

//       // Run pipeline (synchronously for simplicity)
//       const outDir = path.join(process.cwd(), "server", "out", String(run._id));
//       await fs.mkdir(outDir, { recursive: true });
//       console.log("📂 Output directory created at:", outDir);

//       let results;
//       try {
//         console.log("⚙️ Running pipeline...");
//         results = await runPipeline({ resumePath, jdPath, outDir });
//         console.log("✅ Pipeline finished successfully");
//       } catch (err) {
//         console.error("❌ Pipeline execution failed:", err);
//         await Run.findByIdAndUpdate(run._id, {
//           status: "error",
//           error: String(err)
//         });
//         return res.status(500).json({
//           message: "Scoring failed",
//           error: String(err)
//         });
//       }

//       const {
//         parsed_resume,
//         parsed_jd,
//         scoring,
//         summary_recommendations
//       } = results;
//       const totalScore = Number(scoring?.total_score || 0);
//       console.log("📊 Total Score calculated:", totalScore);

//       await Run.findByIdAndUpdate(
//         run._id,
//         {
//           status: "done",
//           parsed_resume,
//           parsed_jd,
//           scoring,
//           summary_recommendations,
//           totalScore
//         },
//         { new: true }
//       );
//       console.log("✅ Run updated to 'done' in DB");

//       // respond with final result
//       console.log("📤 Sending response for run ID:", run._id);
//       return res.json({
//         runId: run._id,
//         totalScore,
//         summary: summary_recommendations?.summary || "",
//         recommendations: scoring?.recommendations || [],
//         parsed_resume,
//         parsed_jd,
//         scoring
//       });
//     } catch (e) {
//       console.error("💥 Server error:", e);
//       return res.status(500).json({ message: "Server error" });
//     }
//   }
// );


// GET /api/ats/history
router.get("/history", authRequired, async (req, res) => {
  const runs = await Run.find({ userId: req.user.id })
    .sort({ createdAt: -1 })
    .select("-__v -updatedAt");
  res.json(runs);
});


router.get("/download/:runId/:fileType", authRequired, async (req, res) => {
  const { runId, fileType } = req.params;

  const run = await Run.findById(runId);
  if (!run || String(run.userId) !== String(req.user.id)) {
    return res.status(404).json({ message: "Not found" });
  }

  let filePath, originalName;
  if (fileType === "resume") {
    filePath = run.resumeFilePath;
    originalName = run.resumeOriginalName || "resume.pdf";
  } else if (fileType === "jd") {
    if (run.jdFilePath) {
      filePath = run.jdFilePath;
      originalName = run.jdOriginalName || "job_description.pdf";
    } else if (run.jdText) {
      // Serve JD text as .txt download
      res.setHeader("Content-Disposition", "attachment; filename=job_description.txt");
      res.setHeader("Content-Type", "text/plain");
      return res.send(run.jdText);
    }
  }

  if (!filePath) {
    return res.status(404).json({ message: "File not found" });
  }
  res.setHeader("Content-Disposition", `attachment; filename="${originalName}"`);
  res.setHeader("Content-Type", "application/octet-stream");
  res.download(filePath, originalName);
});


// Get latest run for current user
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

// Get full result of a run
router.get("/result/:id", authRequired, async (req, res) => {
  const run = await Run.findOne({ _id: req.params.id, userId: req.user.id });
  if (!run) return res.status(404).json({ message: "Not found" });
  // console.log("--------------",run)
  return res.json(run);
});



export default router;


