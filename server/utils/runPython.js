import { spawn } from "child_process";
import fs from "fs/promises";
import path from "path";

export async function runPipeline({ resumePath, jdPath, outDir }) {
  console.log("⚙️ [Pipeline] Starting runPipeline...");
  console.log("➡️ Inputs:", { resumePath, jdPath, outDir });

  const args = [
    "-u",
    process.env.PY_SCRIPT,
    "--resume",
    resumePath,
    "--jd",
    jdPath,
    "--outdir",
    outDir
  ];

  if (String(process.env.PY_USE_NO_LLM).toLowerCase() === "true") {
    args.push("--no-llm");
    console.log("⚠️ [Pipeline] Running in NO-LLM mode");
  }

  // Ensure Azure vars are passed through environment (if needed)
  const env = { ...process.env };

  console.log("🐍 [Pipeline] Spawning Python process:", process.env.PYTHON_BIN || "python", args);

  const child = spawn(process.env.PYTHON_BIN || "python", args, { env });

  let stdout = "";
  let stderr = "";

  child.stdout.on("data", (d) => {
    const msg = d.toString();
    stdout += msg;
    console.log("📥 [Python STDOUT]:", msg.trim());
  });

  child.stderr.on("data", (d) => {
    const msg = d.toString();
    stderr += msg;
    console.error("⚠️ [Python STDERR]:", msg.trim());
  });

  const exitCode = await new Promise((res) => child.on("close", res));
  console.log("📤 [Pipeline] Python process exited with code:", exitCode);

  if (exitCode !== 0) {
    console.error("❌ [Pipeline] Python failed:", stderr || stdout);
    throw new Error(`Python failed (code ${exitCode}): ${stderr || stdout}`);
  }

  // Read the JSON outputs
  const readJSON = async (fname) => {
    const p = path.join(outDir, fname);
    console.log(`📄 [Pipeline] Reading output JSON: ${fname}`);
    const txt = await fs.readFile(p, "utf-8");
    const parsed = JSON.parse(txt);
    console.log(`✅ [Pipeline] Parsed ${fname} successfully`);
    return parsed;
  };

  const parsed_resume = await readJSON("parsed_resume.json");
  const parsed_jd = await readJSON("parsed_jd.json");
  const scoring = await readJSON("scoring.json");
  const summary_recommendations = await readJSON("summary_recommendations.json");

  console.log("🎯 [Pipeline] Finished reading all outputs successfully");

  // Optionally cleanup outDir here if needed
  return { parsed_resume, parsed_jd, scoring, summary_recommendations, rawStdout: stdout };
}
