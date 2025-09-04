import mongoose from "mongoose";

const RunSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", index: true, required: true },

  // Inputs
  resumeFilePath: String,
  resumeOriginalName: String,
  jdFilePath: String,
  jdOriginalName: String,
  jdText: String,               // if JD was provided as raw text



  // Outputs
  parsed_resume: Object,
  parsed_jd: Object,
  scoring: Object,
  summary_recommendations: Object,
  totalScore: Number,

  // Status
  status: { type: String, enum: ["processing", "done", "error"], default: "processing", index: true },
  error: String
}, { timestamps: true });

export default mongoose.model("Run", RunSchema);
