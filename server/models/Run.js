import mongoose from "mongoose";

const runSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
    index: true
  },
  
  // GridFS file references instead of file paths
  resumeFileId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true
  },
  resumeOriginalName: {
    type: String,
    required: true
  },
  
  jdFileId: {
    type: mongoose.Schema.Types.ObjectId,
    required: false
  },
  jdOriginalName: {
    type: String,
    required: false
  },
  jdText: {
    type: String,
    required: false
  },
  
  status: {
    type: String,
    enum: ["processing", "done", "error"],
    default: "processing",
    index: true
  },
  
  error: {
    type: String,
    required: false
  },
  
  // Parsed results
  parsed_resume: {
    type: mongoose.Schema.Types.Mixed,
    required: false
  },
  parsed_jd: {
    type: mongoose.Schema.Types.Mixed,
    required: false
  },
  
  // Scoring results
  scoring: {
    type: mongoose.Schema.Types.Mixed,
    required: false
  },
  summary_recommendations: {
    type: mongoose.Schema.Types.Mixed,
    required: false
  },
  
  totalScore: {
    type: Number,
    required: false,
    min: 0,
    max: 100
  }
}, {
  timestamps: true
});

// Indexes for better query performance
runSchema.index({ userId: 1, createdAt: -1 });
runSchema.index({ status: 1, userId: 1 });

const Run = mongoose.model("Run", runSchema);

export default Run;