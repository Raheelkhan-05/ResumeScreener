import mongoose from "mongoose";
import { GridFSBucket } from "mongodb";

let bucket;

export const initGridFS = () => {
  if (!bucket) {
    bucket = new GridFSBucket(mongoose.connection.db, {
      bucketName: "uploads"
    });
    console.log("✅ GridFS initialized");
  }
  return bucket;
};

export const getGridFSBucket = () => {
  if (!bucket) {
    throw new Error("GridFS not initialized. Call initGridFS first.");
  }
  return bucket;
};