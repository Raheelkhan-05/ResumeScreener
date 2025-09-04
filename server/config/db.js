import mongoose from "mongoose";

export const connectDB = async (uri) => {
  await mongoose.connect(uri, { autoIndex: true });
  console.log("MongoDB connected");
};
