import mongoose from "mongoose";
import bcrypt from "bcryptjs";

const UserSchema = new mongoose.Schema({
  username: { type: String, unique: true, required: true }, 
  email: { type: String, unique: true, index: true, required: true },
  passwordHash: { type: String, required: true },
}, { timestamps: true });

UserSchema.methods.verifyPassword = function (pw) {
  return bcrypt.compare(pw, this.passwordHash);
};

UserSchema.statics.signup = async function (email, password, username) {
  const hash = await bcrypt.hash(password, 10);
  return this.create({ email, username, passwordHash: hash });
};

export default mongoose.model("User", UserSchema);
