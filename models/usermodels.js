import mongoose from "mongoose";

const userSchema = new mongoose.Schema(
  {
    name: { 
      type: String,
       required: true },
    email: { 
      type: String, 
      required: true,
      unique: true },
    password: { 
      type: String, 
      required: true },
    isAccountVerified: { 
      type: Boolean, 
      default: false },
    verifyOtp: { 
      type: String, 
      default: "" },
    verifyOtpExpireAt: {
       type: Date },
    resendOtp:{
      type:String,
      default:""
    },
    resendOtpExpireAt:{
      type:Number,
      default:0
    }

  },
  { timestamps: true }
);

export default mongoose.model("User", userSchema);
