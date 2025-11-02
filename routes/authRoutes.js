import express from "express";
import { register, login, logout, sendVerifyOtp, verifyEmail , resendVerifyOtp,sendResetOtp, resetPassword} from "../controllers/authController.js";
import userAuth from "../middleware/usermiddle.js";

const router = express.Router();

router.post("/register", register);
router.post("/login", login);
router.post("/logout", logout);
router.post("/sendverifyotp", userAuth, sendVerifyOtp);
router.post("/verifyaccount", userAuth, verifyEmail);
router.post("/resendverifyotp", userAuth, resendVerifyOtp);
router.post("/sendResetOtp", sendResetOtp);
router.post("/resetPassword", resetPassword);


export default router;
