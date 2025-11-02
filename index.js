import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import "dotenv/config";
import connectDB from "./config/connection.js";
import authRoutes from "./routes/authRoutes.js";
import userRouter from "./routes/userRoutes.js";

import recipeRoutes from "./routes/recipeRoutes.js";


const app = express();
const PORT = process.env.PORT || 8000;

// Connect MongoDB
connectDB(process.env.MONGODB_URI);

// Middleware
app.use(express.json());
app.use(cookieParser());
app.use(
  cors({
    origin: "http://localhost:5173", // frontend URL
    credentials: true,
  })
);

// Test Route
app.get("/", (req, res) => {
  res.send(" API is working!");
});

// API Routes
app.use("/api/auth", authRoutes);
app.use("/api/user", userRouter);

app.use("/api/recipes", recipeRoutes);


// Start Server
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
