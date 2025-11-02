import express from "express";
import { getAllRecipes } from "../controllers/recipeController.js";

const router = express.Router();

// GET /api/recipes
router.get("/", getAllRecipes);

export default router;
