import Recipe from "../models/recipeModel.js";

// ✅ Return all recipes as an array of objects
export const getAllRecipes = async (req, res) => {
  try {
    const recipes = await Recipe.find({}).lean(); // use lean() for plain JS objects
    res.json(recipes); // send only array
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
