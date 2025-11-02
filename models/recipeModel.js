import mongoose from "mongoose";

const recipeSchema = new mongoose.Schema(
  {
    recipe_id: { type: Number, required: true },
    recipe_name: { type: String, required: true },
    aver_rate: { type: Number },
    image_url: { type: String },
    review_nums: { type: Number },
    calories: { type: Number },
    fat: { type: Number },
    carbohydrates: { type: Number },
    protein: { type: Number },
    cholesterol: { type: Number },
    sodium: { type: Number },
    fiber: { type: Number },
    ingredients_list: { type: String }, // stored as a string in your dataset
    veg_nonveg: { type: String },
    cuisine_type: { type: String },
    region_type: { type: String },
  },
  { timestamps: true }
);

export default mongoose.model("Recipe", recipeSchema, "recipes");
