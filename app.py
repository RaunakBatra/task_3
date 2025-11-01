# app.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import os


# ----------------------------------------------------
# 1️⃣ Initialize FastAPI app
# ----------------------------------------------------
app = FastAPI(title="Food Recommendation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------
# 2️⃣ Load dataset
# ----------------------------------------------------
df = pd.read_csv("final_recipe.csv")

df["combined_text"] = (
    df["recipe_name"].fillna('') + " " +
    df["ingredients_list"].fillna('') + " " +
    df["cuisine_type"].fillna('') + " " +
    df["region_type"].fillna('') + " " +
    df["veg_nonveg"].fillna('')
)

df["ingredients_list"] = df["ingredients_list"].astype(str)


# ----------------------------------------------------
# 3️⃣ TF-IDF vectorizer
# ----------------------------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["ingredients_list"])


# ----------------------------------------------------
# 4️⃣ Veg/Non-Veg classification
# ----------------------------------------------------
nonveg_keywords = ['chicken','mutton','egg','fish','pork','beef','bacon','meat','turkey','lamb','shrimp']

def get_veg_nonveg(name, ingredients):
    text = str(name).lower() + " " + str(ingredients).lower()
    return "Non-Veg" if any(word in text for word in nonveg_keywords) else "Veg"

df["veg_nonveg"] = df.apply(lambda x: get_veg_nonveg(x["recipe_name"], x["ingredients_list"]), axis=1)


# ----------------------------------------------------
# 5️⃣ Cuisine classification
# ----------------------------------------------------
cuisine_keywords = {
    'Indian': ['paneer','masala','biryani','dal','curry','roti','paratha','sambar','tikka','idli','dosa'],
    'Chinese': ['noodle','fried rice','manchurian','schezwan','spring roll','momo'],
    'Italian': ['pizza','pasta','lasagna','spaghetti','mozzarella','garlic bread'],
    'Mexican': ['taco','burrito','nacho','quesadilla','salsa','tortilla'],
    'American': ['burger','fries','sandwich','steak','pancake','donut'],
    'Japanese': ['sushi','ramen','tempura','miso','teriyaki'],
    'French': ['croissant','souffle','crepe','baguette','ratatouille'],
    'Thai': ['pad thai','green curry','tom yum','lemongrass'],
    'Korean': ['kimchi','bibimbap','bulgogi','kimbap','gochujang'],
}

def detect_cuisine_fuzzy(name, ingredients, threshold=80):
    text = str(name).lower() + " " + str(ingredients).lower()
    best_match, best_score = None, 0
    for cuisine, keywords in cuisine_keywords.items():
        for word in keywords:
            score = fuzz.partial_ratio(word, text)
            if score > best_score:
                best_match, best_score = cuisine, score
    return best_match if best_score >= threshold else 'Other'

df["cuisine_type"] = df.apply(lambda x: detect_cuisine_fuzzy(x["recipe_name"], x["ingredients_list"]), axis=1)


# ----------------------------------------------------
# 6️⃣ Region classification
# ----------------------------------------------------
region_keywords = {
    'North Indian': ['paneer','butter chicken','naan','dal makhani','paratha'],
    'South Indian': ['idli','dosa','sambar','rasam','pongal'],
    'East Indian': ['momo','machher jhol','rasgulla','litti chokha'],
    'West Indian': ['dhokla','thepla','vada pav','pav bhaji','poha']
}

def detect_region_rapid(name, ingredients, threshold=80):
    text = (str(name) + " " + str(ingredients)).lower()
    best_match, best_score = None, 0
    for region, keywords in region_keywords.items():
        for word in keywords:
            score = fuzz.partial_ratio(word, text)
            if score > best_score:
                best_match, best_score = region, score
    return best_match if best_score >= threshold else 'Other Region'

df["region_type"] = df.apply(lambda x: detect_region_rapid(x["recipe_name"], x["ingredients_list"]), axis=1)


# ----------------------------------------------------
# 7️⃣ Recommendation functions
# ----------------------------------------------------
def recommend_food(input_value, top_n=10, by='name'):
    if by == 'name':
        all_recipes = df['recipe_name'].str.lower().tolist()
        best_match, score, _ = process.extractOne(input_value.lower(), all_recipes)
        idx = df[df['recipe_name'].str.lower() == best_match].index[0]
    elif by == 'url':
        idx = df[df['image_url'] == input_value].index[0]
    else:
        return pd.DataFrame()

    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    return df.iloc[similar_indices][['recipe_id','recipe_name','image_url','cuisine_type','region_type','veg_nonveg','aver_rate']]


def recommend_by_nutrition(calories=None, protein=None, fat=None, carbs=None, top_n=10):
    df_copy = df.copy()
    df_copy["nutrition_score"] = 0

    if calories:
        df_copy["nutrition_score"] += abs(df_copy["calories"] - calories)
    if protein:
        df_copy["nutrition_score"] += abs(df_copy["protein"] - protein)
    if fat:
        df_copy["nutrition_score"] += abs(df_copy["fat"] - fat)
    if carbs:
        df_copy["nutrition_score"] += abs(df_copy["carbohydrates"] - carbs)

    best_matches = df_copy.sort_values("nutrition_score").head(top_n)
    return best_matches[[
        "recipe_name", "calories", "protein", "fat", "carbohydrates",
        "cuisine_type", "region_type", "veg_nonveg", "image_url"
    ]]


pairing_rules = {
    'burger': ['fries','coke','ice cream'],
    'pizza': ['garlic bread','coke','salad'],
    'biryani': ['raita','gulab jamun','lassi'],
    'pasta': ['garlic bread','juice','brownie'],
}

def recommend_pair(food_name, top_n=5):
    all_recipes = df['recipe_name'].str.lower().tolist()
    best_match, score, _ = process.extractOne(food_name.lower(), all_recipes)
    base_recipe = df[df['recipe_name'].str.lower() == best_match]

    cuisine = base_recipe['cuisine_type'].values[0]
    region = base_recipe['region_type'].values[0]
    vegtype = base_recipe['veg_nonveg'].values[0]

    df_pair = df[
        (df['cuisine_type'] == cuisine) &
        (df['region_type'] == region) &
        (df['veg_nonveg'] == vegtype)
    ]

    if df_pair.empty:
        df_pair = df.sample(top_n)
    else:
        df_pair = df_pair.sample(min(top_n, len(df_pair)))

    return df_pair[[
        "recipe_name", "cuisine_type", "region_type", "veg_nonveg",
        "aver_rate", "image_url"
    ]]


# ----------------------------------------------------
# 8️⃣ API Endpoints
# ----------------------------------------------------
@app.get("/")
def home():
    return {"message": "🍴 Food Recommendation API is running successfully!"}


@app.get("/recommend")
def recommend_food_advanced(
    food_name: Optional[str] = Query(None, description="Food name or ingredient"),
    veg_nonveg: Optional[str] = Query(None, description="Veg or Non-Veg"),
    cuisine_type: Optional[str] = Query(None, description="Cuisine type"),
    region_type: Optional[str] = Query(None, description="Region type"),
    min_calories: Optional[float] = Query(0, description="Minimum calories"),
    max_calories: Optional[float] = Query(2000, description="Maximum calories")
):
    try:
        df_filtered = df.copy()

        # 🥗 Apply filters
        if veg_nonveg:
            df_filtered = df_filtered[df_filtered["veg_nonveg"].str.lower() == veg_nonveg.lower()]
        if cuisine_type:
            df_filtered = df_filtered[df_filtered["cuisine_type"].str.lower() == cuisine_type.lower()]
        if region_type:
            df_filtered = df_filtered[df_filtered["region_type"].str.lower() == region_type.lower()]

        df_filtered = df_filtered[
            (df_filtered["calories"] >= min_calories) & (df_filtered["calories"] <= max_calories)
        ]

        if df_filtered.empty:
            return {"message": "No recipes found with the given filters."}

        # 🍽️ Recommend based on food name
        if food_name:
            df_filtered = df_filtered.reset_index(drop=True)
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(df_filtered["combined_text"])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

            match = df_filtered[df_filtered["recipe_name"].str.lower().str.contains(food_name.lower())]
            if match.empty:
                return {"message": f"No recipe found similar to '{food_name}'."}

            match_idx = match.index[0]
            sim_scores = list(enumerate(cosine_sim[match_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:8]
            indices = [i[0] for i in sim_scores]
            results = df_filtered.iloc[indices]
        else:
            # 🎲 Random sample if no name provided
            results = df_filtered.sample(min(6, len(df_filtered)))

        return results[[
            "recipe_name", "calories", "protein", "fat", "carbohydrates",
            "veg_nonveg", "cuisine_type", "region_type", "image_url"
        ]].to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}


@app.get("/recommend_by_url")
def recommend_by_url_endpoint(image_url: str):
    recommendations = recommend_food(image_url, by='url')
    return recommendations.to_dict(orient='records')


@app.get("/recommend_by_nutrition")
def recommend_by_nutrition_endpoint(
    calories: Optional[int] = Query(None),
    protein: Optional[int] = Query(None),
    fat: Optional[int] = Query(None),
    carbs: Optional[int] = Query(None)
):
    recommendations = recommend_by_nutrition(calories, protein, fat, carbs)
    return recommendations.to_dict(orient='records')


@app.get("/recommend_pair")
def recommend_pair_endpoint(food_name: str):
    recommendations = recommend_pair(food_name)
    return recommendations.to_dict(orient='records')


# ----------------------------------------------------
# 9️⃣ Run app
# ----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
