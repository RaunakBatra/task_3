import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional  # ✅ FIXED: added Optional import
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import uvicorn

app = FastAPI(title="Food Recommendation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load dataset
df = pd.read_csv("final_recipe.csv")

# --- Data Cleaning ---
nutrition_features = ['calories', 'protein', 'fat', 'carbohydrates']
df[nutrition_features] = df[nutrition_features].fillna(df[nutrition_features].mean())

# --- Nutrition-based Clustering ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[nutrition_features])
kmeans = KMeans(n_clusters=8, random_state=42)
df['nutrition_cluster'] = kmeans.fit_predict(X_scaled)

# --- Veg/Non-Veg Detection ---
nonveg_keywords = [
    'chicken','mutton','egg','fish','pork','beef','bacon',
    'meat','turkey','lamb','shrimp','crab','prawn','ham','sausage'
]

df["veg_nonveg"] = df.apply(
    lambda x: "Non-Veg" if any(k in str(x["recipe_name"]).lower() + " " + str(x["ingredients_list"]).lower() for k in nonveg_keywords)
    else "Veg",
    axis=1
)

# --- Cuisine Classification ---
cuisine_keywords = {
    'Indian': ['paneer','masala','biryani','dal','curry','roti','paratha','tikka','idli','dosa','sabzi','korma','pulao','kachori'],
    'Chinese': ['noodle','fried rice','manchurian','schezwan','spring roll','momo','chowmein'],
    'Italian': ['pizza','pasta','lasagna','spaghetti','mozzarella','risotto','bruschetta'],
    'Mexican': ['taco','burrito','nacho','quesadilla','salsa','tortilla','enchilada'],
    'American': ['burger','fries','sandwich','steak','pancake','donut','brownie'],
    'Japanese': ['sushi','ramen','tempura','miso','teriyaki'],
    'French': ['croissant','souffle','crepe','baguette','ratatouille'],
    'Thai': ['pad thai','green curry','tom yum','lemongrass','basil'],
    'Korean': ['kimchi','bibimbap','bulgogi','kimbap','gochujang']
}

def detect_cuisine(name, ingredients):
    text = f"{name} {ingredients}".lower()
    scores = {cuisine: sum(word in text for word in words) for cuisine, words in cuisine_keywords.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"

df["cuisine_type"] = df.apply(lambda x: detect_cuisine(x["recipe_name"], x["ingredients_list"]), axis=1)

# --- Region Classification ---
region_keywords = {
    'North Indian': ['paneer','butter chicken','naan','dal makhani','paratha','chole','rajma','korma'],
    'South Indian': ['idli','dosa','sambar','rasam','pongal','uttapam','curd rice'],
    'East Indian': ['momo','machher jhol','rasgulla','litti chokha','sandesh'],
    'West Indian': ['dhokla','thepla','vada pav','pav bhaji','poha','shrikhand']
}

def detect_region(name, ingredients):
    text = f"{name} {ingredients}".lower()
    scores = {region: sum(word in text for word in words) for region, words in region_keywords.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other Region"

df["region_type"] = df.apply(lambda x: detect_region(x["recipe_name"], x["ingredients_list"]), axis=1)

# --- TF-IDF Feature Extraction ---
df["ingredients_list"] = df["ingredients_list"].astype(str)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["ingredients_list"])

# --- Weighted Score Calculation ---
if "aver_rate" in df.columns and "review_nums" in df.columns:
    C = df["aver_rate"].mean()
    m = df["review_nums"].quantile(0.60)
    df["weighted_score"] = df.apply(
        lambda x: (x["review_nums"]/(x["review_nums"]+m))*x["aver_rate"] + (m/(x["review_nums"]+m))*C
        if not pd.isna(x["aver_rate"]) and not pd.isna(x["review_nums"]) else C, axis=1
    )
else:
    df["weighted_score"] = np.random.uniform(3.0, 4.5, len(df))

# --- Routes ---

@app.get("/")
def home():
    return {"message": "🍴 Food Recommendation API is running!"}


@app.get("/recommend")
def recommend_food_dynamic(food_name: Optional[str]=Query(None),
                           veg_nonveg: Optional[str]=Query(None),
                           cuisine_type: Optional[str]=Query(None),
                           region_type: Optional[str]=Query(None)):
    try:
        df_filtered = df.copy()
        searched_idx = None

        if food_name:
            match_result = process.extractOne(food_name.lower(), df['recipe_name'].str.lower().tolist())
            if match_result:
                best_match = match_result[0]
                searched_idx = df.index[df["recipe_name"].str.lower() == best_match][0]

        if veg_nonveg:
            df_filtered = df_filtered[df_filtered["veg_nonveg"].str.lower() == veg_nonveg.lower()]
        if cuisine_type:
            df_filtered = df_filtered[df_filtered["cuisine_type"].str.lower() == cuisine_type.lower()]
        if region_type:
            df_filtered = df_filtered[df_filtered["region_type"].str.lower() == region_type.lower()]

        if df_filtered.empty:
            df_filtered = df.copy()

        if searched_idx is not None:
            tfidf_filtered = tfidf.transform(df_filtered["ingredients_list"])
            cosine_sim = linear_kernel(tfidf_matrix[searched_idx], tfidf_filtered).flatten()
            df_filtered["similarity"] = cosine_sim
            df_filtered["final_score"] = df_filtered["similarity"] * 0.6 + df_filtered["weighted_score"] * 0.4
            df_filtered = df_filtered.sort_values(by="final_score", ascending=False)
            results = df_filtered.head(7)
        else:
            results = df_filtered.sort_values(by="weighted_score", ascending=False).head(7)

        return results[[
            "recipe_name","calories","protein","fat","carbohydrates",
            "veg_nonveg","cuisine_type","region_type","weighted_score","image_url"
        ]].to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}


def get_current_season():
    month = datetime.now().month
    if month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "monsoon"
    else:
        return "winter"


@app.get("/recommend_by_season")
def recommend_by_season(season: Optional[str] = Query(None), top_n: int = 7):
    try:
        if not season:
            season = get_current_season()
        season = season.lower()

        season_keywords = {
            "summer": ["mango", "buttermilk", "lassi", "salad", "lemon", "coconut", "ice cream", "kulfi", "juice", "smoothie"],
            "winter": ["soup", "paratha", "halwa", "gajar", "saag", "tea", "hot chocolate", "samosa"],
            "monsoon": ["pakora", "chai", "tea", "bhajiya", "corn", "soup", "poha", "khichdi"],
            "spring": ["fruit", "salad", "sandwich", "yogurt", "herb", "strawberry", "lemon"]
        }

        keywords = season_keywords.get(season)
        if not keywords:
            return {"error": f"No data for season '{season}'. Try summer, winter, monsoon, or spring."}

        df_seasonal = df[
            df["recipe_name"].str.lower().apply(lambda x: any(k in x for k in keywords)) |
            df["ingredients_list"].str.lower().apply(lambda x: any(k in x for k in keywords))
        ]

        if df_seasonal.empty:
            df_seasonal = df.sort_values(by="weighted_score", ascending=False).head(top_n)

        return df_seasonal[[
            "recipe_name","cuisine_type","region_type","veg_nonveg","weighted_score","image_url"
        ]].head(top_n).to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}


@app.get("/recommend_by_nutrition")
def recommend_by_nutrition(calories: Optional[float]=Query(None),
                           protein: Optional[float]=Query(None),
                           fat: Optional[float]=Query(None),
                           carbs: Optional[float]=Query(None),
                           top_n: int = 5):
    try:
        input_values = np.array([[calories or 0, protein or 0, fat or 0, carbs or 0]])
        input_scaled = scaler.transform(input_values)
        distances = euclidean_distances(input_scaled, X_scaled).flatten()
        similar_indices = np.argsort(distances)[:top_n]
        return df.iloc[similar_indices][[
            "recipe_name","calories","protein","fat","carbohydrates","image_url"
        ]].to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


@app.get("/recommend_pair")
def recommend_pair(food_name: str, top_n: int = 5):
    try:
        best_match, _, _ = process.extractOne(food_name.lower(), df['recipe_name'].str.lower().tolist())
        base = df[df['recipe_name'].str.lower()==best_match]
        cuisine, region, vegtype = base['cuisine_type'].values[0], base['region_type'].values[0], base['veg_nonveg'].values[0]
        df_pair = df[(df['cuisine_type']==cuisine) & (df['region_type']==region) & (df['veg_nonveg']==vegtype)]
        if df_pair.empty:
            df_pair = df.sort_values(by="weighted_score", ascending=False).head(top_n)
        else:
            df_pair = df_pair.sort_values(by="weighted_score", ascending=False).head(top_n)
        return df_pair[["recipe_name","cuisine_type","region_type","veg_nonveg","weighted_score","image_url"]].to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
