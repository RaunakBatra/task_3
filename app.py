import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from rapidfuzz import fuzz, process
import uvicorn
from collections import Counter
import re

app = FastAPI(title="Food Recommendation System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


df = pd.read_csv("final_recipe.csv")
nutrition_features = ['calories', 'protein', 'fat', 'carbohydrates']
df[nutrition_features] = df[nutrition_features].fillna(df[nutrition_features].mean())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[nutrition_features])
kmeans = KMeans(n_clusters=8, random_state=42)
df['nutrition_cluster'] = kmeans.fit_predict(X_scaled)


nonveg_keywords = ['chicken','mutton','egg','fish','pork','beef','bacon','meat','turkey','lamb','shrimp']
df["veg_nonveg"] = df.apply(
    lambda x: "Non-Veg" if any(word in str(x["recipe_name"]).lower() + " " + str(x["ingredients_list"]).lower() for word in nonveg_keywords) else "Veg",
    axis=1
)


cuisine_keywords = {
    'Indian': ['paneer','masala','biryani','dal','curry','roti','paratha','sambar','tikka','idli','dosa'],
    'Chinese': ['noodle','fried rice','manchurian','schezwan','spring roll','momo'],
    'Italian': ['pizza','pasta','lasagna','spaghetti','mozzarella','garlic bread'],
    'Mexican': ['taco','burrito','nacho','quesadilla','salsa','tortilla'],
    'American': ['burger','fries','sandwich','steak','pancake','donut'],
    'Japanese': ['sushi','ramen','tempura','miso','teriyaki'],
    'French': ['croissant','souffle','crepe','baguette','ratatouille'],
    'Thai': ['pad thai','green curry','tom yum','lemongrass'],
    'Korean': ['kimchi','bibimbap','bulgogi','kimbap','gochujang']
}

def detect_cuisine(name, ingredients):
    text = str(name).lower() + " " + str(ingredients).lower()
    best_match, best_score = None, 0
    for cuisine, keywords in cuisine_keywords.items():
        for word in keywords:
            score = fuzz.partial_ratio(word, text)
            if score > best_score:
                best_match, best_score = cuisine, score
    return best_match if best_score >= 80 else 'Other'

df["cuisine_type"] = df.apply(lambda x: detect_cuisine(x["recipe_name"], x["ingredients_list"]), axis=1)

region_keywords = {
    'North Indian': ['paneer','butter chicken','naan','dal makhani','paratha'],
    'South Indian': ['idli','dosa','sambar','rasam','pongal'],
    'East Indian': ['momo','machher jhol','rasgulla','litti chokha'],
    'West Indian': ['dhokla','thepla','vada pav','pav bhaji','poha']
}

def detect_region(name, ingredients):
    text = str(name).lower() + " " + str(ingredients).lower()
    best_match, best_score = None, 0
    for region, keywords in region_keywords.items():
        for word in keywords:
            score = fuzz.partial_ratio(word, text)
            if score > best_score:
                best_match, best_score = region, score
    return best_match if best_score >= 80 else 'Other Region'

df["region_type"] = df.apply(lambda x: detect_region(x["recipe_name"], x["ingredients_list"]), axis=1)


df["ingredients_list"] = df["ingredients_list"].astype(str)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["ingredients_list"])


def extract_main_dishes(df, min_freq=2):
    stopwords = {"with", "and", "in", "the", "recipe", "style", "dish", "homemade", "easy"}
    all_words = []
    for name in df['recipe_name'].str.lower():
        words = re.findall(r'\b\w+\b', name)
        words = [w for w in words if w not in stopwords]
        all_words.extend(words)
    word_counts = Counter(all_words)
    main_dishes = [word for word, count in word_counts.items() if count >= min_freq]
    return main_dishes

DYNAMIC_MAIN_DISHES = extract_main_dishes(df)


@app.get("/")
def home(): 
    return {"message":"🍴 Food Recommendation API is running!"}

@app.get("/recommend")
def recommend_food_dynamic(food_name: Optional[str]=Query(None),
                            veg_nonveg: Optional[str]=Query(None),
                            cuisine_type: Optional[str]=Query(None),
                            region_type: Optional[str]=Query(None)):
    try:
        df_filtered = df.copy()
        searched_recipe = None
        main_dish = None
        secondary_words = []

        if food_name:
            food_name_lower = food_name.lower()
            words = food_name_lower.split()

            for word in words:
                if word in DYNAMIC_MAIN_DISHES:
                    main_dish = word
                    break
            secondary_words = [w for w in words if w != main_dish] if main_dish else words

            match_result = process.extractOne(food_name_lower, df['recipe_name'].str.lower().tolist())
            if match_result:
                best_match = match_result[0]
                searched_idx = df.index[df["recipe_name"].str.lower() == best_match][0]
                searched_recipe = df.loc[searched_idx]
            else:
                return {"message": f"No recipe found similar to '{food_name}'."}

        if veg_nonveg:
            df_filtered = df_filtered[df_filtered["veg_nonveg"].str.lower() == veg_nonveg.lower()]
        if cuisine_type:
            df_filtered = df_filtered[df_filtered["cuisine_type"].str.lower() == cuisine_type.lower()]
        if region_type:
            df_filtered = df_filtered[df_filtered["region_type"].str.lower() == region_type.lower()]

        if df_filtered.empty:
            df_filtered = df.copy()

        if food_name and searched_recipe is not None:
            tfidf_filtered = tfidf.transform(df_filtered["ingredients_list"])
            cosine_sim = linear_kernel(tfidf_matrix[searched_idx], tfidf_filtered).flatten()
            df_filtered = df_filtered.copy()
            df_filtered["similarity"] = cosine_sim

            if main_dish:
                pattern = rf'\b{re.escape(main_dish)}\b'
                df_filtered["similarity"] += df_filtered["recipe_name"].str.lower().apply(
                    lambda x: 3.0 if re.search(pattern, x) else 0
                )

            for word in secondary_words:
                pattern = rf'\b{re.escape(word)}\b'
                df_filtered["similarity"] += df_filtered["recipe_name"].str.lower().apply(
                    lambda x: 1.0 if re.search(pattern, x) else 0
                )

            if main_dish in ["dosa","idli"]:
                df_filtered["similarity"] += df_filtered["cuisine_type"].str.lower().apply(
                    lambda x: 0.5 if x == "south indian" else 0
                )

            df_filtered = df_filtered.sort_values(by="similarity", ascending=False)

            if searched_recipe['recipe_id'] not in df_filtered['recipe_id'].values:
                df_filtered = pd.concat([searched_recipe.to_frame().T, df_filtered], ignore_index=True)

            results = df_filtered.head(7)
        else:
            results = df_filtered.sample(min(6, len(df_filtered)))

        return results[["recipe_name","calories","protein","fat","carbohydrates",
                        "veg_nonveg","cuisine_type","region_type","image_url"]].to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}




def nutrition_recommendations(calories: Optional[float]=None, protein: Optional[float]=None, fat: Optional[float]=None, carbs: Optional[float]=None, top_n: int=5):
    input_values = np.array([[calories if calories else 0, 
                              protein if protein else 0, 
                              fat if fat else 0, 
                              carbs if carbs else 0]])
    input_scaled = scaler.transform(input_values)
    distances = euclidean_distances(input_scaled, X_scaled).flatten()
    similar_indices = np.argsort(distances)[:top_n]
    return df.iloc[similar_indices][["recipe_name","calories","protein","fat","carbohydrates"]].to_dict(orient="records")

@app.get("/recommend_by_nutrition")
def recommend_by_nutrition_endpoint(calories: Optional[float]=Query(None),
                                    protein: Optional[float]=Query(None),
                                    fat: Optional[float]=Query(None),
                                    carbs: Optional[float]=Query(None),
                                    top_n: int = 5):
    try:
        return nutrition_recommendations(calories, protein, fat, carbs, top_n)
    except Exception as e:
        return {"error": str(e)}

def recommend_pair(food_name: str, top_n: int = 5):
    best_match, _, _ = process.extractOne(food_name.lower(), df['recipe_name'].str.lower().tolist())
    if not best_match:
        return {"error": f"No close match found for '{food_name}'"}
    base = df[df['recipe_name'].str.lower()==best_match]
    cuisine, region, vegtype = base['cuisine_type'].values[0], base['region_type'].values[0], base['veg_nonveg'].values[0]
    df_pair = df[(df['cuisine_type']==cuisine) & (df['region_type']==region) & (df['veg_nonveg']==vegtype)]
    if df_pair.empty: df_pair = df.sample(min(top_n,len(df)))
    else: df_pair = df_pair.sample(min(top_n,len(df_pair)))
    return df_pair[["recipe_name","cuisine_type","region_type","veg_nonveg","aver_rate","image_url"]].to_dict(orient='records')

@app.get("/recommend_pair")
def recommend_pair_endpoint(food_name: str):
    try:
        return recommend_pair(food_name)
    except Exception as e:
        return {"error": str(e)}


@app.get("/top_dishes")
def top_dishes(region_type: Optional[str]=Query(None), cuisine_type: Optional[str]=Query(None), top_n: int=5):
    try:
        df_filtered = df.copy()
        if region_type:
            df_filtered = df_filtered[df_filtered["region_type"].str.lower() == region_type.lower()]
        if cuisine_type:
            df_filtered = df_filtered[df_filtered["cuisine_type"].str.lower() == cuisine_type.lower()]
        if df_filtered.empty:
            return {"message":"No dishes found for given filters."}
        df_filtered = df_filtered.sort_values(by="aver_rate", ascending=False)
        return df_filtered.head(top_n)[["recipe_name","veg_nonveg","cuisine_type","region_type","aver_rate","image_url"]].to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

if __name__=="__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT",8000)))
