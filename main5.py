import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import uvicorn

# --- Data Loading and Setup ---

app = FastAPI(title="Food Recommendation System (Precision Enhanced)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data (Assuming 'recipe.csv' exists and has required columns)
try:
    df = pd.read_csv("recipe.csv")
    df['ingredients_list'] = df['ingredients_list'].astype(str)
    # Ensure all columns used for math operations are numeric and handle NaNs
    for col in ['calories', 'protein', 'fat', 'carbohydrates', 'aver_rate']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
except FileNotFoundError:
    raise RuntimeError("recipe.csv not found. Please ensure the data file is present.")
except KeyError as e:
    raise RuntimeError(f"Missing required column in recipe.csv: {e}")

# --- TF-IDF Setup ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['ingredients_list'])


# --- Classification Logic (Enhanced) ---

# 1. Veg/Non-Veg Classification
nonveg_keywords = ['chicken','mutton','egg','fish','pork','beef','bacon','meat','turkey','lamb','shrimp','prawn','crab','oyster']

def get_veg_nonveg(name, ingredients):
    text = str(name).lower() + " " + str(ingredients).lower()
    return "Non-Veg" if any(word in text for word in nonveg_keywords) else "Veg"

df['veg_nonveg'] = df.apply(lambda x: get_veg_nonveg(x['recipe_name'], x['ingredients_list']), axis=1)


# 2. Cuisine Classification (Expanded Keywords for Better Coverage)
cuisine_keywords = {
    'Indian': ['paneer','masala','biryani','dal','curry','roti','paratha','sambar','tikka','idli','dosa','naan','tandoori','pakora','jalebi'],
    'Chinese': ['noodle','fried rice','manchurian','schezwan','spring roll','momo','wok','dim sum','chop suey','bao'],
    'Italian': ['pizza','pasta','lasagna','spaghetti','mozzarella','garlic bread','pesto','risotto','tiramisu','gelato'],
    'Mexican': ['taco','burrito','nacho','quesadilla','salsa','tortilla','chili','enchilada','guacamole','fajita'],
    'American': ['burger','fries','sandwich','steak','pancake','donut','mac and cheese','bbq','coleslaw','waffle'],
    'Japanese': ['sushi','ramen','tempura','miso','teriyaki','sashimi','udon','wasabi','edamame'],
    'French': ['croissant','souffle','crepe','baguette','ratatouille','quiche','escargot','tartar','biscotti'],
    'Thai': ['pad thai','green curry','tom yum','lemongrass','red curry','coconut milk','satay'],
    'Korean': ['kimchi','bibimbap','bulgogi','kimbap','gochujang','galbi','tteokbokki'],
    'Mediterranean': ['falafel', 'hummus', 'pita', 'kebab', 'tahini', 'gyros', 'tabbouleh'],
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

df['cuisine_type'] = df.apply(lambda x: detect_cuisine_fuzzy(x['recipe_name'], x['ingredients_list']), axis=1)


# 3. Region Classification (Expanded Indian Regional Keywords)
region_keywords = {
    'North Indian (Punjabi)': ['paneer','butter chicken','naan','dal makhani','paratha','chhole','rajma','tandoori'],
    'South Indian (Dravidian)': ['idli','dosa','sambar','rasam','pongal','vada','uttapam','upma','appam'],
    'East Indian (Bengali/Odia)': ['momo','machher jhol','rasgulla','litti chokha','mishti','sandesh','paturi'],
    'West Indian (Gujarati/Maharashtrian)': ['dhokla','thepla','vada pav','pav bhaji','poha','fafda','khandvi','misal pav']
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

df['region_type'] = df.apply(lambda x: detect_region_rapid(x['recipe_name'], x['ingredients_list']), axis=1)


# 4. Recipe Type Classification (Significantly Expanded Keywords for Breakfast/Dessert)
recipe_type_keywords = {
    'Dessert': ['cake', 'cookie', 'brownie', 'ice cream', 'pudding', 'pie', 'tart', 'mousse', 'cheesecake', 'sorbet', 'macaron', 'custard', 'flan', 'cupcake', 'sundae'],
    'Breakfast': ['pancake', 'waffle', 'omelette', 'scrambled egg', 'smoothie', 'oats', 'muffin', 'granola', 'cereal', 'frittata', 'toast', 'bagel', 'eggs benedict'],
    'Soup': ['soup', 'broth', 'stew', 'chowder', 'bisque', 'gazpacho'],
    'Main Course': ['curry', 'steak', 'roast', 'casserole', 'biryani', 'masala', 'pasta', 'lasagna', 'tuna', 'chop', 'loaf', 'risotto'],
    'Snack': ['sandwich', 'fries', 'taco', 'momo', 'roll', 'burger', 'chips', 'dip', 'nachos', 'wings', 'samosa']
}

def detect_recipe_type(name, ingredients, threshold=75):
    text = str(name).lower() + " " + str(ingredients).lower()
    best_match, best_score = 'Main Course', 0 # Default to Main Course
    
    for recipe_type, keywords in recipe_type_keywords.items():
        for word in keywords:
            # Use fuzz.token_sort_ratio for slightly better phrase matching
            score = fuzz.token_sort_ratio(word, text) 
            if score > best_score:
                best_match, best_score = recipe_type, score
    
    # If the score is very low, stick to the default 'Main Course'
    return best_match if best_score >= threshold else 'Main Course'

df['recipe_type'] = df.apply(lambda x: detect_recipe_type(x['recipe_name'], x['ingredients_list']), axis=1)

# Ensure recipe names are lowercased for robust searching later
df['recipe_name_lower'] = df['recipe_name'].str.lower()


# --- Recommendation Functions ---

def recommend_food(input_value, top_n=10, by='name'):
    """
    Recommends recipes based on similarity (content-based).
    Precision Enhanced: Sorts by cosine similarity AND recipe name for better clustering of similar dishes (e.g., varieties of 'Dosa').
    """
    if by == 'name':
        all_recipes = df['recipe_name_lower'].tolist()
        best_match, score, _ = process.extractOne(input_value.lower(), all_recipes)
        
        if score < 70:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                                detail=f"No close match found for '{input_value}'. Best match score was {score:.1f}%.")
            
        idx = df[df['recipe_name_lower'] == best_match].index[0]
    elif by == 'url':
        idx = df[df['image_url'] == input_value].index[0]
    else:
        return pd.DataFrame()

    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Create a temporary DataFrame to sort by similarity and recipe name
    temp_df = df.copy()
    temp_df['similarity'] = cosine_sim
    
    # Sort primarily by similarity (descending) and secondarily by recipe name (ascending)
    temp_df = temp_df.sort_values(by=['similarity', 'recipe_name'], ascending=[False, True])
    
    # Return top N results (including the search term itself)
    return temp_df.head(top_n)[['recipe_id','recipe_name','image_url','cuisine_type','region_type','veg_nonveg','aver_rate', 'recipe_type']]

def recommend_by_nutrition(calories=None, protein=None, fat=None, carbs=None, top_n=10):
    """
    Recommends recipes based on matching nutritional targets.
    Precision Enhanced: Prioritizes perfect matches and only scores based on provided metrics.
    """
    df_copy = df.copy()
    
    # Initialize score to zero
    df_copy['nutrition_score'] = 0.0
    metrics_provided = 0

    # Only calculate score for metrics provided by the user
    if calories is not None:
        df_copy['nutrition_score'] += abs(df_copy['calories'] - calories)
        metrics_provided += 1
        
    if protein is not None:
        df_copy['nutrition_score'] += abs(df_copy['protein'] - protein)
        metrics_provided += 1
        
    if fat is not None:
        df_copy['nutrition_score'] += abs(df_copy['fat'] - fat)
        metrics_provided += 1
        
    if carbs is not None:
        df_copy['nutrition_score'] += abs(df_copy['carbohydrates'] - carbs)
        metrics_provided += 1

    if metrics_provided == 0:
        # Fallback: If no metrics provided, just sort by rating
        return df_copy.sort_values('aver_rate', ascending=False).head(top_n)[
            ['recipe_name','calories','protein','fat','carbohydrates','cuisine_type','region_type','image_url', 'recipe_type']
        ]
        
    # Sort by the combined absolute difference (lower score is better)
    # The lowest possible score is 0.0 (perfect match)
    return df_copy.sort_values('nutrition_score').head(top_n)[
        ['recipe_name','calories','protein','fat','carbohydrates','cuisine_type','region_type','image_url', 'recipe_type']
    ]

# Pairing Rules: Increased inputs for better coverage
pairing_rules = {
    'burger': ['fries','coleslaw','onion rings'],
    'pizza': ['garlic bread','caesar salad','soda'],
    'biryani': ['raita','mirchi ka salan','gulab jamun'],
    'pasta': ['meatballs','red wine','tiramisu'],
    'steak': ['mashed potatoes','asparagus','cabernet sauvignon'],
    'taco': ['salsa','guacamole','margarita'],
    'curry': ['rice','naan','samosa'],
}

def recommend_pair(food_name, top_n=10): 
    """
    Provides food pairing suggestions based on keyword rules and cuisine matching.
    Precision Enhanced: Samples popular recipes from the same cuisine when no rule match is found.
    """
    
    all_recipes = df['recipe_name_lower'].tolist()
    best_match_name, score, _ = process.extractOne(food_name.lower(), all_recipes)

    if score < 70:
        # Fallback to general recommendations if the food name isn't recognized
        return df.sample(top_n)[['recipe_name','cuisine_type','region_type','veg_nonveg','aver_rate','image_url', 'recipe_type']]

    text = best_match_name.lower()
    possible_pairs = []
    
    # 1. Check direct keyword pairing rules
    for k, v in pairing_rules.items():
        if k in text:
            possible_pairs.extend(v)
            break
            
    # 2. If no rule match, recommend other recipes from the same cuisine
    if not possible_pairs:
        # Find the cuisine of the best-matched recipe
        cuisine = df.loc[df['recipe_name_lower'] == best_match_name, 'cuisine_type'].iloc[0]
        
        # Ensure we only sample popular recipes that are NOT the base recipe
        same_cuisine_recipes = df[
            (df['cuisine_type'] == cuisine) & 
            (df['recipe_name_lower'] != best_match_name)
        ].sort_values(by='aver_rate', ascending=False) # Sort by rating for popularity
        
        if not same_cuisine_recipes.empty:
            # Sample only from the most popular recipes
            sampled_recipes = same_cuisine_recipes['recipe_name'].head(min(50, len(same_cuisine_recipes))).sample(min(top_n, len(same_cuisine_recipes)), replace=False).tolist()
            possible_pairs.extend(sampled_recipes)
        
    
    # 3. Filter the main DataFrame to get the details of the paired recipes
    pair_df_list = []
    unique_pairs = list(set([p.lower() for p in possible_pairs]))
    
    for pair_name in unique_pairs:
        match_name, match_score, _ = process.extractOne(pair_name, all_recipes, score_cutoff=85)
        if match_name:
            pair_df_list.append(df[df['recipe_name_lower'] == match_name].iloc[0])
            
    df_pairs = pd.DataFrame(pair_df_list).drop_duplicates(subset=['recipe_name']).head(top_n)

    if df_pairs.empty:
        # Final fallback
        df_pairs = df.sample(top_n)
        
    return df_pairs[['recipe_name','cuisine_type','region_type','veg_nonveg','aver_rate','image_url', 'recipe_type']]

# --- FastAPI Endpoints ---

@app.get("/")
def home():
    return {"message": "🍴 Food Recommendation API is running successfully! Check /docs for API details."}

@app.get("/recommend_by_name", response_model=List[Dict[str, Any]])
def recommend_by_name_endpoint(food_name: str, top_n: int = Query(10, gt=0)):
    """Content-based recommendation by dish name."""
    recommendations = recommend_food(food_name, top_n, by='name')
    return recommendations.to_dict(orient='records')

@app.get("/recommend_by_url", response_model=List[Dict[str, Any]])
def recommend_by_url_endpoint(image_url: str, top_n: int = Query(10, gt=0)):
    """Content-based recommendation by image URL."""
    # Note: Requires 'image_url' column to be fully populated in recipe.csv
    recommendations = recommend_food(image_url, top_n, by='url')
    return recommendations.to_dict(orient='records')

@app.get("/recommend_by_nutrition", response_model=List[Dict[str, Any]])
def recommend_by_nutrition_endpoint(
    calories: Optional[int] = Query(None),
    protein: Optional[int] = Query(None),
    fat: Optional[int] = Query(None),
    carbs: Optional[int] = Query(None),
    top_n: int = Query(10, gt=0)
):
    """Nutrition-based recommendation by macro targets."""
    recommendations = recommend_by_nutrition(calories, protein, fat, carbs, top_n)
    return recommendations.to_dict(orient='records')

@app.get("/recommend_pair", response_model=List[Dict[str, Any]])
def recommend_pair_endpoint(food_name: str, top_n: int = Query(10, gt=0)):
    """Food pairing suggestions based on rules and cuisine type."""
    recommendations = recommend_pair(food_name, top_n)
    return recommendations.to_dict(orient='records')

@app.get("/recommend/filter", response_model=List[Dict[str, Any]])
def filter_recommendations_endpoint(
    cuisine_type: Optional[str] = Query(None, description="Filter by cuisine (e.g., Indian, Italian). Case-insensitive."),
    region_type: Optional[str] = Query(None, description="Filter by regional type (e.g., North Indian, South Indian). Case-insensitive."),
    recipe_type: Optional[str] = Query(None, description="Filter by dish type (e.g., Dessert, Breakfast, Main Course). Case-insensitive."),
    veg_nonveg: Optional[str] = Query(None, description="Filter by 'Veg' or 'Non-Veg'. Case-insensitive."),
    top_n: int = Query(10, gt=0, description="Number of results to return."),
    sort_by_rating: bool = Query(True, description="If true, sort results by average rating (highest first).")
):
    """
    **Unified Filtering Endpoint**
    Filters recipes based on Cuisine, Region, Recipe Type, and Veg/Non-veg status.
    """
    df_filtered = df.copy()

    # Apply filters (Case-insensitive matching)
    if cuisine_type:
        df_filtered = df_filtered[df_filtered['cuisine_type'].str.lower() == cuisine_type.lower()]
    
    if region_type:
        df_filtered = df_filtered[df_filtered['region_type'].str.lower() == region_type.lower()]

    if recipe_type:
        df_filtered = df_filtered[df_filtered['recipe_type'].str.lower() == recipe_type.lower()]
        
    if veg_nonveg:
        df_filtered = df_filtered[df_filtered['veg_nonveg'].str.lower() == veg_nonveg.lower()]

    if df_filtered.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail="No recipes found matching the specified filters.")

    # Sort results
    if sort_by_rating:
        df_filtered = df_filtered.sort_values(by='aver_rate', ascending=False)

    # Return top N results
    return df_filtered.head(top_n)[['recipe_name','image_url','cuisine_type','region_type','veg_nonveg','aver_rate', 'recipe_type']].to_dict(orient='records')


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
