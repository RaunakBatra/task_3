# 🍴 Food Recommendation System API

A Machine Learning-powered **Food Recommendation API** built using **FastAPI**, deployed on **Render**, and trained on custom recipe-nutrition data.  
The system recommends dishes based on:

✅ Similar ingredients  
✅ Veg / Non-Veg filter  
✅ Cuisine type  
✅ Indian regional style  
✅ Nutrition similarity  
✅ Seasonal preferences  
✅ Pairing suggestions  

---

## 🚀 Live Demo

| API Route | URL |
|----------|-----|
| Base URL | https://food-recommendation-api-9off.onrender.com/ |
| Docs (Swagger UI) | https://food-recommendation-api-9off.onrender.com/docs |

---

## 🧠 Features

- 🥗 Content-based recipe recommendations using **TF-IDF + Cosine Similarity**
- 🧮 Nutrition-based matching using **KMeans + Euclidean Distance**
- 🔥 Custom keyword-based cuisine detection (Indian, Italian, Korean, Chinese etc.)
- 🌍 Indian regional classification (North, South, East, West)
- 🥦 Veg / Non-Veg auto-detection
- 🍽️ Ingredient-based pairing recommendations
- ☀️ Seasonal food recommendations
- ⚡ FastAPI + Uvicorn + Scikit-Learn
- ☁️ Deployed on Render

---

## 🧾 API Endpoints

### ✅ 1. Home
```
GET /
```

### ✅ 2. General Food Recommendation
```
GET /recommend?food_name=pizza
```

### ✅ 3. Seasonal Foods
```
GET /recommend_by_season?season=summer
```

### ✅ 4. Nutrition Similarity
```
GET /recommend_by_nutrition?calories=200&protein=15&fat=5&carbs=30
```

### ✅ 5. Pair Recommendations
```
GET /recommend_pair?food_name=butter chicken
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|-----|-------|
| Python | Core logic |
| FastAPI | Web Framework |
| Uvicorn | Server |
| Scikit-Learn | Clustering, Normalization |
| Pandas / NumPy | Data processing |
| TF-IDF | Ingredient vectorization |
| Render | Cloud hosting |

---

## 📦 Installation (Local)

```bash
git clone https://github.com/RaunakBatra/task_3.git
cd task_3
pip install -r requirements.txt
uvicorn app:app --reload
```

Then visit:

```
http://127.0.0.1:8000/docs
```

---

## 📁 Dataset

Custom curated **recipe + nutrition + cuisine classification** dataset.

File: `final_recipe.csv`

---

## 🌍 Deployment

Deployed on **Render** using command:

```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

---

## 📌 Future Enhancements

- 🔐 Auth support (JWT)
- 📷 Image-based food recognition
- 🛍️ Grocery ingredient recommender
- 📱 Frontend app (React / Streamlit UI)

---

## 👨‍💻 Author

**Raunak Batra**  
Food Recommendation ML System • 2025  

📎 *Star ⭐ the repo if you found this useful!*
