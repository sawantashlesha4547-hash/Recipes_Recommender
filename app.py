from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import re

app = Flask(__name__)

# ==========================================================================
# 1. INITIALIZE DATASET & ARTIFACTS
# ==========================================================================
try:
    recipe_df = pd.read_csv("recipes.csv")
    # Standardize spaces and sanitize text columns to prevent key errors
    recipe_df.columns = recipe_df.columns.str.strip()
    recipe_df.rename(columns={
        'Fats (g)': 'Fats',
        'Carbs (g)': 'Carbs',
        'Protein (g)': 'Protein',
        'Cholesterol (mg)': 'Cholesterol',
        'Fiber (g)': 'Fiber'
    }, inplace=True)
    
    # Fill structural missing strings or links safely
    recipe_df['Ingredients'] = recipe_df['Ingredients'].fillna('').astype(str)
    recipe_df['Image_URL'] = recipe_df['Image_URL'].fillna('').astype(str)

    # Load ML artifacts securely
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model.pkl", "rb") as f:
        knn = pickle.load(f)
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to load data matrices or serialized models: {e}")
    recipe_df = pd.DataFrame()

# ==========================================================================
# 2. HYBRID MATCHING & RECOMMENDATION ENGINE
# ==========================================================================
def recommend_recipes(input_features, max_results=10):
    if recipe_df.empty:
        return pd.DataFrame()

    # Unpack properties explicitly
    fats, carbs, protein, cholesterol, fiber, ingredients_text = input_features
    ingredients_cleaned = str(ingredients_text).lower().strip()
    
    # --- PHASE 1: EXACT MATCH STRING SCORING ---
    if ingredients_cleaned:
        # Tokenize user text safely using regex to catch spaces and commas
        user_ingredients = [ing.strip() for ing in re.split(r'[\s,]+', ingredients_cleaned) if ing.strip()]
        
        def count_matches(recipe_ings):
            recipe_ings_lower = str(recipe_ings).lower()
            return sum(1 for ui in user_ingredients if ui in recipe_ings_lower)
            
        recipe_df['match_count'] = recipe_df['Ingredients'].apply(count_matches)
    else:
        recipe_df['match_count'] = 0

    # --- PHASE 2: ML VECTOR NEIGHBORHOOD SPACE FILLING ---
    try:
        # Create numerical array matching what scaler expects
        num_features = np.array([[float(fats), float(carbs), float(protein), float(cholesterol), float(fiber)]])
        input_features_scaled = scaler.transform(num_features)
        
        # Vectorize incoming raw text string
        input_ingredients_transformed = vectorizer.transform([ingredients_cleaned])
        
        # Fuse spaces horizontally across arrays
        input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])
        
        # Pull distance distributions across index map for all recipes
        distances, indices = knn.kneighbors(input_combined, n_neighbors=len(recipe_df))
        knn_recommendations = recipe_df.iloc[indices[0]].copy()
        knn_recommendations['knn_distance'] = distances[0]
    except Exception as e:
        print(f"[ML PIPELINE ERROR] Matrix transformation or fallback lookup failed: {e}")
        knn_recommendations = recipe_df.copy()
        knn_recommendations['knn_distance'] = 0

    # --- PHASE 3: COMPOSITING & CLEANUP ---
    # Sort primarily by match_count (desc), then by knn_distance (asc)
    recommendations = knn_recommendations.sort_values(by=['match_count', 'knn_distance'], ascending=[False, True])

    # Prune unwanted tracking variables and output target subset safely
    cols = ['Dish', 'Fats', 'Carbs', 'Protein', 'Cholesterol', 'Fiber', 'Ingredients', 'Image_URL']
    existing_cols = [c for c in cols if c in recommendations.columns]
    
    return recommendations[existing_cols].head(max_results)

# ==========================================================================
# 3. INTERACTIVE FLASK ENDPOINTS
# ==========================================================================
@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    query = ""
    
    if request.method == "POST":
        ingredients = request.form.get("ingredients", "").strip()
        
        # Internal helper block preventing runtime calculation crashing from empty fields
        def parse_input(field_name):
            val = request.form.get(field_name, "").strip()
            return float(val) if val else 0.0

        fats = parse_input("fats")
        carbs = parse_input("carbs")
        protein = parse_input("protein")
        cholesterol = parse_input("cholesterol")
        fiber = parse_input("fiber")

        # Bundle characteristics cleanly
        input_features = [fats, carbs, protein, cholesterol, fiber, ingredients]
        
        # Execute processing pipeline
        results_df = recommend_recipes(input_features, max_results=10)
        
        if not results_df.empty:
            results = results_df.to_dict(orient='records')
        query = ingredients

    return render_template("index.html", results=results, query=query)

if __name__ == "__main__":
    app.run(debug=True)