import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def train_and_save_models():
    # 1. Load the original dataset
    df = pd.read_csv("recipe_final (1).csv")
    
    # 2. Rename columns to match what app.py expects
    df.rename(columns={
        'Fats (g)': 'Fats',
        'Carbs (g)': 'Carbs',
        'Protein (g)': 'Protein',
        'Cholesterol (mg)': 'Cholesterol',
        'Fiber (g)': 'Fiber'
    }, inplace=True)
    
    # Strip any whitespace from column names just in case
    df.columns = df.columns.str.strip()
    
    # 3. Save the cleaned dataset as recipes.csv
    df.to_csv("recipes.csv", index=False)
    print("Saved cleaned dataset to recipes.csv")
    
    # 4. Prepare features
    # Numeric features
    numeric_cols = ['Fats', 'Carbs', 'Protein', 'Cholesterol', 'Fiber']
    
    # Fill missing values if any
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df['Ingredients'] = df['Ingredients'].fillna('')
    
    # Train StandardScaler
    scaler = StandardScaler()
    scaled_numeric_features = scaler.fit_transform(df[numeric_cols])
    
    # Train TfidfVectorizer on Ingredients
    vectorizer = TfidfVectorizer(stop_words='english')
    ingredient_features = vectorizer.fit_transform(df['Ingredients'])
    
    # Combine features for KNN
    combined_features = np.hstack([scaled_numeric_features, ingredient_features.toarray()])
    
    # Train NearestNeighbors model
    knn = NearestNeighbors(n_neighbors=df.shape[0], metric='cosine')
    knn.fit(combined_features)
    
    # 5. Save the models
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    with open("model.pkl", "wb") as f:
        pickle.dump(knn, f)
        
    print("Successfully trained and saved vectorizer.pkl, scaler.pkl, and model.pkl")

if __name__ == "__main__":
    train_and_save_models()
