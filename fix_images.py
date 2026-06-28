import pandas as pd

def fix_images():
    # 10 Beautiful, safe, hand-picked food images from Unsplash
    safe_images = [
        "https://images.unsplash.com/photo-1585937421612-70a008356fbe?q=80&w=600&auto=format&fit=crop", # Curry
        "https://images.unsplash.com/photo-1606491956689-2ea866880c84?q=80&w=600&auto=format&fit=crop", # Dish
        "https://images.unsplash.com/photo-1601050690597-df0568f70950?q=80&w=600&auto=format&fit=crop", # Samosa / Pastry
        "https://images.unsplash.com/photo-1565557623262-b51c2513a641?q=80&w=600&auto=format&fit=crop", # Tikka
        "https://images.unsplash.com/photo-1589301760014-d929f39ce9b1?q=80&w=600&auto=format&fit=crop", # Naan
        "https://images.unsplash.com/photo-1631452180519-c014fe946bc0?q=80&w=600&auto=format&fit=crop", # Biryani
        "https://images.unsplash.com/photo-1626777552726-4a6b54c97e46?q=80&w=600&auto=format&fit=crop", # Spices
        "https://images.unsplash.com/photo-1617692855027-33b14f061079?q=80&w=600&auto=format&fit=crop", # Thali
        "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?q=80&w=600&auto=format&fit=crop", # Bowl
        "https://images.unsplash.com/photo-1603894584373-5ac82bce8d61?q=80&w=600&auto=format&fit=crop"  # Platter
    ]
    
    df = pd.read_csv("recipes.csv")
    
    # Apply them cyclically
    df['Image_URL'] = [safe_images[i % len(safe_images)] for i in range(len(df))]
    
    # Hardcode a specific one for egg curry just in case
    df.loc[df['Dish'] == 'Egg Curry', 'Image_URL'] = "https://images.unsplash.com/photo-1585937421612-70a008356fbe?q=80&w=600&auto=format&fit=crop"
    
    df.to_csv("recipes.csv", index=False)
    print("Fixed all images to safe curated Unsplash food photography.")

if __name__ == "__main__":
    fix_images()
