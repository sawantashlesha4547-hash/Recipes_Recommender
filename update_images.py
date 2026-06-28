import pandas as pd
import urllib.parse

def update_images():
    df = pd.read_csv("recipes.csv")
    
    # Generate loremflickr URLs based on the dish name to get unique images for each dish
    def get_image_url(dish_name):
        safe_name = urllib.parse.quote(dish_name.replace(" ", "").lower())
        return f"https://loremflickr.com/600/400/indian,food,{safe_name}/all"
        
    df['Image_URL'] = df['Dish'].apply(get_image_url)
    
    df.to_csv("recipes.csv", index=False)
    print("Updated recipes.csv with working image links.")

if __name__ == "__main__":
    update_images()
