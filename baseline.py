from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pathlib import Path  
import glob
from sklearn.metrics.pairwise import linear_kernel

directory_path = "/Users/norabaccam/Documents/ista_439/final_proj/recipes/"
files = glob.glob(f"{directory_path}/*.txt")
titles = [Path(text).stem for text in files]

def create_recipes_lst(): 
    tfidf_vectorizer = TfidfVectorizer(input='filename', stop_words='english')
    tfidf_vector = tfidf_vectorizer.fit_transform(files)

    recipes = []

    for i in range(tfidf_vector.shape[0]):
        recipe_dct = {}

        cosine_similarities = linear_kernel(tfidf_vector[i:i+1], tfidf_vector).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-5:-1]

        name_lst = []
        for index in related_docs_indices:
            name_lst.append(titles[index])

        cosine_vals = cosine_similarities[related_docs_indices][1:]
        recipe_dct["recipe_name"] = name_lst[0]
        recipe_dct["recipe_matches"] = name_lst[1:]
        recipe_dct["cosine_similarities"] = cosine_vals

        recipes.append(recipe_dct)
    return recipes

def get_precision(recipes):
    for recipe in recipes:
        curr_recipe = recipe["recipe_name"]
        print("Current recipe:", curr_recipe)
        r_file = open(directory_path+ curr_recipe + ".txt")
        for line in r_file:
            print(line, end="")
        
        recipe["precision_at_3"] = 0.0
        recipe["precision_at_3_lst"] = []
        for compare in recipe["recipe_matches"]:
            print("-" * 40)
            print("Comparison recipe:", compare)
            c_file = open(directory_path+ compare + ".txt")
            for line in c_file:
                print(line, end="")
            print()
            relevant = input("Is this recipe relevant? (y/n): ").lower()
            if relevant == "y":
                recipe["precision_at_3"] += 1
                recipe["precision_at_3_lst"].append("y")
            else:
                recipe["precision_at_3_lst"].append("n")
        recipe["precision_at_3"] = recipe["precision_at_3"] / 3

        print()
        print("NEXT RECIPE....")
        print()


def main():
    recipes = create_recipes_lst()
    print(recipes)
    get_precision(recipes)
    print(recipes)

main()

