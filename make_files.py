import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re

df = pd.read_csv("recipes.csv")

for i in range(len(df)):
    title = df["title"][i]
    ingredients = df["ingredients"][i]
    ing_string = re.sub(r"[^a-zA-Z0-9 ]","", ingredients)
    fname = re.sub('[^A-Za-z ]+', '', title).lower()
    fname = fname.replace(" ", "_")
    path = "/Users/norabaccam/Documents/ista_439/final_proj/recipes2/" + fname + ".txt"
    f = open(path, "w")
    lst = [step.strip(" \"',").replace("\u00b0", "") for step in df["directions"][i][1:-1].split('."') if step]
    f.write(ing_string + "\n")
    for step in lst:
        f.write(step + "\n")
    f.close()
