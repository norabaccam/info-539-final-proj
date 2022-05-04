'''
Code moved from Google CoLab.
Fancy model file.
'''

#@title Load the Universal Sentence Encoder's TF Hub module
from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt     
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from pathlib import Path  
import glob
from sklearn.metrics.pairwise import linear_kernel

module_url = "http://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
    return model(input)

directory_path = "/Users/norabaccam/Documents/ista_439/final_proj/recipes2/"
files = glob.glob(f"{directory_path}/*.txt")
titles = [Path(text).stem for text in files]
messages = []

for title in titles:
    path = directory_path + title + ".txt"
    recipe = open(path)
    title = title.replace("_", " ")
    recipe_txt = title + "\n" + Path(path).read_text()
    messages.append(recipe_txt)

# Reduce logging output.
logging.set_verbosity(logging.ERROR)

message_embeddings = embed(messages)

for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
  print("Recipe: {}".format(messages[i]))
  print("Embedding size: {}".format(len(message_embedding)))
  message_embedding_snippet = ", ".join(
      (str(x) for x in message_embedding[:3]))
  print("Embedding: [{}, ...]\n".format(message_embedding_snippet))


def create_recipes_lst(): 
    recipes = []
    for i in range(message_embeddings.shape[0]):
        recipe_dct = {}

        cosine_similarities = linear_kernel(message_embeddings[i:i+1], message_embeddings).flatten()
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

'''
Figure code provided on Google Colab USE model.
'''
def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  sns.set(font_scale=1.2)
  labels = [lab.split("\n")[0] for lab in labels]
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")

def run_and_plot(messages_):
  messages_ = messages_
  message_embeddings_ = embed(messages_)
  plot_similarity(messages_, message_embeddings_, 90)

def get_precision(recipes):
    count = 1
    for recipe in recipes:
        curr_recipe = recipe["recipe_name"]
        print("Current recipe " + str(count) + ":", curr_recipe)
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
        count += 1
        print()
        print("NEXT RECIPE....")
        print()

r = create_recipes_lst()
print(r)
get_precision(r)
print(r)
