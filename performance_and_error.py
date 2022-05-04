from results_baseline import *
from results_fancy import *

import random
random.seed(8)

# error analysis TO-DO:
# 1. sample 50-100 samples
# 2. categorize them 
directory_path = "/Users/norabaccam/Documents/ista_439/final_proj/recipes2/"

def error_at_3():
    random.shuffle(fancy)
    for recipe in fancy[:100]:
        if recipe['precision_at_3'] == 0.0:
            curr_recipe = recipe["recipe_name"]
            print("Current recipe:", curr_recipe)
            r_file = open(directory_path+ curr_recipe + ".txt")
            for line in r_file:
                print(line, end="")
            
            for compare in recipe["recipe_matches"]:
                print("-" * 40)
                print("Comparison recipe:", compare)
                c_file = open(directory_path+ compare + ".txt")
                for line in c_file:
                    print(line, end="")
                print()
            print()
            print("NEXT RECIPE....")
            print()

def error_at_1():
    random.shuffle(fancy)
    for recipe in fancy[:100]:
        if recipe['precision_at_3_lst'][0] == 'n':
            curr_recipe = recipe["recipe_name"]
            print("Current recipe:", curr_recipe)
            r_file = open(directory_path+ curr_recipe + ".txt")
            for line in r_file:
                print(line, end="")
            
            for compare in recipe["recipe_matches"]:
                print("-" * 40)
                print("Comparison recipe:", compare)
                c_file = open(directory_path+ compare + ".txt")
                for line in c_file:
                    print(line, end="")
                print()
                break
            print()
            print("NEXT RECIPE....")
            print()

def avg_precision_1(model_results):
    total = 0
    for recipe in model_results:
        if recipe["precision_at_3_lst"][0] == "y":
            total += 1
    return total / len(model_results)

def avg_precision_3(model_results):
    total = 0
    for recipe in model_results:
        total += recipe["precision_at_3"]
    return total / len(model_results)

# PRINT RESULTS
def print_results():
    baseline_p1 = avg_precision_1(baseline_results)
    print(f"Average Precision@1 score for baseline model: {round(baseline_p1, 2)}")
    baseline_p3 = avg_precision_3(baseline_results)
    print(f"Average Precision@3 score for baseline model: {round(baseline_p3, 2)}")

    print()

    fancy_p1 = avg_precision_1(fancy)
    print(f"Average Precision@1 score for fancy model: {round(fancy_p1, 2)}")
    fancy_p3 = avg_precision_3(fancy)
    print(f"Average Precision@3 score for fancy model: {round(fancy_p3, 2)}")

print_results()