from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
from flask_cors import CORS
import ast
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app)

# Load model and supporting files
model = load_model("model.keras")
print("Model input shape:", model.input_shape)
print("Model summary:")
model.summary()

with open("data_cat.pkl", "rb") as f:
    model.classes_ = pickle.load(f)

data = pd.read_csv("dataset.csv")
with open("ratings.json", "r") as f:
    ratings = json.load(f)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def preprocess_image(image_bytes):
    print("Starting image preprocessing")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"Image opened successfully, size: {image.size}")
        
        target_height = model.input_shape[1]
        target_width = model.input_shape[2]
        image = image.resize((target_width, target_height))
        print(f"Image resized to {target_width}x{target_height}")
        
        image_array = np.array(image) / 255.0
        print(f"Image array shape before reshape: {image_array.shape}")
            
        reshaped_array = image_array.reshape((1,) + model.input_shape[1:])
        print(f"Final image array shape: {reshaped_array.shape}")
        
        return reshaped_array
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        raise e

def get_ingredients(product_name):
    print("Searching for product:", product_name)
    
    # Normalize both sides
    data['Product Name'] = data['Product Name'].str.strip().str.lower()
    product_name = product_name.strip().lower()
    
    # Calculate similarity scores for all products
    data['similarity'] = data['Product Name'].apply(lambda x: similar(x, product_name))
    
    # Get the most similar product
    best_match = data.loc[data['similarity'].idxmax()]
    similarity_score = best_match['similarity']
    
    print(f"Best match: {best_match['Product Name']} with similarity: {similarity_score}")
    
    if similarity_score < 0.5:  # Threshold for considering it a match
        print("No good match found")
        return []
    
    try:
        ingredients_str = best_match['Ingredients']
        ingredients_dict = ast.literal_eval(ingredients_str)
        return [(key.strip().lower(), value) for key, value in ingredients_dict.items()]
    except:
        print("Failed to parse ingredients as dictionary")
        return []

def evaluate_ingredients(ingredients):
    good, bad = [], []
    total_weighted_healthiness = 0
    total_weightage = 0

    for ing, percentage in ingredients:
        print(f"Checking ingredient: {ing}")
        # Use fuzzy matching for ratings
        best_match = None
        best_score = 0
        
        for rating_key, rating_value in ratings.items():
            score = similar(ing.lower(), rating_key.lower())
            if score > best_score:
                best_score = score
                best_match = (rating_key, rating_value)
        
        if best_score > 0.7:  # Only consider matches above 70% similarity
            rating_value = best_match[1]
            print(f"Best match for {ing}: {best_match[0]} with rating {rating_value}")
            
            total_weighted_healthiness += percentage * rating_value
            total_weightage += percentage

            if rating_value >= 12:
                good.append(f"{ing} ({percentage}%)")
            else:
                bad.append(f"{ing} ({percentage}%)")
        else:
            print(f"No good match found for {ing}")

    if total_weightage == 0:
        health_percentage = 0
    else:
        healthiness_score = total_weighted_healthiness / total_weightage
        health_percentage = (healthiness_score / 20) * 100

    print(f"Final scores - Total Weighted: {total_weighted_healthiness}, Total Weight: {total_weightage}, Percentage: {health_percentage}")
    return round(health_percentage, 2), good, bad

@app.route("/predict", methods=["POST"])
def predict():
    if "image" in request.files:
        print("Image file received")
        try:
            image = request.files["image"].read()
            print(f"Image size: {len(image)} bytes")
            
            processed_image = preprocess_image(image)
            print("Image preprocessed successfully")
            print(f"Processed image shape: {processed_image.shape}")
            
            prediction = model.predict(processed_image)
            print(f"Model prediction shape: {prediction.shape}")
            class_index = np.argmax(prediction)
            predicted_class = model.classes_[class_index]
            print(f"Raw predicted class: {predicted_class}")
            
            # Clean up predicted class name
            predicted_class = predicted_class.replace('_', ' ').lower()
            print(f"Cleaned predicted class: {predicted_class}")
            
            ingredients = get_ingredients(predicted_class)
            print(f"Found ingredients: {ingredients}")
            
            if not ingredients:
                return jsonify({
                    "error": "Could not identify the food item or find its ingredients. Please try again with a clearer image.",
                    "product": predicted_class
                }), 404
            
        except Exception as e:
            print(f"Error during image processing: {str(e)}")
            return jsonify({"error": str(e)}), 500
    elif "product_name" in request.form:
        product_name = request.form["product_name"]
        ingredients = get_ingredients(product_name)
        predicted_class = product_name
    else:
        return jsonify({"error": "No image or product name provided"}), 400

    if not ingredients:
        return jsonify({"error": "Product not found or no ingredients listed"}), 404

    health_score, good_ingredients, bad_ingredients = evaluate_ingredients(ingredients)
    print(f"Health evaluation complete - score: {health_score}")

    return jsonify({
        "product": predicted_class,
        "ingredients": ingredients,
        "good_ingredients": good_ingredients,
        "bad_ingredients": bad_ingredients,
        "health_score": health_score
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
# from flask import Flask, request, jsonify
# import pickle
# import pandas as pd
# import json
# import numpy as np
# from PIL import Image
# import io
# from tensorflow.keras.models import load_model
# from flask_cors import CORS
# import ast
# from difflib import SequenceMatcher
# import tensorflow as tf

# app = Flask(__name__)
# CORS(app)

# # Load model and supporting files
# model = load_model("model.keras")
# print("Model input shape:", model.input_shape)
# print("Model summary:")
# model.summary()

# with open("data_cat.pkl", "rb") as f:
#     model.classes_ = pickle.load(f)

# data = pd.read_csv("dataset.csv")
# with open("ratings.json", "r") as f:
#     ratings = json.load(f)

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# def preprocess_image(image_bytes):
#     print("Starting image preprocessing")
#     try:
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         print(f"Image opened successfully, size: {image.size}")
        
#         target_height = model.input_shape[1]
#         target_width = model.input_shape[2]
#         image = image.resize((target_width, target_height))
#         print(f"Image resized to {target_width}x{target_height}")
        
#         image_array = np.array(image) / 255.0
#         print(f"Image array shape before reshape: {image_array.shape}")
            
#         reshaped_array = image_array.reshape((1,) + model.input_shape[1:])
#         print(f"Final image array shape: {reshaped_array.shape}")
        
#         return reshaped_array
#     except Exception as e:
#         print(f"Error in image preprocessing: {str(e)}")
#         raise e

# def get_ingredients(product_name):
#     print("Searching for product:", product_name)
    
#     data['Product Name'] = data['Product Name'].str.strip().str.lower()
#     product_name = product_name.strip().lower()
    
#     data['similarity'] = data['Product Name'].apply(lambda x: similar(x, product_name))
    
#     best_match = data.loc[data['similarity'].idxmax()]
#     similarity_score = best_match['similarity']
    
#     print(f"Best match: {best_match['Product Name']} with similarity: {similarity_score}")
    
#     if similarity_score < 0.5:
#         print("No good match found")
#         return []
    
#     try:
#         ingredients_str = best_match['Ingredients']
#         ingredients_dict = ast.literal_eval(ingredients_str)
#         return [(key.strip().lower(), value) for key, value in ingredients_dict.items()]
#     except:
#         print("Failed to parse ingredients as dictionary")
#         return []

# def evaluate_ingredients(ingredients):
#     good, bad = [], []
#     total_weighted_healthiness = 0
#     total_weightage = 0

#     for ing, percentage in ingredients:
#         print(f"Checking ingredient: {ing}")
#         best_match = None
#         best_score = 0
        
#         for rating_key, rating_value in ratings.items():
#             score = similar(ing.lower(), rating_key.lower())
#             if score > best_score:
#                 best_score = score
#                 best_match = (rating_key, rating_value)
        
#         if best_score > 0.7:
#             rating_value = best_match[1]
#             print(f"Best match for {ing}: {best_match[0]} with rating {rating_value}")
            
#             total_weighted_healthiness += percentage * rating_value
#             total_weightage += percentage

#             if rating_value >= 12:
#                 good.append(f"{ing} ({percentage}%)")
#             else:
#                 bad.append(f"{ing} ({percentage}%)")
#         else:
#             print(f"No good match found for {ing}")

#     if total_weightage == 0:
#         health_percentage = 0
#     else:
#         healthiness_score = total_weighted_healthiness / total_weightage
#         health_percentage = (healthiness_score / 20) * 100

#     print(f"Final scores - Total Weighted: {total_weighted_healthiness}, Total Weight: {total_weightage}, Percentage: {health_percentage}")
#     return round(health_percentage, 2), good, bad

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "image" in request.files:
#         print("Image file received")
#         try:
#             image = request.files["image"].read()
#             print(f"Image size: {len(image)} bytes")
            
#             processed_image = preprocess_image(image)
#             print("Image preprocessed successfully")
#             print(f"Processed image shape: {processed_image.shape}")
            
#             prediction = model.predict(processed_image)
#             print(f"Model prediction shape: {prediction.shape}")
#             score = tf.nn.softmax(prediction)
#             confidence = np.max(score) * 100
#             class_index = np.argmax(score)
#             predicted_class = model.classes_[class_index]
#             print(f"Raw predicted class: {predicted_class} with confidence: {confidence}%")
            
#             predicted_class = predicted_class.replace('_', ' ').lower()
#             print(f"Cleaned predicted class: {predicted_class}")
            
#             ingredients = get_ingredients(predicted_class)
#             print(f"Found ingredients: {ingredients}")
            
#             if not ingredients:
#                 return jsonify({
#                     "error": "Could not identify the food item or find its ingredients. Please try again with a clearer image.",
#                     "product": predicted_class
#                 }), 404
            
#         except Exception as e:
#             print(f"Error during image processing: {str(e)}")
#             return jsonify({"error": str(e)}), 500
#     elif "product_name" in request.form:
#         product_name = request.form["product_name"]
#         ingredients = get_ingredients(product_name)
#         predicted_class = product_name
#     else:
#         return jsonify({"error": "No image or product name provided"}), 400

#     if not ingredients:
#         return jsonify({"error": "Product not found or no ingredients listed"}), 404

#     health_score, good_ingredients, bad_ingredients = evaluate_ingredients(ingredients)
#     print(f"Health evaluation complete - score: {health_score}")

#     return jsonify({
#         "product": predicted_class,
#         "ingredients": ingredients,
#         "good_ingredients": good_ingredients,
#         "bad_ingredients": bad_ingredients,
#         "health_score": health_score
#     })

# if __name__ == "__main__":
#     app.run(debug=True, port=5001)
