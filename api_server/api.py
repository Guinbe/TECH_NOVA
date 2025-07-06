from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import google.generativeai as genai

# --- 1. INITIALISATION ET CONFIGURATION ---
print("--- Démarrage du serveur API ---")
app = Flask(__name__)

# Configuration du modèle de classification
MODEL_PATH = 'cameroun_food_model.h5'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['ekwang', 'eru', 'jollof-ghana', 'ndole', 'palm-nut-soup', 'unknow', 'waakye']

# --- CONFIGURATION DE L'API GEMINI ---
try:
    # Colle ta NOUVELLE clé API ici, juste avant de lancer
    GEMINI_API_KEY = "AIzaSyBlBNPtnSY-mgRC11VJRJxEp6-CSF0eaJw" 
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Modèle Gemini configuré avec succès.")
except Exception as e:
    print(f"ERREUR : Impossible de configurer Gemini. Vérifiez votre clé API. Erreur : {e}")
    gemini_model = None

# --- CHARGEMENT DU MODÈLE LOCAL ---
try:
    print(f"Chargement du modèle depuis '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modèle de classification chargé avec succès.")
except Exception as e:
    print(f"ERREUR CRITIQUE : Impossible de charger le modèle '{MODEL_PATH}'. Erreur: {e}")
    model = None

# --- FONCTION POUR OBTENIR LA DESCRIPTION (inchangée) ---
def get_dish_description_from_gemini(dish_name):
    if not gemini_model or dish_name == "unknow":
        return "Description non disponible."
    try:
        prompt = f"Tu es un expert de la cuisine africaine. Décris brièvement et de manière alléchante le plat camerounais appelé '{dish_name}' en 25 mots maximum."
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API Gemini : {e}")
        return "Description temporairement indisponible."

# --- Fonction prepare_image (inchangée) ---
def prepare_image(image):
    img = image.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==============================================================================
# --- POINTS D'ENTRÉE DE L'API ---
# ==============================================================================

# --- POINT D'ENTRÉE POUR L'IMAGE (inchangé) ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None: return jsonify({'error': 'Modèle non dispo.'}), 500
    if 'file' not in request.files: return jsonify({'error': 'Fichier manquant.'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'Fichier non sélectionné.'}), 400
    try:
        image = Image.open(io.BytesIO(file.read()))
        prepared_image = prepare_image(image)
        predictions = model.predict(prepared_image)
        score = float(np.max(predictions[0]))
        predicted_class_name = CLASS_NAMES[int(np.argmax(predictions[0]))]
        description = get_dish_description_from_gemini(predicted_class_name)
        response_data = {'prediction': {'plat': predicted_class_name, 'confiance': score, 'description': description}}
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': f'Erreur de traitement image: {e}'}), 500

# --- NOUVEAU : POINT D'ENTRÉE POUR LE TEXTE ---
@app.route('/identify_by_text', methods=['POST'])
def identify_by_text():
    """Reçoit une description textuelle et utilise Gemini pour l'identifier."""
    if not gemini_model:
        return jsonify({'error': 'Le service d\'identification par texte n\'est pas disponible.'}), 500

    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({'error': 'Description textuelle manquante.'}), 400

    user_description = data['description']
    print(f"Reçu pour identification textuelle : '{user_description}'")

    try:
        possible_dishes = ", ".join([name for name in CLASS_NAMES if name != "unknow"])
        prompt = f"""
        Tu es un expert de la cuisine camerounaise. Un utilisateur a décrit un plat comme ceci : "{user_description}".
        En te basant sur cette description, quel plat parmi la liste suivante correspond le mieux : [{possible_dishes}] ?
        Réponds UNIQUEMENT avec le nom du plat au format "nom_du_plat" (par exemple 'poulet_dg').
        Si aucun plat ne correspond, réponds "unknow".
        """
        
        response = gemini_model.generate_content(prompt)
        identified_dish = response.text.strip().lower()

        if identified_dish not in CLASS_NAMES:
            identified_dish = "unknow"
        
        print(f"Gemini a identifié : '{identified_dish}'")
        description = get_dish_description_from_gemini(identified_dish)

        response_data = {
            'prediction': {
                'plat': identified_dish,
                'confiance': 0.99 if identified_dish != "unknow" else 0.40, # On simule une confiance
                'description': description
            }
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Erreur de traitement texte : {e}'}), 500

# --- Point d'entrée index (inchangé) ---
@app.route('/', methods=['GET'])
def index():
    return "<h1>Bienvenue sur l'API KmerFood Lens v3 (avec recherche texte) !</h1>"

# --- LANCEMENT DU SERVEUR (inchangé) ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)