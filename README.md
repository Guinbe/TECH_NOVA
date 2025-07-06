<<<<<<< HEAD
KmerFood Lens - Le Shazam de la Cuisine Camerounaise

Description du Projet

KmerFood Lens est une application mobile innovante conçue pour le hackathon, servant de pont entre la technologie et le riche patrimoine culinaire du Cameroun. Grâce à une intelligence artificielle avancée, l'application permet aux utilisateurs d'identifier instantanément des plats traditionnels camerounais, d'obtenir des informations culturelles et d'interagir avec les résultats de manière intuitive et utile.
Architecture Technique
Le projet repose sur une architecture moderne full-stack, comprenant :

- Modèle de Deep Learning : Basé sur MobileNetV2, entraîné localement avec 7 classes différentes pour la reconnaissance d'images.
- Modèle de Langage : Google Gemini pour la génération de descriptions et l'identification textuelle.
- Backend : API en Python (Flask) orchestrant les deux modèles d'IA.
- Frontend : Application Android native (Java) offrant une expérience utilisateur riche et interactive.

Fonctionnalités Clés

- Reconnaissance par Image : Prenez ou choisissez une photo d'un plat, et l'IA l'identifie en quelques secondes.
- Reconnaissance par Texte & Voix : Décrivez un plat avec vos propres mots ou à voix haute, et l'IA le retrouve pour vous.
- Descriptions Intelligentes : Obtenez une description riche et engageante du plat identifié, générée dynamiquement par une IA.
- Accessibilité : Fonction Text-to-Speech pour lire les résultats à voix haute, rendant l'application accessible aux personnes malvoyantes.
- Actions Contextuelles : Pour chaque plat reconnu, trouvez instantanément une recette sur le web ou des restaurants à proximité grâce à la géolocalisation.

Déploiement et Tests
Pour lancer et tester KmerFood Lens, deux environnements sont nécessaires : un pour le serveur backend (IA) et un pour l'application Android.
Prérequis
Backend

Python 3.9+
Environnement virtuel (venv)
Clé d'API valide pour Google Gemini (disponible ici).

Frontend

Android Studio (dernière version recommandée)
Appareil Android physique ou émulateur
L'ordinateur et l'appareil Android doivent être sur le même réseau Wi-Fi

Étape 1 : Lancement du Serveur Backend

Clonez le projet et ouvrez un terminal à la racine du dossier.

Créez et activez l'environnement virtuel :
# Créer l'environnement
python -m venv venv

#activer l'execution des scripts
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Activer sur Windows
venv\Scripts\activate

# Activer sur macOS/Linux
source venv/bin/activate


Installez les dépendances Python :
python -m pip install -r requirements.txt

Note : Si requirements.txt n'est pas fourni, installez manuellement :
python -m pip install flask tensorflow google-generativeai Pillow numpy


Configurez la clé API Gemini :

Ouvrez le fichier api.py.
Remplacez GEMINI_API_KEY = "VOTRE_CLE_API_ICI" par votre clé API Gemini valide.


Lancez le serveur :
python api.py

Le terminal affichera que le serveur tourne sur http://0.0.0.0:5000. Notez l'adresse IP locale (ex. : http://192.168.1.10:5000).


Étape 2 : Lancement de l'Application Android

Ouvrez le projet Android :
Lancez Android Studio.
Cliquez sur Open et sélectionnez le dossier de l'application Android. Laissez Gradle synchroniser le projet.


Configurez l'adresse IP du serveur :
Ouvrez MainActivity.java et ResultActivity.java.
Remplacez les constantes API_IDENTIFY_TEXT_URL et API_UPLOAD_URL par l'adresse IP du serveur (ex. : http://192.168.1.10:5000/...).


Autorisez le trafic réseau (pare-feu) :
Assurez-vous que le pare-feu autorise les connexions entrantes sur le port TCP 5000 (Windows : Pare-feu Windows Defender > Paramètres avancés > Règles de trafic entrant > Nouvelle règle).


Lancez l'application :
Connectez un appareil Android ou utilisez un émulateur.
Cliquez sur Run 'app' (icône verte) dans Android Studio.



Étape 3 : Scénarios de Test
Test du Parcours Image

Sur l'écran d'accueil, cliquez sur l'icône +.
Choisissez une image de plat depuis la galerie.
Vérifiez que l'écran de résultat affiche le nom, la description, et que les boutons (Écouter, Recette, Restaurant) sont fonctionnels.

Test du Parcours Texte/Voix

Sur l'écran d'accueil, cliquez sur l'icône micro, décrivez un plat, ou tapez une description.
Cliquez sur Envoyer.
Vérifiez que l'écran de résultat affiche les informations correspondantes.

Test de Robustesse

Arrêtez le serveur Python (CTRL+C).
Essayez une reconnaissance depuis l'application. Un message d'erreur clair ("Échec de la connexion...") doit s'afficher, sans crash de l'application.
=======
# Tech_Nova
KmerFood Lens est une application mobile innovante conçue pour le hackathon, servant de pont entre la technologie et le riche patrimoine culinaire du Cameroun. Grâce à une intelligence artificielle avancée, l'application permet aux utilisateurs d'identifier instantanément des plats traditionnels camerounais.
>>>>>>> 5487aa3e4369eeadff476967779b495acef9f00e
