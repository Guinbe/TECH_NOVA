import tensorflow as tf
import os
import matplotlib.pyplot as plt

def main():
    """Fonction principale pour orchestrer l'entraînement."""
    
    # --- 1. DÉFINITION DES CONSTANTES ET CHEMINS ---
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001 # Taux d'apprentissage fixe

    BASE_DIR = 'Dataset_CamPlate'
    TRAIN_DIR = os.path.join(BASE_DIR, 'train')
    VAL_DIR = os.path.join(BASE_DIR, 'val')
    TEST_DIR = os.path.join(BASE_DIR, 'test') # Ajout du chemin de test
    
    print("--- Démarrage de l'entraînement ---")
    print(f"TensorFlow Version: {tf.__version__}")

    # --- 2. CHARGEMENT DES DONNÉES ---
    print("\n[ETAPE 1/6] Chargement des données...")
    try:
        # On charge maintenant les 3 ensembles
        train_dataset, validation_dataset, test_dataset, class_names = load_data(TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE)
        print(f"Classes trouvées : {class_names}")
    except Exception as e:
        print(f"ERREUR lors du chargement des données : {e}")
        return

    # --- 3. CONSTRUCTION DU MODÈLE ---
    print("\n[ETAPE 2/6] Construction du modèle...")
    try:
        model = build_model(num_classes=len(class_names), image_size=IMAGE_SIZE)
        model.summary()
    except Exception as e:
        print(f"ERREUR lors de la construction du modèle : {e}")
        return

    # --- 4. COMPILATION DU MODÈLE ---
    print("\n[ETAPE 3/6] Compilation du modèle...")
    try:
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
    except Exception as e:
        print(f"ERREUR lors de la compilation : {e}")
        return

    # --- 5. ENTRAÎNEMENT ---
    print("\n[ETAPE 4/6] Lancement de l'entraînement...")
    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=validation_dataset
        )
        print("Entraînement terminé avec succès.")
        save_plot(history, EPOCHS)
    except Exception as e:
        print(f"ERREUR pendant l'entraînement : {e}")
        return
        
    # --- 6. ÉVALUATION FINALE SUR L'ENSEMBLE DE TEST ---
    print("\n[ETAPE 5/6] Évaluation sur l'ensemble de test...")
    try:
        test_loss, test_accuracy = model.evaluate(test_dataset)
        print(f"Résultats sur l'ensemble de test -> Perte: {test_loss:.4f} - Précision: {test_accuracy*100:.2f}%")
    except Exception as e:
        print(f"ERREUR lors de l'évaluation sur le jeu de test : {e}")
        return

    # --- 7. SAUVEGARDE DU MODÈLE ---
    print("\n[ETAPE 6/6] Sauvegarde du modèle final...")
    try:
        model.save('cameroun_food_model.h5')
        print("Modèle sauvegardé dans 'cameroun_food_model.h5'")
        print("\n--- BRAVO, LE SCRIPT EST TERMINÉ ! ---")
    except Exception as e:
        print(f"ERREUR lors de la sauvegarde du modèle : {e}")

def load_data(train_dir, val_dir, test_dir, image_size, batch_size):
    """Charge les datasets d'entraînement, de validation et de test."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size
    )
    class_names = train_ds.class_names # On définit les classes à partir du dossier train
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        class_names=class_names # On force les mêmes classes pour éviter les erreurs
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        class_names=class_names # On force les mêmes classes pour éviter les erreurs
    )
    
    # Optimisation du chargement
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names

def build_model(num_classes, image_size):
    """Construit le modèle de classification."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(*image_size, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def save_plot(history, epochs):
    """Sauvegarde les graphiques de précision et de perte."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Précision (Entraînement)')
    plt.plot(epochs_range, val_acc, label='Précision (Validation)')
    plt.legend(loc='lower right')
    plt.title('Précision')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Perte (Entraînement)')
    plt.plot(epochs_range, val_loss, label='Perte (Validation)')
    plt.legend(loc='upper right')
    plt.title('Perte')
    
    plt.savefig('training_history.png')
    print("Graphique de performance sauvegardé dans 'training_history.png'")


# Point d'entrée du script
if __name__ == "__main__":
    main()