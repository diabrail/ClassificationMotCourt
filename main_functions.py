# main_functions.py

from collecteDonnees.Collecte import DataDownloader
from exploration_donnees.exploration import DataExplorer
from pretraitement_donnees.pretraitement import DataPreprocessor
from representation_donnees.representation import TextRepresentation
from separation_donnees.separation import DataSplitter
from model_naive_bayes.naive_bayes import TextClassifier
from sklearn.metrics import accuracy_score

def main():
    url = "https://www.usna.edu/Users/cs/nchamber/data/twitter/general%20tweets.txt"
    local_filename = "general_tweets.txt"

    # Télécharger les données
    print("----------Télécharger les données----------------")
    downloader = DataDownloader(url, local_filename)
    downloader.download_data()

    # Explorer les données
    print("----------Explorer les données----------------")
    explorer = DataExplorer(filename=local_filename)
    explorer.explore_data().copy()

    # Prétraitement des données
    print("----------Prétraitement des données----------------")
    preprocessor = DataPreprocessor(filename=local_filename)
    preprocessed_data = preprocessor.preprocess_data()

    # Afficher les premières lignes du DataFrame résultant
    print(preprocessed_data.head())

    # Utilisation de la classe TextRepresentation
    representation = TextRepresentation(df=preprocessed_data)
    X_train_tfidf, X_test_tfidf, y_train, y_test = representation.tfidf_representation()

    # Afficher la forme des vecteurs TF-IDF
    print("Shape of X_train_tfidf:", X_train_tfidf.shape)
    print("Shape of X_test_tfidf:", X_test_tfidf.shape)

    # Utilisation de la classe DataSplitter
    splitter = DataSplitter(X=X_train_tfidf, y=y_train)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data()

    # Afficher la forme des ensembles
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of X_test:", X_test.shape)

    # Utilisation de la classe TextClassifier
    classifier = TextClassifier(X_train_tfidf, X_val, X_test_tfidf, y_train, y_val, y_test)

    # Ajustement des hyperparamètres sur l'ensemble de validation
    alphas = [0.1, 0.5, 1.0, 1.5, 2.0]
    best_alpha = None
    best_accuracy = 0.0

    for alpha in alphas:
        print(f"Entraînement avec alpha={alpha}")
        trained_classifier = classifier.train_classifier(alpha=alpha)

        # Évaluer le modèle sur l'ensemble de validation
        accuracy_validation = accuracy_score(y_val, trained_classifier.predict(X_val))
        print(f"Précision sur l'ensemble de validation: {accuracy_validation:.4f}\n")

        # Mettre à jour le meilleur alpha si nécessaire
        if accuracy_validation > best_accuracy:
            best_accuracy = accuracy_validation
            best_alpha = alpha

    # Entraîner le modèle avec le meilleur alpha sur l'ensemble d'entraînement complet
    final_classifier = classifier.train_classifier(alpha=best_alpha)

    # Évaluer le modèle sur l'ensemble de test
    classifier.evaluate_classifier(final_classifier, X_test_tfidf, y_test, "de test")
