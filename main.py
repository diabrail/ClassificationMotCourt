# This is a sample Python script.
import nltk
# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import requests
from collecteDonnees.Collecte import *
from exploration_donnees.exploration import *
from pretraitement_donnees.pretraitement import *
from representation_donnees.representation import *
from separation_donnees.separation import *
from model_naive_bayes.naive_bayes import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # Utilisation des classes
    url = "https://www.usna.edu/Users/cs/nchamber/data/twitter/general%20tweets.txt"
    local_filename = "general_tweets.txt"

    # Télécharger les données
    print("------------ Télécharger les données ---------------")
    downloader = DataDownloader(url, local_filename)
    downloader.download_data()

    # Explorer les données
    print("----------Explorer les données----------------")
    explorer = DataExplorer(filename=local_filename)
    explorer.explore_data()
    explored_data = explorer.get_explored_data().copy()

    # Prétraitement des données
    nltk.download('punkt')
    nltk.download('stopwords')

    print("------------ Prétraitement des données ---------------")
    preprocessor = DataPreprocessor(filename=local_filename)
    preprocessed_data = preprocessor.preprocess_data()

    print("------------ Afficher les premières lignes du DataFrame résultant ---------------")
    # Afficher les premières lignes du DataFrame résultant
    #print(preprocessed_data.head())

    # Utilisation de la classe TextRepresentation
    print("------------ classe TextRepresentation ---------------")
    representation = TextRepresentation(df=preprocessed_data)
    print("------------ Afficher TextRepresentation---------------")
    X_train_tfidf, X_test_tfidf, y_train, y_test = representation.tfidf_representation()
    # Utilisation de la classe DataSplitter

    splitter = DataSplitter(X=X_train_tfidf, y=y_train)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data()

    # Afficher la forme des ensembles
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of X_test:", X_test.shape)
    print("----------------------TextClassifier -------------------------")
    classifier = TextClassifier(X_train, X_val, X_test, y_train, y_val, y_test)
    print("----------------------Entrainement TextClassifier -------------------------")
    classifier.train_classifier(classifier.X_train, classifier.y_train)



    # Évaluer le classificateur sur l'ensemble d'entraînement
    print("----------------------Évaluer le classificateur sur l'ensemble d'entraînement -------------------------")
    #classifier.evaluate_classifier(classifier.X_train, y_train, "d'entraînement")

    # Évaluer le classificateur sur l'ensemble de validation
    print("----------------------Évaluer le classificateur sur l'ensemble de validation -------------------------")
    #classifier.evaluate_classifier(classifier, classifier.X_validation, classifier.y_validation, "de validation")

    # Évaluer le classificateur sur l'ensemble de test
    print("----------------------Évaluer le classificateur sur l'ensemble de test -------------------------")
    #classifier.evaluate_classifier(classifier.X_test, classifier.y_test, "de test")

    # Utilisation de la classe TextClassifier pour entraîner le modèle
    print("best_classifier nexxx-------------------------------------------------")

    # Prétraitement du nouveau tweet (à adapter en fonction de votre prétraitement)
    new_tweet = "Hannity and Bachman on Health Care Bill - CENTRAL IOWA 912 PROJECT http://ow.ly/yNd"
    #new_tweet = preprocessor.tokenize_text(new_tweet)
    #new_tweet = preprocessor.lemmatize_text(new_tweet)
    print("best_classifier nexxx-------------------------------------------------")
    predicted_class = classifier.classify(new_tweet)

    print(f"Le nouveau tweet est classé dans la catégorie : {predicted_class}")

    # Ajustement des hyperparamètres sur l'ensemble de validation
    alphas = [0.1, 0.5, 1.0, 1.5, 2.0]  # Vous pouvez ajuster cette liste d'hyperparamètres
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
    #classifier.evaluate_classifier(X_test_tfidf, y_test, "de test")

    # Utilisation de la classe TextClassifier pour entraîner le modèle
    classifier = TextClassifier(X_train_tfidf, X_val, X_test_tfidf, y_train, y_val, y_test)

    # Ajustement des hyperparamètres sur l'ensemble de validation
    alphas = [0.1, 0.5, 1.0, 1.5, 2.0]  # Vous pouvez ajuster cette liste d'hyperparamètres
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
    #classifier.evaluate_classifier(X_test_tfidf, y_test, "de test")

    # Utilisation de la classe TextClassifier pour entraîner le modèle avec recherche par grille
    classifier = TextClassifier(X_train_tfidf, X_val, X_test_tfidf, y_train, y_val, y_test)

    # Définir la grille d'hyperparamètres à explorer
    param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}

    # Effectuer la recherche par grille pour trouver les meilleurs hyperparamètres
    best_classifier = classifier.grid_search(param_grid)

    # Évaluer le modèle avec les meilleurs hyperparamètres sur l'ensemble de test
    #classifier.evaluate_classifier(X_test_tfidf, y_test, "de test")

    # Utilisation de la classe TextClassifier pour entraîner le modèle avec recherche par grille
    classifier = TextClassifier(X_train_tfidf, X_val, X_test_tfidf, y_train, y_val, y_test)

    # Définir la grille d'hyperparamètres à explorer
    param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}

    # Effectuer la recherche par grille pour trouver les meilleurs hyperparamètres
    best_classifier = classifier.grid_search(param_grid)

    # Évaluer le modèle avec les meilleurs hyperparamètres sur l'ensemble de test
    #classifier.evaluate_classifier(X_test_tfidf, y_test, "de test")

    # Analyser les erreurs du modèle sur l'ensemble de test
    #classifier.analyze_errors(X_test_tfidf, y_test, "de test")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
