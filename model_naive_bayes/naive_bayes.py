from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt

import representation_donnees.representation
from representation_donnees.representation import *

class TextClassifier:
    def __init__(self, X_train, X_validation, X_test, y_train, y_validation, y_test, vectorizer, preprocessor):
        self.X_train = X_train
        self.X_validation = X_validation
        self.X_test = X_test
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test
        self.classifier = None
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer  # Utilisez le vectorizer fourni en argument
    def train_classifier(self, X, y):
        # Définir les valeurs d'alpha que vous souhaitez essayer
        alphas = [0.1, 0.5, 1.0, 1.5, 2.0]

        # Créer une grille de recherche
        param_grid = {'alpha': alphas}
        grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')

        # Exécuter la recherche par grille sur les données d'entraînement
        grid_search.fit(X, y)

        # Obtenir le meilleur modèle après la recherche par grille
        best_classifier = grid_search.best_estimator_

        # Entraîner le modèle sur l'ensemble complet d'entraînement avec la meilleure valeur d'alpha
        best_classifier.fit(X, y)

        # Enregistrer le meilleur classificateur dans votre classe pour une utilisation ultérieure
        self.classifier = best_classifier
        print("best_classifier :", best_classifier)

        # Afficher la meilleure valeur d'alpha
        print(f"Meilleure valeur d'alpha : {grid_search.best_params_['alpha']}")

    def evaluate_classifier(self, X, y, set_name):
        if self.classifier is None:
            raise ValueError("Le classificateur n'a pas été entraîné. Utilisez train_classifier avant l'évaluation.")

        # Prédire les labels sur les données
        y_pred = self.classifier.predict(X)

        # Évaluer la performance du modèle
        accuracy = accuracy_score(y, y_pred)
        print(f"Précision sur l'ensemble {set_name}: {accuracy:.4f}\n")

        # Afficher le rapport de classification
        print(f"Rapport de classification sur l'ensemble {set_name}:\n")
        print(classification_report(y, y_pred))

        # Afficher la matrice de confusion
        print(f"Matrice de confusion sur l'ensemble {set_name}:\n")
        print(confusion_matrix(y, y_pred))

        # Visualiser la matrice de confusion avec seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y)),
                    yticklabels=sorted(set(y)))
        plt.title(f"Matrice de confusion sur l'ensemble {set_name}")
        plt.xlabel('Prédiction')
        plt.ylabel('Vraie valeur')
        plt.show()

    def grid_search(self, param_grid):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Les données d'entraînement ne sont pas définies.")

        # Initialiser un classificateur Naive Bayes
        classifier = MultinomialNB()

        # Utiliser la recherche par grille pour trouver les meilleurs hyperparamètres
        grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', verbose=1)
        grid_search.fit(self.X_train, self.y_train)

        # Afficher les meilleurs hyperparamètres
        print("Meilleurs hyperparamètres trouvés par la recherche par grille:")
        print(grid_search.best_params_)

        # Utiliser les meilleurs hyperparamètres pour entraîner le modèle
        best_classifier = grid_search.best_estimator_

        self.classifier = best_classifier

        return best_classifier

    def classify(self, tweet):
        if self.classifier is None:
            raise ValueError(
                "Le classificateur n'a pas été entraîné. Utilisez train_classifier avant la classification.")

        # Utilisez le modèle pour classer le tweet
        print("tweet :", tweet)
        tweet = self.preprocessor.clean_tweet(tweet)
        tweet = self.preprocessor.tokenize_text(tweet)
        tweet = self.preprocessor.lemmatize_text(tweet)
        print("tweet apres transform :", tweet)
        preprocessed_tweet = ' '.join(tweet)
        preprocessed_tweet = str(preprocessed_tweet).lower()
        print("preprocessed_tweet apres transform :", preprocessed_tweet)

        tweet_tfidf = self.vectorizer.transform([preprocessed_tweet])

        print("tweet_tfidf apres transform :", tweet_tfidf)
        prediction = self.classifier.predict(tweet_tfidf)

        # Retournez le résultat de la classification
        return prediction[0]

