from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB


class TextClassifier:
    def __init__(self, X_train=None, X_validation=None, X_test=None, y_train=None, y_validation=None, y_test=None, vectorizer=None,
                 preprocessor=None):
        self.X_train = X_train
        self.X_validation = X_validation
        self.X_test = X_test
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test
        self.vectorizer = vectorizer
        self.classifier = MultinomialNB()
        self.preprocessor = preprocessor


    def train_classifier(self, X_train, y_train):
        # Entraîner le classificateur sur les données d'entraînement
        self.classifier.fit(X_train, y_train)

    def evaluate_classifier(self, X, y, set_name):
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




    def classify(self, tweet):
        # Prétraitement du tweet (nettoyage, lemmatisation, etc.) si nécessaire

        # Utilisez le modèle pour classer le tweet
        #tweet = self.preprocessor.clean_tweet(tweet)
        tweet = self.preprocessor.tokenize_text(tweet)
        tweet = self.preprocessor.lemmatize_text(tweet)
        preprocessed_tweet = ' '.join(tweet)
        preprocessed_tweet = str(preprocessed_tweet).lower()

        tweet_tfidf = self.vectorizer.transform([preprocessed_tweet])
        prediction = self.classifier.predict(tweet_tfidf)
        # Retournez le résultat de la classification
        return prediction[0]

