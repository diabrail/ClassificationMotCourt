from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TextClassifier:
    def __init__(self, X_train, X_validation, X_test, y_train, y_validation, y_test):
        self.X_train = X_train
        self.X_validation = X_validation
        self.X_test = X_test
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()

    def train_classifier(self):
        # Initialiser un classificateur Naive Bayes
        classifier = MultinomialNB()

        # Entraîner le classificateur sur les données d'entraînement
        classifier.fit(self.X_train, self.y_train)

        return classifier

    def train_classifier(self, X_train, y_train):
        # Entraînez votre classificateur avec les données d'entraînement
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_tfidf, y_train)

    def evaluate_classifier(self, classifier, X, y, set_name):
        # Prédire les labels sur les données
        y_pred = classifier.predict(X)

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
        tweet_tfidf = self.vectorizer.transform([tweet])
        prediction = self.classifier.predict(tweet_tfidf)

        # Retournez le résultat de la classification
        return prediction[0]
