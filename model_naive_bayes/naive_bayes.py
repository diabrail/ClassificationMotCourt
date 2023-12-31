from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from pretraitement_donnees.pretraitement import *

# C. Modélisation

## 1. Sélection du modèle de classification (Naive Bayes)

class TextClassifier:
    def __init__(self, X_train, X_validation, X_test, y_train, y_validation, y_test, vectorizer, preprocessor=None):
        self.X_train = X_train
        self.X_validation = X_validation
        self.X_test = X_test
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test
        self.classifier = MultinomialNB()
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer  # Utilisez le vectorizer fourni en argument

    def train_classifier(self, X, y):
        # Entraîner le classificateur sur les données d'entraînement
        self.classifier.fit(X, y)

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

        # Visualiser la matrice de confusion avec seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y)),
                    yticklabels=sorted(set(y)))
        plt.title(f"Matrice de confusion sur l'ensemble {set_name}")
        plt.xlabel('Prédiction')
        plt.ylabel('Vraie valeur')
        plt.show()

    def classify(self, tweet):
        # Utilisez le modèle pour classer le tweet
        tweet = self.preprocessor.clean_tweet(tweet)
        tweet = self.preprocessor.tokenize_text(tweet)
        tweet = self.preprocessor.lemmatize_text(tweet)
        preprocessed_tweet = ' '.join(tweet)
        preprocessed_tweet = str(preprocessed_tweet).lower()

        tweet_tfidf = self.vectorizer.transform([preprocessed_tweet])
        prediction = self.classifier.predict(tweet_tfidf)

        # Retournez le résultat de la classification
        return prediction[0]


    # Utilisation de la classe TextClassifier pour entraîner le modèle
    classifier = TextClassifier(X_train_tfidf, X_val, X_test_tfidf, y_train, y_val, y_test, representation.vectorizer)
    classifier.train_classifier(X_train_tfidf, y_train)
    
    # Évaluer le classificateur sur l'ensemble de test
    classifier.evaluate_classifier(X_test_tfidf, y_test, "de test")
    
    # Classer un tweet
    tweet = "@RajeevMasand I like to ask challenging and unique q's ones that can them thinking about the diaspora. Please tweet if KJo comes on..A MUST!"
    classification_result = classifier.classify(tweet)
    
    print(f"Classification du tweet : {classification_result}")

