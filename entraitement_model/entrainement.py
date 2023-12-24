from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB


class TextClassifier:
    def __init__(self, X_train, X_validation, X_test, y_train, y_validation, y_test):
        self.X_train = X_train
        self.X_validation = X_validation
        self.X_test = X_test
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test
        self.classifier = None

    def train_classifier(self, alpha=1.0):
        # Initialiser un classificateur Naive Bayes avec l'hyperparamètre alpha
        classifier = MultinomialNB(alpha=alpha)

        # Entraîner le classificateur sur les données d'entraînement
        classifier.fit(self.X_train, self.y_train)

        self.classifier = classifier

        return classifier

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

