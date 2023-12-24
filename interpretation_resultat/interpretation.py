class TextClassifier:
    # ... (les autres méthodes restent inchangées)

    def analyze_errors(self, X, y, set_name):
        # Prédire les labels sur les données
        y_pred = self.classifier.predict(X)

        # Trouver les indices des erreurs de prédiction
        incorrect_indices = [i for i, (true_label, pred_label) in enumerate(zip(y, y_pred)) if true_label != pred_label]

        # Afficher un échantillon d'erreurs
        print(f"Exemples d'erreurs sur l'ensemble {set_name}:\n")
        for idx in incorrect_indices[:min(5, len(incorrect_indices))]:
            print(f"Vrai label : {y[idx]}, Prédiction : {y_pred[idx]}")
            print(f"Texte : {X[idx]}\n")
