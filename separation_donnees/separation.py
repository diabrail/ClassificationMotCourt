# Utilisation de la classe DataSplitter
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split_data(self):
        # Division des données en ensembles d'entraînement, de validation et de test
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

