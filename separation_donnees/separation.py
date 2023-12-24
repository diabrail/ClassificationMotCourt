from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split_data(self, test_size=0.2, validation_size=0.25, random_state=42):
        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        # Division des données d'entraînement en ensembles d'entraînement et de validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test

