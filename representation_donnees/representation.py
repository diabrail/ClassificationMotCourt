from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class TextRepresentation:
    def __init__(self, df):
        self.df = df

    def tfidf_representation(self):
        # Séparation des données en ensembles d'entraînement et de test
        print("-----------------tfidf_representation---------------")
        X_train, X_test, y_train, y_test = train_test_split(self.df['Lemmatized_Text'].apply(' '.join), self.df['Label'], test_size=0.2, random_state=42)
        print("-----------------tfidf_representation---------------")
        # Création d'un vecteur TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Vous pouvez ajuster le nombre maximal de fonctionnalités
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        return X_train_tfidf, X_test_tfidf, y_train, y_test

