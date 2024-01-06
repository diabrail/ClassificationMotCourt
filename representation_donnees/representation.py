from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class TextRepresentation:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer()

    def tfidf_representation(self):
        # Assurez-vous que 'Lemmatized_Text' et 'Label' existent dans votre DataFrame
        if 'Lemmatized_Text' not in self.df.columns or 'Label' not in self.df.columns:
            raise ValueError("Les colonnes 'Lemmatized_Text' et 'Label' sont nécessaires dans le DataFrame.")

        X_train, X_test, y_train, y_test = train_test_split(self.df['Lemmatized_Text'].apply(' '.join),
                                                            self.df['Label'],
                                                            test_size=0.2,
                                                            random_state=42)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        return X_train_tfidf, X_test_tfidf, y_train, y_test

    def represent_single_tweet(self, tweet):
        # Assurez-vous que le vectorizer est déjà ajusté (fitted) sur l'ensemble d'entraînement
        if self.vectorizer is None or not hasattr(self.vectorizer, 'vocabulary_'):
            raise ValueError("Le vectorizer doit être ajusté (fitted) avant de représenter un tweet unique.")

        # Prétraiter le tweet
        preprocessed_tweet = self.preprocessor.clean_tweet(tweet).lower()
        preprocessed_tweet = self.preprocessor.tokenize_text(preprocessed_tweet)
        preprocessed_tweet = self.preprocessor.lemmatize_text(preprocessed_tweet)

        # Convertir le tweet prétraité en une chaîne de caractères
        preprocessed_tweet_str = ' '.join(preprocessed_tweet)

        # Représenter le tweet avec le vectorizer TF-IDF
        tweet_tfidf = self.vectorizer.transform([preprocessed_tweet_str])

        return tweet_tfidf

