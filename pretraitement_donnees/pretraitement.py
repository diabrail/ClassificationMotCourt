import re

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class DataPreprocessor:
    def __init__(self, filename):
        self.filename = filename
        self.stop_words = set(stopwords.words('english'))  # Liste des stop words en anglais

    def preprocess_data(self):
        # Charger les données dans un DataFrame
        df = pd.read_csv(self.filename, delimiter='\t', header=None, names=['Text', 'Label'])

        # Nettoyage des tweets
        df['Cleaned_Text'] = df['Text'].apply(self.clean_tweet)

        # Tokenisation et suppression des stop words des tweets
        df['Tokenized_Text'] = df['Cleaned_Text'].apply(self.tokenize_text)

        # Réduction des mots à leur forme de base (utilisation de la lemmatisation)
        df['Lemmatized_Text'] = df['Tokenized_Text'].apply(self.lemmatize_text)

        # Réduction des mots à leur forme de base (utilisation du stemming)
        # df['Stemmed_Text'] = df['Tokenized_Text'].apply(self.stem_text)

        return df[['Text', 'Label', 'Cleaned_Text', 'Tokenized_Text', 'Lemmatized_Text']]

    def clean_tweet(self, text):
        # Supprimer les mentions
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        # Supprimer les liens
        text = re.sub('https?://[A-Za-z0-9./]+', '', text)
        # Supprimer la ponctuation et les caractères spéciaux
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convertir le texte en minuscules
        text = text.lower()
        return text

    def tokenize_text(self, text):
        # Tokenisation des tweets en mots et suppression des stop words
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return tokens

    def lemmatize_text(self, tokens):
        # Réduction des mots à leur forme de base (utilisation de la lemmatisation)
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def stem_text(self, tokens):
        # Réduction des mots à leur forme de base (utilisation du stemming)
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return stemmed_tokens




