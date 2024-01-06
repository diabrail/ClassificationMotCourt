import nltk
from flask import Flask, render_template, request
from sklearn.exceptions import NotFittedError  # Ajout de cette importation
from collecteDonnees.Collecte import DataDownloader
from exploration_donnees.exploration import DataExplorer
from model_naive_bayes.naive_bayes import TextClassifier
from representation_donnees.representation import TextRepresentation
from pretraitement_donnees.pretraitement import DataPreprocessor
from separation_donnees.separation import DataSplitter

# Télécharger les données
url = "https://www.usna.edu/Users/cs/nchamber/data/twitter/general%20tweets.txt"
local_filename = "general_tweets.txt"
downloader = DataDownloader(url, local_filename)
downloader.download_data()

# Explorer les données
explorer = DataExplorer(filename=local_filename)
explorer.explore_data()
explored_data = explorer.get_explored_data().copy()

# Prétraitement des données
nltk.download('punkt')
nltk.download('stopwords')
preprocessor = DataPreprocessor(filename=local_filename)
preprocessed_data = preprocessor.preprocess_data()

# Utiliser la classe TextRepresentation pour créer un vecteur TF-IDF
representation = TextRepresentation(df=preprocessed_data)


# Entraîner le classificateur avec les données d'entraînement
X_train_tfidf, X_test_tfidf, y_train, y_test = representation.tfidf_representation()
# Utilisation de la classe DataSplitter


splitter = DataSplitter(X=X_train_tfidf, y=y_train)

X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data()

classifier = TextClassifier(X_train, X_val, X_test, y_train, y_val, y_test, representation.vectorizer, preprocessor)
classifier.train_classifier(X_train, y_train)  # Passer les données d'entraînement

# Initialiser l'application Flask
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        tweet = request.form['tweet']

        try:
            tweet = TextRepresentation(df=preprocessed_data).represent_single_tweet(tweet)
            result = classifier.classify(tweet)
            print("----------- result ---------------- : ", result)
        except NotFittedError as e:
            result = "Erreur : Le classificateur n'est pas correctement entraîné."
            print(f"Erreur : {e}")

        return render_template('resultat.html', tweet=tweet, result=result)

if __name__ == '__main__':
    app.run(debug=True)
