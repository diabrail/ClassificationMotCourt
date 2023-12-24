from flask import Flask, render_template, request
from model_naive_bayes.naive_bayes import * # Assurez-vous que le chemin est correct
from representation_donnees.representation import TextRepresentation  # Importez la classe TextRepresentation si nécessaire

app = Flask(__name__)
classifier = TextClassifier()  # Initialisez votre modèle ici
representation = TextRepresentation()  # Initialisez votre classe TextRepresentation si nécessaire

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        tweet = request.form['tweet']

        # Prétraitement du tweet (si nécessaire)
        preprocessed_tweet = representation.preprocess_text(tweet)

        # Représentation du tweet en vecteur TF-IDF
        tweet_tfidf = representation.transform_text_to_tfidf(preprocessed_tweet)

        # Classification du tweet
        result = classifier.classify(tweet_tfidf)  # Méthode à implémenter dans votre classe TextClassifier

        return render_template('result.html', tweet=tweet, result=result)

if __name__ == '__main__':
    app.run(debug=True)
