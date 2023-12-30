from flask import Flask, render_template, request
from model_naive_bayes.naive_bayes import TextClassifier
from representation_donnees.representation import TextRepresentation
from pretraitement_donnees.pretraitement import DataPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
classifier = TextClassifier()
representation = TextRepresentation()
preprocessor = DataPreprocessor(filename=None)
tfidf_vectorizer = TfidfVectorizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        tweet = request.form['tweet']

        # Prétraitement du tweet
        preprocessed_tweet = preprocessor.clean_tweet(tweet)
        preprocessed_tweet = preprocessor.tokenize_text(preprocessed_tweet)
        preprocessed_tweet = preprocessor.lemmatize_text(preprocessed_tweet)

        # Représentation du tweet en vecteur TF-IDF
        tweet_tfidf = tfidf_vectorizer.transform([preprocessed_tweet])

        # Classification du tweet
        result = classifier.classify(tweet_tfidf)

        return render_template('index.html', tweet=tweet, result=result)

if __name__ == '__main__':
    app.run(debug=True)
