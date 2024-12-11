import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request # type: ignore
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
faq_data = pd.read_csv("nlp_dataset.csv")

# Convert all questions to lowercase
faq_data['Question'] = faq_data['Question'].str.lower()

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Tokenize, remove stopwords, and stem
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Preprocess dataset questions
faq_data['Processed_Question'] = faq_data['Question'].apply(preprocess_text)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed questions
tfidf_matrix = tfidf_vectorizer.fit_transform(faq_data['Processed_Question'])

# Function to get answer using TF-IDF
def get_answer_tfidf(user_input):
    # Preprocess user input
    processed_input = preprocess_text(user_input)
    user_tfidf = tfidf_vectorizer.transform([processed_input])  # Vectorize input
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    most_similar_idx = cosine_similarities.argmax()
    # Return answer if similarity is above threshold
    if cosine_similarities[0, most_similar_idx] > 0.2:
        return faq_data.iloc[most_similar_idx]['Answer']
    else:
        return "I'm sorry, I don't have an answer for that. Try asking another NLP question."

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def chatbot():
    response = None
    user_input = None
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        if user_input:
            response = get_answer_tfidf(user_input)  # Get answer
    return render_template("index.html", response=response, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)




