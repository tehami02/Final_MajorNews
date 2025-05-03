from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from textblob import TextBlob
import base64
from io import BytesIO
from gtts import gTTS
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Replace with actual GitHub links to your CSV files
NEWS_DATA_URLS = {
    'The Indian Express': "https://raw.githubusercontent.com/tehami02/News_Analysis_NEW/refs/heads/main/Datanews5.csv",
    'Times of India': "https://raw.githubusercontent.com/tehami02/News_Analysis_NEW/refs/heads/main/Datanews5.csv"
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    source = request.form['source']  # Get selected news source
    session['publisher'] = source  # Save the selected publisher in session
    return redirect(url_for('display_news', source=source))

@app.route('/display_news/<source>')
def display_news(source):
    csv_url = NEWS_DATA_URLS.get(source, '')
    if not csv_url:
        return "News source not found", 404

    df = pd.read_csv(csv_url)
    print("CSV Columns:", df.columns)  # Check the actual column names

    if 'heading' not in df.columns:
        return "CSV file does not have a 'heading' column", 500

    headings = df['heading'].dropna().tolist()
    print("Extracted Headings:", headings)  # Print extracted headings

    return render_template('display_news.html', headings=headings, source=source)


@app.route('/analyze')
def analyze():
    heading = request.args.get('heading')
    source = session.get('publisher', '')
    csv_url = NEWS_DATA_URLS.get(source, '')

    if not csv_url:
        return "News source not found", 404
    
    df = pd.read_csv(csv_url)
    data_row = df[df['heading'] == heading]
    
    if data_row.empty:
        return "Heading not found", 404
    
    article_content = data_row['data'].values[0]

    # Perform text processing
    vectorizer = CountVectorizer(stop_words='english')
    bow_matrix = vectorizer.fit_transform([article_content])
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(bow_matrix)

    bow_data = dict(zip(vectorizer.get_feature_names_out(), bow_matrix.toarray()[0]))
    tfidf_data = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))

    top_bow_words = sorted(bow_data.items(), key=lambda item: item[1], reverse=True)[:5]
    top_tfidf_words = sorted(tfidf_data.items(), key=lambda item: item[1], reverse=True)[:5]

    # Generate synthetic data for Ridge regression
    np.random.seed(42)
    synthetic_data = np.random.normal(0, 0.01, (10, tfidf_matrix.shape[1])) + tfidf_matrix.toarray()
    synthetic_y = np.random.rand(10) * 100

    combined_data = np.vstack([tfidf_matrix.toarray(), synthetic_data])
    combined_y = np.hstack([np.mean(synthetic_y), synthetic_y])

    model = Ridge(alpha=1.0)
    model.fit(combined_data, combined_y)
    predictions = model.predict(combined_data)

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(combined_y)), combined_y, color='blue', label='Actual Values')
    plt.plot(range(len(predictions)), predictions, color='red', label='Predicted Values')
    plt.title('Ridge Regression Analysis')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Perform sentiment analysis
    blob = TextBlob(article_content)
    sentiment_score = blob.sentiment.polarity
    sentiment_label = "Positive 😊" if sentiment_score > 0.1 else "Negative 😡" if sentiment_score < 0 else "Neutral 😐"

    return render_template('analysis_news.html', article_content=article_content, bow_data=bow_data, tfidf_data=tfidf_data, image_data=image_base64, top_bow_words=top_bow_words, top_tfidf_words=top_tfidf_words, sentiment_score=sentiment_score, sentiment_label=sentiment_label)

@app.route('/tts/<heading>')
def text_to_speech(heading):
    source = session.get('publisher', '')
    csv_url = NEWS_DATA_URLS.get(source, '')

    if not csv_url:
        return "News source not found", 404

    df = pd.read_csv(csv_url)
    data_row = df[df['heading'] == heading]

    if data_row.empty:
        return "Heading not found", 404

    article_content = data_row['data'].values[0]
    
    # Convert text to speech
    tts = gTTS(text=article_content, lang='en')
    tts.save("static/audio.mp3")

    return render_template('analysis_news.html', article_content=article_content, audio_file="static/audio.mp3")


if __name__ == "__main__":
    app.run(debug=True)