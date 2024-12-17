import os
import re
import string
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import nltk

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('indonesian'))
load_dotenv()
try:
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT"))
    )
    cursor = conn.cursor()
except mysql.connector.Error as err:
    print(f"Error: {err}")
    exit(1)
cursor.execute("SELECT id, judul FROM judul_skripsi")

data = cursor.fetchall()
data_ids = [row[0] for row in data]
data_judul = [row[1] for row in data]
data_judul = list(set(data_judul))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

data_judul = [preprocess_text(judul) for judul in data_judul]
data_latih, data_uji = train_test_split(data_judul, test_size=0.9, random_state=42)

def create_labels(data):
    return [1 if len(judul.split()) > 3 else 0 for judul in data]

label_latih = create_labels(data_latih)
label_sebenarnya = create_labels(data_uji)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
tfidf_matrix_latih = vectorizer.fit_transform(data_latih)
tfidf_matrix_uji = vectorizer.transform(data_uji)
threshold = 0.5
similarity_matrix = cosine_similarity(tfidf_matrix_uji, tfidf_matrix_latih)

prediksi = [
    label_latih[np.argmax(sim)] if np.max(sim) > threshold else 0
    for sim in similarity_matrix
]

correct_predictions = sum(1 for true, pred in zip(label_sebenarnya, prediksi) if true == pred)
total_predictions = len(label_sebenarnya)
akurasi = (correct_predictions / total_predictions) * 100

print("Classification Report:")
report = classification_report(label_sebenarnya, prediksi, target_names=["Pendek", "Panjang"])
print(report)
print(f"Akurasi: {akurasi:.2f}%")
cursor.close()
conn.close()
