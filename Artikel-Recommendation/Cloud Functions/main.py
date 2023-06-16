import functions_framework
import pandas as pd
from google.cloud import storage
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import string
import regex
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# import stopword remover class
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Preprocessing article form dataset, input data
def preprocess_article(text):
    #Case folding
    case_folding = text.apply(lambda x: x.lower())
    #Number Removal
    num_removal = case_folding.apply(lambda x: regex.sub(r'\d+', '', x))
    #String punctuation
    string_punc = num_removal.apply(lambda x: x.translate(str.maketrans("","",string.punctuation)))
    #Remove lansia word
    lansia_remover = string_punc.apply(lambda x: x.replace('lansia', ''))
    #whitespace removing
    whitespace_remover = lansia_remover.apply(lambda x: x.strip())
    #remove stopword
    #create stopword removal
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    stopword_remover = factory.create_stop_word_remover()
    stopword_removal =  whitespace_remover.apply(lambda x: stopword_remover.remove(x))
    #text stemming
    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    clean_data = stopword_removal.apply(lambda x: stemmer.stem(x))
    #return clean text
    return clean_data  

# Tidak masuk fungsi
# Read CSV file
df = pd.read_csv('article_dataset.csv')
column = df['judul']
clean_data = preprocess_article(column)
article_title = clean_data.tolist()
# Inisialisasi  Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(article_title)
# Ubah judul artikel menjadi urutan angka
sequences = tokenizer.texts_to_sequences(article_title)
# Tentukan panjang maksimum 
max_length = 14
# Padding urutan agar memiliki panjang yang sama
padded_article_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

@functions_framework.http
def hello_http(request):
    #Custom function
    # Preprocessing input text, input string
    def preprocess_input(text):
        #Case folding
        case_folding = text.lower()
        #Number Removal
        num_removal = regex.sub(r'\d+', '', case_folding)
        #String punctuation
        string_punc = num_removal.translate(str.maketrans("","",string.punctuation))
        #Remove lansia word
        lansia_remover = string_punc.replace('lansia', '')
        #whitespace removing
        whitespace_remover = lansia_remover.strip()
        #remove stopword
        #create stopword removal
        factory = StopWordRemoverFactory()
        stopword_remover = factory.create_stop_word_remover()
        stopword_removal = stopword_remover.remove(whitespace_remover)
        #text stemming
        # create stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        clean_text = stemmer.stem(stopword_removal)
        #return clean text
        return clean_text

    # Get Input
    input_data = request.get_json()

    # Input article title
    title = input_data["judul"]
    preprocessed_input = preprocess_input(title)
    # Convert it to list
    title_list = []
    title_list.append(preprocessed_input)
    
    # Ubah judul input menjadi sequences
    input_sequence = tokenizer.texts_to_sequences(title_list)
    input_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post')

    
    # Model
    global my_model
    if 'my_model' not in globals():
        storage_client = storage.Client()
        bucket = storage_client.get_bucket('article-recommendation-update')
        blob_model = bucket.blob('Artikel-Recommendation/Model Dataset Without Content/my_model.h5')
        blob_model.download_to_filename("/tmp/my_model.h5")
        my_model = load_model("/tmp/my_model.h5")

    input_vector = my_model.predict(input_padded)
    
     # Hitung cosine similiarity 
    similarities = []
    for article_vector in my_model.predict(padded_article_sequences):
        similarity = cosine_similarity(input_vector, article_vector.reshape(1, -1))
        similarities.append(similarity[0][0])
        
    # Urutkan indeks berdasarkan skor kemiripan
    ranked_indices = np.argsort(similarities)[::-1]

    ids = [int(idx) for idx in ranked_indices]

    results = []
    
    # Jadikan list dalam list
    for idx in ids:
        article_id = idx
        article_title = df['judul'][idx]
        article_content = df['konten'][idx]

        # Tambahkan id, judul, dan konten ke dalam dictionary
        result = {
            "id": article_id,
            "judul": article_title,
            "konten": article_content
        }

        # Tambahkan dictionary ke dalam list
        results.append(result)

    #return index dari artikel
    return {
        "result": results
    }
    







    
   

   



