import functions_framework
import pandas as pd
from google.cloud import storage
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import cosine_similarity
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#Return data
caregiver_return = pd.read_csv("database_populating_caregiver.csv", usecols=lambda x: x not in ['Matching'])
kolom_layanan = ['Mengurus rumah', 'Membantu pergerakan dan aktivitas fisik', 'Membantu konsumsi obat dan makanan',
                 'Mengecek kesehatan rutin secara mandiri', 'Mendampingi dan menjaga', 'Memasangkan alat medis khusus',
                 'Memeriksakan rutin ke dokter']
kolom_bahasa = ['Indonesia', 'Inggris', 'Jawa', 'Sunda', 'Melayu']

# Numeric
caregiver_num = pd.read_csv("database_populating_caregiver.csv", usecols=lambda x: x not in ['id', 'Matching', 'Kota', 'Nama', 'Nomor Whatssap'])

# City
prov_kota = pd.read_csv("Prov_Kota.csv")
# Membuat kolom baru dengan penggabungan province dan city
# prov_kota['concat'] = prov_kota.apply(lambda row: row['kota'] + ' ' + row['provinsi'], axis=1)
kota_mapping = prov_kota.set_index('kota')['provinsi'].to_dict()

caregiver_city = pd.read_csv("database_populating_caregiver.csv", usecols= ['Kota'])
caregiver_city['Provinsi'] = caregiver_city['Kota'].map(kota_mapping)
# caregiver_city['concat'] = caregiver_city.apply(lambda row: row['Kota'] + ' ' + row['Provinsi'], axis=1)

@functions_framework.http
def hello_http(request):
    """## Get Input"""

    input_data = request.get_json()

    lansia_vec = np.array([input_data["Mobilitas"], input_data["Penyakitlain"], input_data["Hipertensi"],
                        input_data["Diabetes"], input_data["Reumatik"], input_data["Penyakitjantung"], input_data["Asma"],
                        input_data["Stroke"], input_data["Mengurusrumah"], input_data["Membantupergerakandanaktivitasfisik"],
                        input_data["Membantukonsumsiobatdanmakanan"], input_data["Mengecekkesehatanrutinsecaramandiri"],
                        input_data["Mendampingidanmenjaga"], input_data["Memasangkanalatmediskhusus"],
                        input_data["Memeriksakanrutinkedokter"], input_data["Indonesia"], input_data["Inggris"], input_data["Jawa"],
                        input_data["Sunda"], input_data["Melayu"]])

    # Contoh Input
    new_lansia_city = input_data["Kota"]
    new_lansia_city = new_lansia_city.upper()

    #Numeric
    scaler = StandardScaler()
    caregiver_num_scaled = scaler.fit_transform(caregiver_num)

    caregiver_vecs = np.array(caregiver_num_scaled)
    scaled_item_vecs = scaler.fit_transform(caregiver_vecs)
    
    lansia_vecs = [lansia_vec] * len(caregiver_vecs)
    scaled_user_vecs = scaler.fit_transform(lansia_vecs)

    # Model
    global model_num

    if 'model_num' not in globals():
        storage_client = storage.Client()
        bucket = storage_client.get_bucket('caregiver-recommendation-update')
        blob_model = bucket.blob('Model-Development/Newest-Model/caregiver_model.h5')
        blob_model.download_to_filename("/tmp/caregiver_model.h5")
        model_num = load_model("/tmp/caregiver_model.h5")

    predictions = model_num.predict([scaled_user_vecs, scaled_item_vecs])

    #City
    list_caregiver_city = caregiver_city["Provinsi"].tolist()
    list_caregiver_city = [kota.lower() for kota in list_caregiver_city]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(prov_kota["provinsi"])
    vocab_size = len(tokenizer.word_index) + 1

    city_seq = tokenizer.texts_to_sequences(prov_kota["provinsi"])
    max_seq_length = max(len(seq) for seq in city_seq)
    
    province = kota_mapping.get(new_lansia_city)
    # new_lansia_city = new_lansia_city + " " + province
    # new_lansia_city = new_lansia_city.lower()
    new_lansia_city = province.lower()

    for_similarity = list_caregiver_city
    for_similarity.append(new_lansia_city)

    for_similarity_seq = tokenizer.texts_to_sequences(for_similarity)
    for_similarity_input = pad_sequences(for_similarity_seq, maxlen=max_seq_length, padding="post")

    # Model
    global model_city

    if 'model_city' not in globals():
        storage_client = storage.Client()
        bucket = storage_client.get_bucket('caregiver-recommendation-update')
        blob_model = bucket.blob('Model-Development/Newest-Model/city_model.h5')
        blob_model.download_to_filename("/tmp/city_model.h5")
        model_city = load_model("/tmp/city_model.h5")

    #Calculate city similarity
    cost_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model_city.compile(optimizer=opt,loss=cost_fn)
    embeddings = model_city.predict(for_similarity_input)

    final_score = []
    for i in range(len(embeddings) - 1):
        const = tf.keras.losses.cosine_similarity(embeddings[-1], embeddings[i], axis=-1)
        skor = predictions[i] * -const.numpy()
        final_score.append(skor)

    sorted_index = np.argsort(-np.array(final_score), axis=0).reshape(-1)  # dimulai dari 0
    ids = [int(idx + 1) for idx in sorted_index]  # plus 1 karena id caregiver tidak dimulai dari 0

    result = []
    for idx in ids[:50]:
        user = caregiver_return[caregiver_return['id'] == idx]
        layanan = ""
        layanan = ", ".join([kolom for kolom in kolom_layanan if user[kolom].values == 1])
        bahasa = ""
        bahasa = ", ".join([kolom for kolom in kolom_bahasa if user[kolom].values == 1])

        user_info = {
            "id": idx,
            "nama": user["Nama"].values.tolist()[0],
            "nomor": user["Nomor Whatssap"].values.tolist()[0],
            "gender": user["Jenis kelamin"].values.tolist()[0],
            "pendidikan": user["Pendidikan terakhir"].values.tolist()[0],
            "tahun_pengalaman": user["Lama pengalaman (tahun)"].values.tolist()[0],
            "jumlah_lansia_pernah_dirawat": user["Jumlah lansia pernah dirawat"].values.tolist()[0],
            "umur": user["Umur"].values.tolist()[0],
            "layanan": layanan,
            "bahasa": bahasa,
            "kota": user["Kota"].values.tolist()[0],
            "gaji": user["Gaji"].values.tolist()[0]
        }
        result.append(user_info)

    return {
        "result": result
    }