import streamlit as st

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

from sklearn import metrics

st.set_page_config(
    page_title="Project"
)

tab1, tab2, tab3 = st.tabs(["Dataset","Prepocessing","Implementation"])

with tab1:
    st.title("Penambangan Data ")
    st.write("### Dosen Pengampu : Mula'ab, S.Si., M.Kom.")
    st.write("##### Kelompok ")
    st.write("##### Fadetul Fitriyeh- 200411100189 ")
    st.write("##### R.Bella Aprilia Damayanti - 200411100082")
    st.write('Dataset yang digunakan yaitu dataset XL AXIATA yang diambil dari yahoo.com')
    

    df = pd.read_csv("https://raw.githubusercontent.com/RBellaApriliaDamayanti22/Datasets/main/XL-axiata.csv")
    st.write("Dataset https://finance.yahoo.com/quote/PTXKY/history?p=PTXKY: ")
    st.write(df)

with tab2:
    st.write("Data preprocessing adalah proses yang mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini penting dilakukan karena data mentah sering kali tidak memiliki format yang teratur. Selain itu, data mining juga tidak dapat memproses data mentah, sehingga proses ini sangat penting dilakukan untuk mempermudah proses berikutnya, yakni analisis data.")
    st.write("Data preprocessing adalah proses yang penting dilakukan guna mempermudah proses analisis data. Proses ini dapat menyeleksi data dari berbagai sumber dan menyeragamkan formatnya ke dalam satu set data.")
    
    scaler = st.radio(
    "Pilih Metode Normalisasi Data : ",
    ('Tanpa Scaler', 'MinMax Scaler'))
    if scaler == 'Tanpa Scaler':
        st.write("Dataset Tanpa Preprocessing : ")
        df_new=df
    elif scaler == 'MinMax Scaler':
        st.write("Dataset setelah Preprocessing dengan MinMax Scaler: ")
        scaler = MinMaxScaler()
        df_drop_column_for_minmaxscaler=df.drop(['Date','Open','High','Low','Close','Adj Close'], axis=1)
        df_scaler = scaler.fit_transform(df_drop_column_for_minmaxscaler)
        df_new = df_scaler
        
    st.write(df_new)

with tab3:
    st.write("""
    <h5>Implementation Naive Bayes</h5>
    <br>
    """, unsafe_allow_html=True)
    col1,col2 = st.columns([2,2])
    with col1:
        a = st.number_input("Volume",0)
        
    submit = st.button('Prediksi')
    if submit:
        scaler = MinMaxScaler()
        df_drop_column_for_minmaxscaler=df.drop(['Date','Open','High','Low','Close','Adj Close'], axis=1)
        df_scaler = scaler.fit_transform(df_drop_column_for_minmaxscaler)

        #Train test split
        training, test = train_test_split(df_scaler,test_size=0.3, random_state=1)#Nilai X training dan Nilai X testing
        training_label, test_label = train_test_split(df['Volume'], test_size=0.3, random_state=1)#Nilai Y training dan Nilai Y testing    

        # model
        gaussian = GaussianNB()
        clf = gaussian.fit(training, training_label)
        y_pred = clf.predict(test)

        #Evaluasi
        akurasi = accuracy_score(test_label, y_pred)

        #Inputan 
        inputs = np.array([
                    a
                ])
        df_min = df_scaler.min()
        df_max = df_scaler.max()
        input_norm = ((inputs - df_min) / (df_max - df_min))
        input_norm = np.array(input_norm).reshape(1, -1)

        input_pred = clf.predict(input_norm)

        st.subheader('Hasil Akurasi')
        st.info(akurasi)
        st.subheader('Hasil Prediksi')
        st.write(input_pred)

        if input_pred == 0 :
            st.write('Tidak Memenuhi ')
        else:
            st.write('Memenuhi')

        # st.subheader('Preprocessing')
        # st.write("Case Folding :",lower_case_isi)
        # st.write("Cleansing :",clean_symbols)
        # st.write("Slang Word :",slang)
        # st.write("Steaming :",stem)

        # st.subheader('Akurasi')
        # st.info(akurasi)

        # st.subheader('Prediksi')
        # if y_preds == "Positif":
        #     st.success('Positive')
        # else:
        #     st.error('Negative')