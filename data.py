import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.title("Penambangan Data")
st.write("### Dosen Pengampu: Mula'ab, S.Si., M.Kom.")
st.write("##### Kelompok")
st.write("##### Fadetul Fitriyeh- 200411100189")
st.write("##### R.Bella Aprilia Damayanti - 200411100082")
st.write('Dataset yang digunakan yaitu dataset XL AXIATA yang diambil dari yahoo.com')

data_set_description, upload_data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Preprocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link :  https://github.com/RBellaApriliaDamayanti22/uas")
    st.write("###### Untuk Wa saya anda bisa hubungi nomer ini : http://wa.me/6289658567766 ")

with upload_data:
    df = pd.read_csv('https://raw.githubusercontent.com/RBellaApriliaDamayanti22/Datasets/main/XL-axiata.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    df = df.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close'])
    X = df.drop(columns=['Volume'])
    y = df['Volume'].values

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(scaled_features, columns=X.columns)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_df)

with modeling:
    training, test, training_label, test_label = train_test_split(scaled_df, y, test_size=0.2, random_state=1)

    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decision Tree')
        submitted = st.form_submit_button("Submit")

        if naive:
            gaussian = GaussianNB()
            gaussian.fit(training, training_label)
            y_pred = gaussian.predict(test)
            gaussian_accuracy = round(100 * accuracy_score(test_label, y_pred), 2)
            st.write('Model Gaussian Naive Bayes accuracy score:', gaussian_accuracy)

        if k_nn:
            K = 10
            knn = KNeighborsClassifier(n_neighbors=K)
            knn.fit(training, training_label)
            knn_pred = knn.predict(test)
            knn_accuracy = round(100 * accuracy_score(test_label, knn_pred), 2)
            st.write("Model K-Nearest Neighbors accuracy score:", knn_accuracy)

        if destree:
            dt = DecisionTreeClassifier()
            dt.fit(training, training_label)
            dt_pred = dt.predict(test)
            dt_accuracy = round(100 * accuracy_score(test_label, dt_pred), 2)
            st.write("Model Decision Tree accuracy score:", dt_accuracy)

with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        volume = st.number_input('Volume:')
        model = st.selectbox('Pilih model untuk prediksi:',
                            ('Gaussian Naive Bayes', 'K-Nearest Neighbors', 'Decision Tree'))
        submitted = st.form_submit_button("Submit")

        if submitted:
            input_data = np.array([[volume]])
            scaler = MinMaxScaler()
            scaled_input = scaler.fit_transform(input_data)
            
            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            elif model == 'K-Nearest Neighbors':
                mod = knn
            elif model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(scaled_input)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Model:', model)
            st.write('Volume:', input_pred[0])
