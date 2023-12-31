import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import warnings
import os
warnings.filterwarnings("ignore")
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from web_function import  about_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from io import StringIO

df = pd.read_csv("breast_cancer_clean.csv")
x = df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean","compactness_mean", "concavity_mean", "concave points_mean","symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]]  # Include all your features
y = df[['diagnosis']]
df.drop('Unnamed: 0', axis=1, inplace=True)


def load_data():
    # Function to load data
    global df,x,y
    df = pd.read_csv("breast_cancer_clean.csv")
    x = df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean","compactness_mean", "concavity_mean", "concave points_mean","symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]]  # Include all your features
    y = df[['diagnosis']]
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df, x, y

def display_about_datasets(df):
    # Function to display information about datasets
    st.title("About Datasets")
    image_url = "https://img.freepik.com/premium-vector/realistic-human-cell-anatomy-diagram-infographic-poster-vector-illustration_1284-71770.jpg?w=2000"
    st.image(image_url, use_column_width=True)
    image_url_2 = "https://www.sciencefacts.net/wp-content/uploads/2020/06/Nucleus-Structure-Diagram.jpg"
    st.image(image_url_2, use_column_width=True)
    ## See the data that you uploaded
    st.write("<h3 class='alert alert-info' style='border-radius: 10px;text-align:center'>Data Dictionary📖</h3>", unsafe_allow_html=True)
    data_dictionary = """
    <table>
    <thead>
    <tr>
    <th style="color:#338FC2;font-size:20px">Columns</th>
    <th style="color:#338FC2;font-size:20px">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">diagnosis</td>
    <td>1 untuk Malignant, 0 untuk Benign</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">radius_mean</td>
    <td>panjang rata rata dari titik pusat nukleus ke titik luar</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">texture_mean</td>
    <td>di scan dengan metode mammografi. Lalu gambar hasil scan diukur nilai rata rata standar deviasi grey scalenya.</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">perimeter_mean</td>
    <td>rata rata keliling sel</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">area_mean</td>
    <td>rata rata luas area sel yang dideteksi</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">smoothness_mean</td>
    <td>rata rata Kelembutan dihitung berdasarkan variasi lokal dalam panjang radius di dalam inti sel. Semakin mendekati nol, semakin sedikit variasi, dan semakin halus inti selnya. Semakin jauh dari nol, semakin bervariasi dan kurang halus inti selnya</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">compactness</td>
    <td>rata rata Kepadatan (compactness) didefinisikan oleh kemampuan sel untuk saling rapat atau terkemas dengan erat. cara menghitung. keliling^2/luas-1.0</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">concavity_mean</td>
    <td>rata rata ukuran seberapa dalam atau parahnya bagian-bagian cekung pada suatu kontur. Semakin tinggi nilai kekonkavitasan, semakin dalam atau parah cekungan tersebut.(dilihat dengan metode mammografi)</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">concave points_mean</td>
    <td>rata rata jumlah bagian-bagian cekung atau cekungan.</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">symmetry_mean</td>
    <td>rata rata Nilai simetrisnya</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">fractal_dimension_mean</td>
    <td>Dimensi fraktal pada dasarnya adalah ukuran kompleksitas geometris suatu objek.  Penggunaan dimensi fraktal dapat memberikan pemahaman lebih lanjut tentang sejauh mana kompleksitas atau struktur jaringan tersebut.</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">radius_se</td>
    <td>Standar deviasi panjang dari titik pusat nukleus ke titik luar</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">texture_se</td>
    <td>Standar deviasi dari nilai standar deviasi grey scalenya</td>
    </tr>
    <tr>/
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">perimeter_se</td>
    <td>Standar deviasi dari keliling sel</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">area_se</td>
    <td>Standar deviasi dari luas area sel yang dideteksi</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">smoothness_se</td>
    <td>Standar deviasi kelembutan</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">compactness_se</td>
    <td>Standar deviasi Kepadatan</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">concavity_se</td>
    <td>Standar deviasi ukuran seberapa dalam atau parahnya bagian-bagian cekung pada suatu kontur.</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">concave points_se</td>
    <td>Standar deviasi jumlah bagian-bagian cekung atau cekungan</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">symmetry_se</td>
    <td>Standar deviasi Nilai simetrisnya</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">fractal_dimension_se</td>
    <td>Standar deviasi Dimensi fraktal(ukuran kompleksitas geometris suatu objek)</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">radius_worst</td>
    <td>Mengambil nilai terbesar dari panjang dari titik pusat nukleus ke titik luar</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">texture_worst</td>
    <td>Mengambil nilai terbesar dari nilai standar deviasi grey scalenya</td>
    </tr>
    <tr>/
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">perimeter_worst</td>
    <td>Mengambil nilai terbesar dari dari keliling sel</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">area_worst</td>
    <td>Mengambil nilai terbesar dari luas area sel yang dideteksi</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">smoothness_worst</td>
    <td>Mengambil nilai terbesar dari kelembutan</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">compactness_worst</td>
    <td>Mengambil nilai terbesar dari Kepadatan</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">concavity_worst</td>
    <td>Mengambil nilai terbesar dari ukuran seberapa dalam atau parahnya bagian-bagian cekung pada suatu kontur.</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">concave points_worst</td>
    <td>Mengambil nilai terbesar dari jumlah bagian-bagian cekung atau cekungan</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">symmetry_se</td>
    <td>Mengambil nilai terbesar dari Nilai simetrisnya</td>
    </tr>
    <tr>
    <td style="color:#338FC2;font-size:20px;font-family:Talis Heavy">fractal_dimension_worst</td>
    <td>Mengambil nilai terbesar dari Dimensi fraktal(ukuran kompleksitas geometris suatu objek)</td>
    </tr>
    </tbody>
    </table>"""
    st.markdown(data_dictionary, unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    st.write("<h4 style='text-align:center'>Data Examples 🧪", unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    st.write(df.head())
    st.write("<h4 style='text-align:center'>Data Description", unsafe_allow_html=True)
    st.write(df.describe())
    # Showing Data information
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    # Menampilkan Size Data
    st.write("<h4>Data Size", unsafe_allow_html=True)
    st.write(df.shape)
    # Menampilkan Kolom pada datasets
    st.write("<h4>Data Column", unsafe_allow_html=True)
    st.write(df.columns)
    # Menampilkan Tipe Data
    st.write("<h4>Data Types", unsafe_allow_html=True)
    st.write(df.dtypes)
    # Korelasi antara 1 attribute dengan attribute yang lain dalam bentuk tabel
    st.write("<h4 style='text-align:center'>Attribute Correlation ", unsafe_allow_html=True)
    st.write(df.corr(method="pearson"))
    # Korelasi menggunakan Heatmap
    st.write("<h4 style='text-align:center'>Heatmap", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(df.corr(),annot = True, cmap ="Accent_r", ax=ax)
    st.pyplot(fig)
    # Pair Plot untuk melihat persebaran hubungan data antara mean column dengan diagnosis
    st.write("<h4 style='text-align:center'>Pair Plot(Correlation Between Mean Column and diagnosis)", unsafe_allow_html=True)
    mean_col = ['diagnosis','radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

    pair_plot_mean_column = sns.pairplot(df[mean_col],hue = 'diagnosis', palette='cubehelix')
    st.pyplot(pair_plot_mean_column)
    # Pair Plot untuk melihat persebaran hubungan data antara worst column dengan diagnosis
    st.write("<h4 style='text-align:center'>Pair Plot(Correlation Between Worst Column and diagnosis)", unsafe_allow_html=True)
    worst_col = ['diagnosis','radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
    pair_plot_worst_column = sns.pairplot(df[worst_col],hue = 'diagnosis', palette='magma')
    st.pyplot(pair_plot_worst_column)
def train_model():
    global model,x,y
    # Function to train the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    dtc = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='entropy',
            max_depth=4, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=42, splitter='best'
    )

    model = dtc.fit(x_train, y_train)

    # Calculate accuracy scores
    accuracy_train = accuracy_score(y_train, dtc.predict(x_train))
    accuracy_test = accuracy_score(y_test, dtc.predict(x_test))

    score = model.score(x,y)

    # Calculate confusion matrix
    confusion_matrix_result = confusion_matrix(y_test, dtc.predict(x_test))

    print(f"Accuracy on training data: {accuracy_train}")
    print(f"Accuracy on testing data: {accuracy_test}")
    return model, x_test, y_test, accuracy_train, accuracy_test, confusion_matrix_result, score, x, y

with st.sidebar:
    sidebar_selected = option_menu("Main Menu", ["About Datasets", 'Train Data', 'Test Data'], 
        icons=['book', 'gear', 'question'], menu_icon="cast", default_index=0)
if(sidebar_selected=="About Datasets"):
    st.title("About Datasets")
    df,x,y = load_data()
    display_about_datasets(df)
elif(sidebar_selected == "Train Data"):
    st.title("Train Datasets")
    df,x,y = load_data()
    uploaded_file = st.file_uploader("Masukkan Data yang Ingin anda training", type=["csv"])
    if st.button("Upload") and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        x = df.drop(columns='diagnosis')
        y = df['diagnosis']
        df.drop('Unnamed: 0', axis = 1, inplace = True)
        st.markdown('<h3 class="alert alert-info" style="border-radius:10px;text-align:center">Datasets 📚</h3>', unsafe_allow_html=True)
        ## See the data that you uploaded
        st.write(df)
        # Showing Data information
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.markdown('<h3 class="alert alert-info" style="border-radius:10px;text-align:center;">Datasets Info 📖</h3>', unsafe_allow_html=True)
        st.text(s)
        st.write("<h4 style='text-align:center'>Attribute Correlation 📊", unsafe_allow_html=True)
        st.write(df.corr(method="pearson"))
        # Korelasi menggunakan Heatmap
        st.write("<h4 style='text-align:center'>Heatmap", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(25, 15))
        sns.heatmap(df.corr(),annot = True, cmap ="Accent_r", ax=ax)
        st.pyplot(fig)
        st.write("<h4 style='text-align:center'>Pair Plot(Correlation Between Mean Column and diagnosis)", unsafe_allow_html=True)
        mean_col = ['diagnosis','radius_mean', 'texture_mean', 'perimeter_mean',
        'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

        pair_plot_mean_column = sns.pairplot(df[mean_col],hue = 'diagnosis', palette='cubehelix')
        st.pyplot(pair_plot_mean_column)
        # Pair Plot untuk melihat persebaran hubungan data antara worst column dengan diagnosis
        st.write("<h4 style='text-align:center'>Pair Plot(Correlation Between Worst Column and diagnosis)", unsafe_allow_html=True)
        worst_col = ['diagnosis','radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst']
        pair_plot_worst_column = sns.pairplot(df[worst_col],hue = 'diagnosis', palette='magma')
        st.pyplot(pair_plot_worst_column)
        st.write("<h4 style='text-align:center'>Report", unsafe_allow_html=True)
        # x = df.drop(columns='diagnosis')
        # y = df['diagnosis']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Create and train the Decision Tree Classifier
        dtc = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='entropy',
            max_depth=4, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=42, splitter='best'
        )

        model = dtc.fit(x_train, y_train)

        # Calculate accuracy scores
        accuracy_train = accuracy_score(y_train, dtc.predict(x_train))
        accuracy_test = accuracy_score(y_test, dtc.predict(x_test))

        # Calculate confusion matrix
        confusion_matrix_result = confusion_matrix(y_test, dtc.predict(x_test))

        # Display heatmap for confusion matrix
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix_result, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Sebenarnya")
        st.pyplot(fig)

        # Display classification report
        classification_report_result = classification_report(y_test, dtc.predict(x_test), output_dict=True)
        st.write("\nClassification Report:")
        st.table(classification_report_result)
        score = model.score(x,y)
        # Display accuracy scores in Streamlit
        st.write(f"Akurasi data training = {accuracy_train}")
        st.write(f"Akurasi data testing = {accuracy_test}")
elif(sidebar_selected == "Test Data"):
    st.title("Predictions")
    if x is None or y is None:
        st.warning("Please upload and train the model before making predictions.")
    else:
        radius_mean = st.number_input("radius_mean", value=None, placeholder="Type a number...")
        st.write('The current raidus_mean is ', radius_mean)
        texture_mean = st.number_input("texture_mean", value=None, placeholder="Type a number...")
        st.write('The current texture_mean is ', texture_mean)
        perimeter_mean = st.number_input("perimeter_mean", value=None, placeholder="Type a number...")
        st.write('The current perimeter_mean is ', perimeter_mean)
        area_mean = st.number_input("area_mean", value=None, placeholder="Type a number...")
        st.write('The current area_mean is ', area_mean)
        smoothness_mean = st.number_input("smoothness_mean", value=None, placeholder="Type a number...")
        st.write('The current smoothness_mean is ', smoothness_mean)
        compactness_mean = st.number_input("compactness_mean", value=None, placeholder="Type a number...")
        st.write('The current compactness_mean is ', compactness_mean)
        concavity_mean = st.number_input("concavity_mean", value=None, placeholder="Type a number...")
        st.write('The current concavity_mean is ', concavity_mean)
        concave_points_mean = st.number_input("concave_points_mean", value=None, placeholder="Type a number...")
        st.write('The current concave_points_mean is ', concave_points_mean)
        symmetry_mean = st.number_input("symmetry_mean", value=None, placeholder="Type a number...")
        st.write('The current symmetry_mean is ', symmetry_mean)
        fractal_dimension_mean = st.number_input("fractal_dimension_mean", value=None, placeholder="Type a number...")
        st.write('The current fractal_dimension_mean is ', fractal_dimension_mean)

        radius_se = st.number_input("radius_se", value=None, placeholder="Type a number...")
        st.write('The current raidus_se is ', radius_se)
        texture_se = st.number_input("texture_se", value=None, placeholder="Type a number...")
        st.write('The current texture_se is ', texture_se)
        perimeter_se = st.number_input("perimeter_se", value=None, placeholder="Type a number...")
        st.write('The current perimeter_se is ', perimeter_se)
        area_se = st.number_input("area_se", value=None, placeholder="Type a number...")
        st.write('The current area_se is ', area_se)
        smoothness_se = st.number_input("smoothness_se", value=None, placeholder="Type a number...")
        st.write('The current smoothness_se is ', smoothness_se)
        compactness_se = st.number_input("compactness_se", value=None, placeholder="Type a number...")
        st.write('The current compactness_se is ', compactness_se)
        concavity_se = st.number_input("concavity_se", value=None, placeholder="Type a number...")
        st.write('The current concavity_se is ', concavity_se)
        concave_points_se = st.number_input("concave_points_se", value=None, placeholder="Type a number...")
        st.write('The current concave_points_se is ', concave_points_se)
        symmetry_se = st.number_input("symmetry_se", value=None, placeholder="Type a number...")
        st.write('The current symmetry_se is ', symmetry_se)
        fractal_dimension_se = st.number_input("fractal_dimension_se", value=None, placeholder="Type a number...")
        st.write('The current fractal_dimension_se is ', fractal_dimension_se)

        radius_worst = st.number_input("radius_worst", value=None, placeholder="Type a number...")
        st.write('The current raidus_worst is ', radius_worst)
        texture_worst = st.number_input("texture_worst", value=None, placeholder="Type a number...")
        st.write('The current texture_worst is ', texture_worst)
        perimeter_worst = st.number_input("perimeter_worst", value=None, placeholder="Type a number...")
        st.write('The current perimeter_worst is ', perimeter_worst)
        area_worst = st.number_input("area_worst", value=None, placeholder="Type a number...")
        st.write('The current area_worst is ', area_worst)
        smoothness_worst = st.number_input("smoothness_worst", value=None, placeholder="Type a number...")
        st.write('The current smoothness_worst is ', smoothness_worst)
        compactness_worst = st.number_input("compactness_worst", value=None, placeholder="Type a number...")
        st.write('The current compactness_worst is ', compactness_worst)
        concavity_worst = st.number_input("concavity_worst", value=None, placeholder="Type a number...")
        st.write('The current concavity_worst is ', concavity_worst)
        concave_points_worst = st.number_input("concave_points_worst", value=None, placeholder="Type a number...")
        st.write('The current concave_points_worst is ', concave_points_worst)
        symmetry_worst = st.number_input("symmetry_worst", value=None, placeholder="Type a number...")
        st.write('The current symmetry_worst is ', symmetry_worst)
        fractal_dimension_worst = st.number_input("fractal_dimension_worst", value=None, placeholder="Type a number...")
        st.write('The current fractal_dimension_worst is ', fractal_dimension_worst)

        features = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,compactness_mean, concavity_mean, concave_points_mean,symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]
        

    if st.button("Prediksi"):
        model, x_test, y_test, accuracy_train, accuracy_test, confusion_matrix_result, score, x, y = train_model()
        score = score
        input_data_as_numpy_array = np.array(features)
        input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
        if model is not None:      
            prediction = model.predict(input_data_reshape)
            st.info("Prediksi Sukses")
            if(prediction == 1):
                st.warning("Pasien Terkena Penyakit Kanker Payudara \"Ganas")
            else:
                st.success("Pasien Terkena Penyakit Kanker Payudara \"Jinak")


            st.write("Model yang digunakan memiliki tingkat akurasi", (score*100),"%")
        else:
            st.warning("Please Train the model before making predictions")

     
        

