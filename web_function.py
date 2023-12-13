import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from web_module import train_model



def about_data():
    df = pd.read_csv("breast_cancer_clean.csv")
    x = df[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]]
    y = df[['diagnosis']]
    df.drop('Unnamed: 0', axis = 1, inplace = True)
    # Showing Image
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
    # ind_col is x-axis
    #dep_col is y-axis or label
    ind_col = [col for col in df.columns if col != 'diagnosis']
    dep_col = 'diagnosis'
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


def load_data():
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

        # Display accuracy scores in Streamlit
        st.write(f"Akurasi data training = {accuracy_train}")
        st.write(f"Akurasi data testing = {accuracy_test}")

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
        return model

def train_model(x,y):
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion='entropy',
        max_depth=4, max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        random_state=42, splitter='best'
    )
    model.fit(x,y)
    score = model.score(x,y)
    return model,score

