import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data():
    uploaded_file = st.file_uploader("Masukkan Data yang Ingin anda training", type=["csv"])
    if st.button("Upload") and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        x = df.drop(columns='diagnosis')
        y = df['diagnosis']
        df.drop('Unnamed: 0', axis = 1, inplace = True)
        return df,x,y

@st.cache_data()
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

def predict(x,y,features):
    model, score = train_model(x,y)

    prediction = model.predict(np.array(features).reshape(1,-1))

    return prediction, score