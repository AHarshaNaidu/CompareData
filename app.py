import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Train vs Test Dataset Comparison")

# File upload
train_file = st.file_uploader("Upload Train Dataset", type=["csv", "xlsx"])
test_file = st.file_uploader("Upload Test Dataset", type=["csv", "xlsx"])

if train_file and test_file:
    # Load datasets
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Display basic info
    st.header("Basic Info")
    st.write("Train Dataset:")
    st.write(train.describe())
    st.write("Test Dataset:")
    st.write(test.describe())

    # Visualize distributions
    st.header("Distributions")
    feature = st.selectbox("Select a feature to compare", train.columns.intersection(test.columns))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train[feature], kde=True, ax=ax[0], color='blue').set_title("Train")
    sns.histplot(test[feature], kde=True, ax=ax[1], color='orange').set_title("Test")
    st.pyplot(fig)

    # Missing values
    st.header("Missing Values")
    st.write("Train Missing Values:")
    st.write(train.isnull().sum() / len(train) * 100)
    st.write("Test Missing Values:")
    st.write(test.isnull().sum() / len(test) * 100)
