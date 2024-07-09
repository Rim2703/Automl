import streamlit as st 
import pandas as pd 
import os 

import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import setup,compare_models,pull,save_model,load_model

with st.sidebar:
    st.image("https://i.pinimg.com/736x/c0/4f/59/c04f597f296d7a4e8a99ca2335541940.jpg")
    st.title("AutoML Streaming")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling and PyCaret.")

if os.path.exists("soucedata.csv"):
    df = pd.read_csv("soucedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for Modelling... ")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("soucedata.csv",index=None)
        st.dataframe(df)


if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile = ydata_profiling.ProfileReport(df, explorative=True)
    st_profile_report(profile)


if choice == "ML":
    st.title("Best Model Analysis")
    chosen_target = st.selectbox("Select your Target", df.columns)
    if st.button("Train model"):
        setup(df,target=chosen_target)
        setup_df = pull()
        st.info("This is the best model analysis")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the Automl model comparision")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,"best_model") 


if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download trained model", f, file_name="trained_model.pkl")