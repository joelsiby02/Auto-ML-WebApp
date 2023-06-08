# Import Deps
import streamlit as st
import pandas as pd
import os

# EDA
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

# ML
from pycaret.classification import setup, compare_models, pull, save_model




with st.sidebar:
    st.image('https://github.com/joelsiby02/Auto-ML-WebApp/blob/1c7e3c821d7e98cd3f63fbbd5ea59ef30c3b2085/Logo.jpg') # Image for sidebar
    st.title('Auto ML App') # title for SideBar
    choice = st.radio("User Tools",['Upload', 'Profiling', 'ML', "Download"] ) # Controling navigations
    st.info("This application allows you to built an Automated ML pipeline using Streamlit, Pandas Profiling & PyCaret") # Info to describe


if os.path.exists("SourceData.csv"):
    df = pd.read_csv("SourceData.csv", index_col = None )

# Defining choice Struchures

if choice == "Upload":
    st.title("Upload your Data for Modeling!")
    file = st.file_uploader("Upload your Data Set here")
    
    if file:
        df = pd.read_csv(file, index_col = None)  # reading the DataSet 
        df.to_csv("SourceData.csv", index = None)
        st.dataframe(df) # Render the dataFrame 


if choice == "Profiling":
    st.title("Exploratory Data Analysis Report")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
    pass


if choice == 'ML':
    st.title("Machine Learning on Progress")
    target = st.selectbox("Select Your Target", df.columns)
    if st.button("Train Model"):
        setup(df, target = target, silent = True)
        setup_df = pull()
        st.info("Experementing ML Settings")
        st.dataframe(setup_df)
        
        # Selecting models using compare_model & pull using pycaret fn's
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Models")
        st.dataframe(compare_df)
        best_model
        
        # saveModel
        save_model(best_model, 'best_model')


if choice == 'Download':
    with open("Best_Model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "Trained_Model.pkl")
    pass
