# Import Deps
import streamlit as st
import pandas as pd
import os

# EDA
# from ydata_profiling import profile_report
import pyarrow.feather as feather
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# ML
from pycaret.classification import setup, compare_models, pull, save_model




with st.sidebar:
    st.image('Logo.jpg') # Image for sidebar
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
        df = pd.read_csv(file, index_col=None, encoding='latin1')  # Specify the appropriate encoding
        feather.write_feather(df, "SourceData.feather")  # Save DataFrame to Feather format
        st.dataframe(df)  # Render the DataFrame
 


if choice == "Profiling":
    st.title("Exploratory Data Analysis Report")
    profile = ProfileReport(df)
    st_profile_report(profile)


if choice == 'ML':
    st.title("Machine Learning on Progress")
    target = st.selectbox("Select Your Target", df.columns)
    if st.button("Train Model"):
        setup(df, target = target) #  silent = True
        setup_df = pull()
        st.info("Experementing ML Settings")
        st.dataframe(setup_df)
        
        # Selecting models using compare_model & pull using pycaret fn's
        best_model = compare_models()
        compare_df = pull()
        st.info("These are the trained Models")
        st.dataframe(compare_df)
        best_model
        
        # saveModel
        save_model(best_model, 'best_model')


if choice == 'Download':
    with open("Best_Model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "Trained_Model.pkl")
        
