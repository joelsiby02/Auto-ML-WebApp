import streamlit as st
from pycaret.classification import load_model

pipeline = load_model("Trained_Model.pkl")
pipeline