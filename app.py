import streamlit as st
import hydralit_components as hc
from eda_app import run_eda_app

# Set page configuration
st.set_page_config(
    page_title="Fire Intensity Prediction",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add header
st.markdown("<h1 style='text-align: center; background-color: #000045; color: #ece5f6'>Fire Intensity Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; background-color: #000045; color: #ece5f6'> CPSC 4820 Project by Calaca, Mahmoud, Nyakundi, Sharma </h4>", unsafe_allow_html=True)

# Define menu data
menu_data = [
    {'id': 1, 'label': "Home", 'icon': "fa fa-home"},
    {'id': 2, 'label': "EDA"},
    {'id': 3, 'label': "ML"}
]

# Create the navigation bar
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    hide_streamlit_markers=False,
    sticky_nav=True,
    sticky_mode='pinned',
    override_theme={'menu_background': '#000080'}
)

# Display content based on the selected menu
if menu_id == 1:
    st.subheader("Home")
    st.write("""
        ### Fire Intensity Prediction using Random Forest
        This dataset contains .....
        #### App Content
        - EDA Section: Exploratory Data Analysis of Data
        - ML Section: ML Predictor App
    """)
elif menu_id == 2:
    st.subheader("EDA")
    run_eda_app()
elif menu_id == 3:
    st.subheader("ML")
    # Add your ML code or function here
    #run_ml_app()
    st.write("Machine Learning content goes here.")