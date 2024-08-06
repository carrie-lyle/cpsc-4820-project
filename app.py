import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Fire Intensity Prediction",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from eda_app import run_eda_app
from ml_app import run_ml_app

# Add header
st.markdown("<h1 style='text-align: center; background-color: #000045; color: #ece5f6'>Fire Intensity Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; background-color: #000045; color: #ece5f6'> CPSC 4820 Project by Calaca, Mahmoud, Nyakundi, Sharma </h4>", unsafe_allow_html=True)

# Define menu data
menu_data = [
    'Home',
    'EDA',
    'Visualizations',
    'Machine Learning'
]

# Create the navigation bar
menu_id = st.sidebar.selectbox("Navigation", menu_data)

# Display content based on the selected menu
if menu_id == 'Home':
    st.subheader("Home")
    st.write("""
        ### Fire Intensity Prediction using Random Forest
        This dataset contains .....
        #### App Content
        - EDA Section: Exploratory Data Analysis of Data
        - Visualizations Section: Visualization of Data
        - Machine Learning Section
    """)
elif menu_id == 'EDA':
    st.subheader("EDA")
    run_eda_app()
elif menu_id == 'Visualizations':
    st.subheader("Visualizations")
    run_eda_app()
elif menu_id == 'Machine Learning':
    st.subheader("Machine Learning Section")
    run_ml_app()
