import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Attribute information for display (you can customize this as needed)
attrib_info = """
#### Attribute Information:
    - temp: Temperature (-18.74 to 43.88)
    - rh: Relative Humidity (11 to 97)
    - ws: Wind Speed (2.32 to 33.71)
    - wd: Wind Direction (0 to 360)
    - pcp: Precipitation (0 to 651.79)
    - ros: Rate of Spread (0 to 100)
    - cfb: Crown Fraction Burned (0 to 100)
    - elev: Elevation (0 to 1122.11)
    - sfl: Surface Fuel Load (0 to 100)
    - cfl: Canopy Fuel Load (0 to 100)
"""

# Load ML Models
@st.cache(allow_output_mutation=True)
def load_model(model_file):
    loaded_model = joblib.load(model_file)
    return loaded_model

def run_ml_app():
    st.subheader("Machine Learning Section")

    # Dropdown menu for algorithm selection
    algorithm = st.selectbox(
        "Select the Algorithm",
        ("Random Forest", "Decision Tree")
    )

    if algorithm == "Random Forest":
        loaded_model = load_model("random_forest_model.pkl")
        model_accuracy = 81.00
    else:
        loaded_model = load_model("decision_tree_model.pkl")
        model_accuracy = 75.00

    with st.expander("Attributes Info"):
        st.markdown(attrib_info, unsafe_allow_html=True)

    # Layout for user inputs
    col1, col2 = st.columns(2)

    with col1:
        temp = st.slider("Temperature (-18.74 to 43.88)", min_value=-18.74, max_value=43.88, value=21.78, step=0.01)
        rh = st.slider("Relative Humidity (11 to 97)", min_value=11.0, max_value=97.0, value=35.48, step=0.01)
        ws = st.slider("Wind Speed (2.32 to 33.71)", min_value=2.32, max_value=33.71, value=9.48, step=0.01)
        wd = st.slider("Wind Direction (0 to 360)", min_value=0.0, max_value=360.0, value=200.0, step=1.0)
        pcp = st.slider("Precipitation (0 to 651.79)", min_value=0.0, max_value=651.79, value=0.21, step=0.01)

    with col2:
        ros = st.slider("Rate of Spread (0 to 100)", min_value=0.0, max_value=100.0, value=0.0, step=0.01)
        cfb = st.slider("Crown Fraction Burned (0 to 100)", min_value=0.0, max_value=100.0, value=0.0, step=0.01)
        elev = st.slider("Elevation (0 to 1122.11)", min_value=0.0, max_value=1122.11, value=528.30, step=0.01)
        sfl = st.slider("Surface Fuel Load (0 to 100)", min_value=0.0, max_value=100.0, value=0.0, step=0.01)
        cfl = st.slider("Canopy Fuel Load (0 to 100)", min_value=0.0, max_value=100.0, value=0.0, step=0.01)

    # Collect inputs
    input_data = {
        'temp': temp,
        'rh': rh,
        'ws': ws,
        'wd': wd,
        'pcp': pcp,
        'ros': ros,
        'cfb': cfb,
        'elev': elev,
        'sfl': sfl,
        'cfl': cfl
    }

    # Display the collected inputs
    with st.expander("Your Selected Options"):
        st.write(input_data)

    # Convert inputs to DataFrame for the model
    input_df = pd.DataFrame([input_data])

    # Add a button to trigger prediction
    if st.button("Predict"):
        # Make predictions
        with st.expander("Prediction Results"):
            prediction = loaded_model.predict(input_df)
            pred_prob = loaded_model.predict_proba(input_df)

            st.write(f"Prediction: {prediction[0]}")
            pred_probability_score = {f"Class {i+1}": pred_prob[0][i] * 100 for i in range(len(pred_prob[0]))}
            st.subheader("Prediction Probability Score")
            st.json(pred_probability_score)

        # Display model accuracy
        with st.expander("Model Accuracy"):
            st.write(f"Model Accuracy: {model_accuracy}%")

# Run the app
if __name__ == '__main__':
    run_ml_app()
