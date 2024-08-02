import streamlit as st
import pandas as pd

# Data Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px

@st.cache_data
def load_data(data):
    df = pd.read_csv(data)
    return df

def run_eda_app():
    st.subheader("EDA Section")
    sample_df = load_data("data/sampled_hotspots.csv")

    submenu = st.sidebar.selectbox("SubMenu", ["Descriptive", "Plots"])
    if submenu == "Descriptive":
        st.dataframe(sample_df)

        with st.expander("Data Types Summary"):
            st.dataframe(sample_df.dtypes)

        with st.expander("Descriptive Summary"):
            st.dataframe(sample_df.describe())

        with st.expander("Intensity Distribution"):
            st.dataframe(sample_df['Intensity'].value_counts())

#Guneet, please do the plots
"""    else:
        st.subheader("Plots")

        # Layouts
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.expander("Dist Plot of Gender"):
                gen_df = df['Gender'].value_counts().to_frame()
                gen_df = gen_df.reset_index()
                gen_df.columns = ['Gender Type', 'Counts']
                p01 = px.pie(gen_df, names='Gender Type', values='Counts')
                st.plotly_chart(p01, use_container_width=True)

            with st.expander("Dist Plot of Class"):
                # Check for NaN values and ensure 'class' is categorical
                if df['class'].isnull().any():
                    st.write("Warning: 'class' column contains NaN values. Dropping NaN values.")
                    df = df.dropna(subset=['class'])
                if not pd.api.types.is_categorical_dtype(df['class']):
                    df['class'] = df['class'].astype('category')

                fig = plt.figure()
                sns.countplot(x=df['class'])
                st.pyplot(fig)

        with col2:
            with st.expander("Gender Distribution"):
                st.dataframe(df['Gender'].value_counts())

            with st.expander("Class Distribution"):
                st.dataframe(df['class'].value_counts())

        with st.expander("Frequency Dist Plot of Age"):
            p = px.bar(freq_df, x='Age', y='count')
            st.plotly_chart(p)

            p2 = px.line(freq_df, x='Age', y='count')
            st.plotly_chart(p2)

        with st.expander("Outlier Detection Plot"):
            fig = plt.figure()
            sns.boxplot(x=df['Age'])
            st.pyplot(fig)

            p3 = px.box(df, x='Age', color='Gender')
            st.plotly_chart(p3)

        with st.expander("Correlation Plot"):
            corr_matrix = df_clean.corr()
            fig = plt.figure(figsize=(20, 10))
            sns.heatmap(corr_matrix, annot=True)
            st.pyplot(fig)

            p3 = px.imshow(corr_matrix)
            st.plotly_chart(p3)
"""