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
    clean_df = load_data("data/cleaned_hotspots.csv")

    submenu = st.sidebar.selectbox("SubMenu", ["Preprocessing","Descriptive", "Plots"])
    if submenu == "Preprocessing":
        st.dataframe(sample_df)

        with st.expander("Data Overview"):
            st.dataframe(sample_df.shape)
            st.dataframe(sample_df.columns)
            st.dataframe(sample_df.dtypes)
            st.dataframe(sample_df.head())
        
        with st.expander("Missing Values"):
            null_values = sample_df.isnull().sum()
            columns_with_nulls = null_values[null_values > 0]
            st.write("Columns with null values:")
            st.dataframe(columns_with_nulls)

    if submenu == "Descriptive":
        st.dataframe(clean_df)

        with st.expander("Data Types Summary"):
            st.dataframe(clean_df.dtypes)

        with st.expander("Descriptive Summary"):
            columns_to_describe = [
                'temp', 'rh', 'ws', 'wd', 'pcp', 'ffmc', 'dmc', 'dc', 
                'isi', 'bui', 'fwi', 'ros', 'sfc', 'tfc', 'hfi', 
                'cfb', 'elev', 'sfl', 'cfl'
            ]

        with st.expander("Descriptive Summary"):
            st.dataframe(clean_df[columns_to_describe].describe())

        with st.expander("Intensity Distribution"):
            st.dataframe(clean_df['Intensity'].value_counts())

        corr_matrix = clean_df[columns_to_describe].corr()

        high_corr_pairs = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns 
                   if i != j and abs(corr_matrix.loc[i, j]) > 0.9]
        
        with st.expander("Highly Correlated Pairs"):
            st.write("Highly correlated pairs are those where the absolute value of the correlation coefficient is greater than 0.9.")
            if high_corr_pairs:
                st.write("The following pairs of features have high correlations:")
            for i, j in high_corr_pairs:
                st.write(f"{i} and {j} with correlation {corr_matrix.loc[i, j]:.2f}")

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