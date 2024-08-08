import streamlit as st
import pandas as pd

# Data Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def load_data(data):
    df = pd.read_csv(data)
    return df

def run_eda_app():
    st.subheader("EDA Section")
    sample_df = load_data("data/sampled_hotspots.csv")
    clean_df = load_data("data/cleaned_hotspots.csv")

    submenu = st.sidebar.selectbox("SubMenu", ["Preprocessing","Descriptive", "Visualizations"])
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

    else:
        viz_menu = st.sidebar.selectbox("Select Visualization", [
            "Head Fire Intensity Over Time",
            "Box Plots of Numerical Columns",
            "Correlation Matrix",
            "Highly Correlated Pairs",
            "Ecozone Distribution",
            "Fuel Distribution",
            "Fuel Type in Ecozones",
            "Head Fire Intensity Scatter Plot",
            "Head Fire Intensity Distribution",
            "Continuous Variables Over Time"
        ])

        if viz_menu == "Head Fire Intensity Over Time":
            clean_df['year'] = pd.to_datetime(clean_df['rep_date']).dt.year
            years_of_interest = [2019, 2020, 2021, 2022, 2023]
            clean_df = clean_df[clean_df['year'].isin(years_of_interest)]

            columns_to_drop = ['lat', 'lon', 'rep_date', 'source', 'sensor', 'satellite', 'agency', 'temp', 'rh', 'ws', 'greenup', 'elev', 'sfl', 'cfl', 'tfc0', 'ecozone', 'sfc0', 'cbh', 'uid', 'fid', 'year']
            columns_to_drop = [col for col in columns_to_drop if col in clean_df.columns]
            hotspots_numeric = clean_df.drop(columns=columns_to_drop)

            st.subheader("Head Fire Intensity Over Time")
            fig = px.line(clean_df, x='rep_date', y='hfi', title='Head Fire Intensity Over Time')
            fig.add_hline(y=20000, line_dash="dash", line_color="red")
            fig.add_hline(y=15000, line_dash="dash", line_color="red")
            fig.add_hline(y=30000, line_dash="dash", line_color="red")
            st.plotly_chart(fig)

        elif viz_menu == "Box Plots of Numerical Columns":
            hotspots_numeric = clean_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            st.subheader("Box Plots of Numerical Columns")
            fig = px.box(hotspots_numeric.melt(), x="variable", y="value", title="Box Plots of Numerical Columns", width=1000, height=700)
            st.plotly_chart(fig)

        elif viz_menu == "Correlation Matrix":
            hotspots_numeric = clean_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            correlation_matrix = hotspots_numeric.corr()
            st.subheader("Correlation Matrix of Numerical Variables")
            fig = px.imshow(correlation_matrix, title="Correlation Matrix", color_continuous_scale='viridis', aspect="auto", width=1000, height=700)
            st.plotly_chart(fig)

        elif viz_menu == "Highly Correlated Pairs":
            hotspots_numeric = clean_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            correlation_matrix = hotspots_numeric.corr()
            high_corr_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns if i != j and correlation_matrix.loc[i, j] > 0.8]
            high_corr_pairs = list(set(tuple(sorted(pair)) for pair in high_corr_pairs))
            st.subheader("Highly Correlated Pairs")
            fig = make_subplots(rows=3, cols=4, subplot_titles=[f"{i} vs {j}\nCorrelation: {correlation_matrix.loc[i, j]:.2f}" for i, j in high_corr_pairs[:12]])
            for idx, (i, j) in enumerate(high_corr_pairs[:12]):
                row, col = divmod(idx, 4)
                fig.add_trace(go.Scatter(x=hotspots_numeric[i], y=hotspots_numeric[j], mode='markers', opacity=0.5), row=row+1, col=col+1)
                fig.update_layout(showlegend=False, height=800, title="Highly Correlated Pairs", width=1000)
                st.plotly_chart(fig)

        elif viz_menu == "Ecozone Distribution":
            clean_df['ecozone'] = clean_df['ecozone'].astype('category')
            st.subheader("Count of Hotspots by Ecozone")
            fig = px.bar(clean_df, x='ecozone', title='Count of Hotspots by Ecozone', width=1000, height=700)
            st.plotly_chart(fig)

        elif viz_menu == "Fuel Distribution":
            clean_df['fuel'] = clean_df['fuel'].astype('category')
            st.subheader("Count of Hotspots by Fuel Type")
            fig = px.bar(clean_df, x='fuel', title='Count of Hotspots by Fuel Type', width=1000, height=700)
            st.plotly_chart(fig)

        elif viz_menu == "Fuel Type in Ecozones":
            clean_df['ecozone'] = clean_df['ecozone'].astype('category')
            clean_df['fuel'] = clean_df['fuel'].astype('category')
            ecozone_4 = clean_df[clean_df['ecozone'] == 4]
            ecozone_14 = clean_df[clean_df['ecozone'] == 14]
            st.subheader("Count of Hotspots by Fuel Type in Ecozone 4")
            fig = px.bar(ecozone_4, x='fuel', title='Count of Hotspots by Fuel Type in Ecozone 4', width=1000, height=700)
            st.plotly_chart(fig)
            st.subheader("Count of Hotspots by Fuel Type in Ecozone 14")
            fig = px.bar(ecozone_14, x='fuel', title='Count of Hotspots by Fuel Type in Ecozone 14', width=1000, height=700)
            st.plotly_chart(fig)

        elif viz_menu == "Head Fire Intensity Scatter Plot":
            clean_df['year'] = pd.to_datetime(clean_df['rep_date']).dt.year
            years_of_interest = [2019, 2020, 2021, 2022, 2023]
            clean_df = clean_df[clean_df['year'].isin(years_of_interest)]

            st.subheader("Head Fire Intensity Categories")
            fig = px.scatter(clean_df, x='temp', y='hfi', color='Intensity', title='Head Fire Intensity Categories', width=1000, height=700)
            st.plotly_chart(fig)

        elif viz_menu == "Head Fire Intensity Distribution":
            clean_df['year'] = pd.to_datetime(clean_df['rep_date']).dt.year
            years_of_interest = [2019, 2020, 2021, 2022, 2023]
            clean_df = clean_df[clean_df['year'].isin(years_of_interest)]

            st.subheader("Distribution of Head Fire Intensity Categories")  
            fig = px.histogram(clean_df, x='Intensity', title='Distribution of Head Fire Intensity Categories', width=1000, height=700)
            st.plotly_chart(fig)

        elif viz_menu == "Continuous Variables Over Time":
            continuous_variables = ['temp', 'rh', 'ws', 'wd', 'pcp', 'ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi', 'ros', 'sfc', 'tfc', 'hfi', 'cfb', 'elev', 'sfl', 'cfl']
            st.subheader("Continuous Variables Over Time")
            fig = make_subplots(rows=5, cols=4, subplot_titles=continuous_variables)
            for idx, var in enumerate(continuous_variables):
                row, col = divmod(idx, 4)
                fig.add_trace(go.Scatter(x=clean_df.index, y=clean_df[var], mode='lines', name=var), row=row+1, col=col+1)
                fig.update_layout(showlegend=False, height=1000, width=1200, title="Continuous Variables Over Time")
                st.plotly_chart(fig)