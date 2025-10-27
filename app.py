# Health Facility ML System

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --------------------------------------
# PAGE SETUP
# --------------------------------------
st.set_page_config(page_title="Health Facility ML Project", layout="wide")
st.title("🏥 Health Facility ML Analysis Dashboard")

st.markdown("""
This app performs **EDA**, **Regression**, **Classification**, and **Clustering**
on the uploaded Health Facility dataset.
""")

# --------------------------------------
# UPLOAD DATA
# --------------------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # --------------------------------------
    # EDA Section
    # --------------------------------------
    st.header("📊 Exploratory Data Analysis (EDA)")

    if st.checkbox("Show Basic Info"):
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write("Missing Values:", df.isnull().sum().sum())

    if st.checkbox("Show Summary Statistics"):
        st.write(df.describe())

    if st.checkbox("Correlation Heatmap"):
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False)
        st.pyplot(fig)

    if st.checkbox("Interactive Scatter Plot"):
        all_cols = df.columns.tolist()
        x_axis = st.selectbox("Select X-axis", all_cols, index=0)
        y_axis = st.selectbox("Select Y-axis", all_cols, index=1)
        fig = px.scatter(df, x=x_axis, y=y_axis, color=None, title="Scatter Plot")
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------
    # DATA PREPROCESSING
    # --------------------------------------
    st.header("⚙️ Data Preprocessing")

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for c in cat_cols:
        df[c] = le.fit_transform(df[c].astype(str))

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    st.write("✅ Data cleaned and encoded successfully.")

    # --------------------------------------
    # MODEL SELECTION
    # --------------------------------------
    st.header("🧠 Choose Model Type")

    task = st.radio("Select ML Task:", ["Regression", "Classification", "Clustering"])

    if task == "Regression":
        target = st.selectbox("Select Target Column", df.columns)
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_choice = st.selectbox("Choose Regression Model", ["Random Forest", "LightGBM", "XGBoost", "CatBoost"])
        
        if st.button("Train Regression Model"):
            if model_choice == "Random Forest":
                model = RandomForestRegressor(random_state=42)
            elif model_choice == "LightGBM":
                model = LGBMRegressor()
            elif model_choice == "XGBoost":
                model = XGBRegressor(eval_metric='rmse')
            else:
                model = CatBoostRegressor(verbose=0)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.write(f"**R² Score:** {r2:.3f}")
            st.write(f"**RMSE:** {rmse:.3f}")

            st.write("### Feature Importances:")
            feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.bar_chart(feat_imp.head(10))

    elif task == "Classification":
        target = st.selectbox("Select Target Column", df.columns)
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_choice = st.selectbox("Choose Classification Model", ["Random Forest", "LightGBM", "XGBoost", "CatBoost"])
        
        if st.button("Train Classification Model"):
            if model_choice == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif model_choice == "LightGBM":
                model = LGBMClassifier()
            elif model_choice == "XGBoost":
                model = XGBClassifier(eval_metric='logloss')
            else:
                model = CatBoostClassifier(verbose=0)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {acc:.4f}")

    else:  # Clustering
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        k = st.slider("Select number of clusters (k)", 2, 10, 3)

        if st.button("Run K-Means Clustering"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(scaled_data)
            st.success(f"K-Means applied successfully with {k} clusters!")

            # PCA Visualization
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(scaled_data)
            fig = px.scatter(x=reduced_data[:,0], y=reduced_data[:,1], color=df['Cluster'].astype(str),
                             title="K-Means Clustering Visualization")
            st.plotly_chart(fig, use_container_width=True)

            st.write("### Cluster Summary:")
            st.dataframe(df.groupby('Cluster').mean())

else:
    st.warning("👆 Please upload a CSV file to begin.")
