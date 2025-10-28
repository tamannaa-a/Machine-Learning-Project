# ======================================
# INTERACTIVE HEALTH FACILITY ML DASHBOARD
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------
st.set_page_config(
    page_title="Health Facility ML Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------
# CUSTOM CSS FOR STYLE
# ---------------------------------------
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        h1, h2, h3 { color: #2C3E50; }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 10em;
        }
        .stButton>button:hover {
            background-color: #1B4F72;
            color: white;
        }
        .reportview-container .markdown-text-container {
            font-size: 1.1em;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------
# APP TITLE AND INTRO
# ---------------------------------------
st.title("🏥 Health Facility Analysis Dashboard")
st.markdown("""
Welcome to the **Health Facility ML Dashboard** —  
a data-driven web app that lets you:
- Perform **Exploratory Data Analysis (EDA)**
- Run **Regression** or **Classification** models
- Explore **K-Means Clustering**
- Visualize data interactively with **Plotly**

Upload your dataset below to get started!
""")

# ---------------------------------------
# UPLOAD DATA
# ---------------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.write("### Dataset Preview:")
    st.dataframe(df.head(), use_container_width=True)

    # Sidebar Options
    st.sidebar.header("⚙️ Configuration Panel")
    section = st.sidebar.radio("Select Section:", ["EDA", "Modeling", "Clustering"])

    # =====================================
    # EDA SECTION
    # =====================================
    if section == "EDA":
        st.subheader("📊 Exploratory Data Analysis")

        st.markdown("##### Basic Information")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", df.isnull().sum().sum())

        with st.expander("🔍 View Dataset Summary"):
            st.write(df.describe())

        with st.expander("📈 Correlation Heatmap"):
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False)
            st.pyplot(fig)

        st.markdown("### Interactive Visuals")
        cols = df.columns.tolist()
        x_axis = st.selectbox("X-axis", cols)
        y_axis = st.selectbox("Y-axis", cols)
        fig = px.scatter(df, x=x_axis, y=y_axis, color=None, title="📊 Interactive Scatter Plot")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Distribution of a Feature")
        col_to_plot = st.selectbox("Select a Column", cols)
        fig = px.histogram(df, x=col_to_plot, nbins=30, color_discrete_sequence=['#2E86C1'])
        st.plotly_chart(fig, use_container_width=True)

    # =====================================
    #  MODELING SECTION
    # =====================================
    elif section == "Modeling":
        st.subheader("🤖 Machine Learning Models")

        task = st.radio("Select Task Type:", ["Regression", "Classification"])
        
        # Label Encoding
        cat_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        df.fillna(df.median(numeric_only=True), inplace=True)

        target = st.selectbox("🎯 Select Target Column", df.columns)
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_choice = st.selectbox("Select Model", 
            ["Random Forest", "LightGBM", "XGBoost", "CatBoost"])

        if st.button("🚀 Run Model"):
            with st.spinner("Training the model... please wait"):
                if task == "Regression":
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

                    st.success(f"**R² Score:** {r2:.3f}")
                    st.info(f"**RMSE:** {rmse:.3f}")

                    st.write("### 🔍 Feature Importance")
                    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    st.bar_chart(fi.head(10))

                else:  # Classification
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
                    st.success(f"✅ Accuracy: {acc:.4f}")

                    st.write("### 🔍 Feature Importance")
                    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    st.bar_chart(fi.head(10))

    # =====================================
    #  CLUSTERING SECTION
    # =====================================
    elif section == "Clustering":
        st.subheader("K-Means Clustering")

        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        k = st.slider("Select number of clusters (k)", 2, 10, 3)

        if st.button("Run K-Means"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(scaled_data)
            st.success(f"✅ K-Means applied successfully with {k} clusters!")

            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(scaled_data)
            fig = px.scatter(x=reduced_data[:,0], y=reduced_data[:,1],
                             color=df['Cluster'].astype(str),
                             title=f"Cluster Visualization (k={k})",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📋 Cluster Summary")
            st.dataframe(df.groupby('Cluster').mean().style.highlight_max(axis=0))

else:
    st.info("Upload your CSV file to start exploring and modeling!")
