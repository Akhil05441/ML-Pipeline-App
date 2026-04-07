import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. UI Configuration & Custom CSS ---
st.set_page_config(page_title="DataSight: ML Pipeline", page_icon="🚀", layout="wide")

# Injecting some custom CSS for a cleaner, modern look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { border-radius: 20px; background-color: #ff4b4b; color: white; border: none; }
    .stButton>button:hover { background-color: #ff3333; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 DataSight: Dynamic ML Pipeline")
st.markdown("Upload your CSV, explore the data, and train models in real-time.")

# --- 2. Sidebar & Data Ingestion ---
with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

@st.cache_data # Caches the data so the app doesn't reload it on every interaction
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Create Tabs for a clean UI
    tab1, tab2, tab3, tab4 = st.tabs(["🗂️ Data Ingestion", "📊 EDA", "🛠️ Preprocessing", "🧠 Model Training"])
    
    # --- TAB 1: Data Ingestion ---
    with tab1:
        st.subheader("Raw Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())
        
        st.dataframe(df.head(15), use_container_width=True)

    # --- TAB 2: Exploratory Data Analysis (EDA) ---
    with tab2:
        st.subheader("Interactive Visualizations")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.markdown("#### Distribution Plot")
            dist_col = st.selectbox("Select column for distribution:", numeric_cols)
            fig = px.histogram(df, x=dist_col, marginal="box", color_discrete_sequence=['#ff4b4b'])
            st.plotly_chart(fig, use_container_width=True)
            
        with col_eda2:
            st.markdown("#### Correlation Heatmap")
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 3: Preprocessing ---
    with tab3:
        st.subheader("Clean & Transform")
        
        # Drop Columns
        cols_to_drop = st.multiselect("Select columns to drop:", df.columns)
        df_clean = df.drop(columns=cols_to_drop)
        
        # Handle Nulls
        drop_nulls = st.checkbox("Drop rows with missing values")
        if drop_nulls:
            df_clean = df_clean.dropna()
            
        # Outlier Removal (Basic Z-Score method)
        st.markdown("#### Outlier Removal")
        outlier_col = st.selectbox("Select column to check for outliers:", numeric_cols)
        z_thresh = st.slider("Z-Score Threshold (higher means keeping more data)", 1.0, 5.0, 3.0)
        
        if st.button("Apply Outlier Removal"):
            mean = df_clean[outlier_col].mean()
            std = df_clean[outlier_col].std()
            df_clean = df_clean[(df_clean[outlier_col] >= mean - z_thresh * std) & 
                                (df_clean[outlier_col] <= mean + z_thresh * std)]
            st.success(f"Outliers removed. New dataset shape: {df_clean.shape}")
            
        st.dataframe(df_clean.head(), use_container_width=True)

    # --- TAB 4: Model Training ---
    with tab4:
        st.subheader("Train & Evaluate Models")
        
        # Target Selection
        target_col = st.selectbox("Select Target Variable (Y):", df_clean.columns)
        features = df_clean.drop(columns=[target_col])
        
        # Encode Categorical Target if necessary
        le = LabelEncoder()
        y = le.fit_transform(df_clean[target_col]) if df_clean[target_col].dtype == 'object' else df_clean[target_col]
        
        # Encode Categorical Features
        X = pd.get_dummies(features, drop_first=True)
        
        # Model Selection
        model_choice = st.selectbox("Select Algorithm", ["Random Forest", "K-Nearest Neighbors"])
        
        col_param1, col_param2 = st.columns(2)
        if model_choice == "Random Forest":
            n_estimators = col_param1.slider("Number of Trees", 10, 200, 100)
            max_depth = col_param2.slider("Max Depth", 1, 20, 10)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        else:
            n_neighbors = col_param1.slider("Number of Neighbors (K)", 1, 15, 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100

        if st.button("Train Model 🚀"):
            with st.spinner("Training in progress..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train & Predict
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Results
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Model Trained Successfully! Accuracy: **{acc:.2%}**")
                
                st.markdown("#### Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
else:
    st.info("Awaiting CSV file to be uploaded in the sidebar.")