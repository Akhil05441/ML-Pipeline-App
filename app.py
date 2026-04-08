import streamlit as st
import pandas as pd
import plotly.express as px
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# UI ENHANCEMENTS & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Ultimate ML Pipeline", page_icon="✨", layout="wide")

# Injecting CSS for attractive UI (Gradients, Hover Effects, Shadows)
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0e1117; }
    
    /* Gradient Headers */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF904B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Button Animations and Styling */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF904B 100%);
        color: white;
        border-radius: 30px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.6);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #1e2530;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✨ The Ultimate ML Pipeline")
st.markdown("From raw data to K-Fold validated models with an interactive UI.")

# ==========================================
# STEP 1: INPUT DATA
# ==========================================
with st.sidebar:
    st.header("📂 1. Input Data")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    # Read the data
    df = load_data(uploaded_file)
    st.toast('Data loaded successfully!', icon='✅')

    # Create Tabs for the Pipeline Steps
    tabs = st.tabs(["📊 2. EDA", "🧹 3. Data Engineering", "🎯 4. Features", "✂️ 5. Split", "🧠 6-8. Model & Validate"])

    # ==========================================
    # STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================
    with tabs[0]:
        st.header("Exploratory Data Analysis")
        st.dataframe(df.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])

        st.subheader("Interactive Distributions")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select Column for Distribution", numeric_cols)
            fig = px.histogram(df, x=selected_col, marginal="box", color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found for plotting.")

    # ==========================================
    # STEP 3: DATA ENGINEERING & CLEANING
    # ==========================================
    with tabs[1]:
        st.header("Data Engineering & Cleaning")
        
        df_clean = df.copy()
        
        # Handle Missing Values
        st.markdown("**Missing Values Handling**")
        missing_count = df_clean.isna().sum().sum()
        if missing_count > 0:
            st.warning(f"Found {missing_count} missing values.")
            if st.checkbox("Drop rows with missing values"):
                df_clean = df_clean.dropna()
                st.success("Missing values dropped!")
        else:
            st.success("No missing values found!")

        # Encode Categorical Variables
        st.markdown("**Categorical Encoding**")
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            if st.checkbox("Auto-Encode Categorical Features (Label Encoding)"):
                le = LabelEncoder()
                for col in cat_cols:
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                st.success("Categorical variables encoded!")
        
        st.dataframe(df_clean.head(), use_container_width=True)

    # ==========================================
    # STEP 4: FEATURE SELECTION
    # ==========================================
    with tabs[2]:
        st.header("Feature Selection")
        target_col = st.selectbox("Select Target Variable (Y)", df_clean.columns)
        
        features = df_clean.drop(columns=[target_col])
        X = features
        y = df_clean[target_col]

        st.markdown("**Automated Feature Selection (SelectKBest)**")
        num_features = st.slider("Number of top features to keep", 1, len(features.columns), len(features.columns))
        
        if st.button("Apply Feature Selection"):
            # Ensure no NaNs or Strings remain before selection
            try:
                selector = SelectKBest(score_func=f_classif, k=num_features)
                X_selected = selector.fit_transform(X, y)
                selected_cols = X.columns[selector.get_support()]
                X = X[selected_cols]
                st.success(f"Selected Features: {', '.join(selected_cols)}")
            except Exception as e:
                st.error(f"Error during feature selection. Ensure data is numeric and clean. ({e})")

    # ==========================================
    # STEP 5: DATA SPLIT
    # ==========================================
    with tabs[3]:
        st.header("Train / Test Split")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
        
        # Scaling toggle
        apply_scaling = st.toggle("Apply Standard Scaling to Features", value=True)
        
        if st.button("Split Data"):
            st.toast('Data successfully split!', icon='✂️')
            # The actual split happens in the next tab to pass variables correctly,
            # but we simulate the UI interaction here.

    # ==========================================
    # STEPS 6, 7 & 8: MODELING, K-FOLD & METRICS
    # ==========================================
    with tabs[4]:
        st.header("Model Selection & Validation")
        
        model_choice = st.selectbox("Select Algorithm", ["Random Forest", "Logistic Regression", "Support Vector Machine"])
        k_folds = st.slider("Number of K-Folds for Validation", 2, 10, 5)
        
        if st.button("🚀 Train & Validate Model"):
            # UI Animation: Progress Bar & Spinner
            progress_bar = st.progress(0)
            with st.spinner("Initializing Pipeline..."):
                time.sleep(0.5)
                
                try:
                    # 5. Split
                    progress_bar.progress(20, text="Splitting Data...")
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    if apply_scaling:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    # 6. Model Selection
                    progress_bar.progress(40, text="Configuring Model...")
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(random_state=42)
                    elif model_choice == "Logistic Regression":
                        model = LogisticRegression(random_state=42, max_iter=1000)
                    else:
                        model = SVC(random_state=42)
                    
                    # 7. K-Fold Validation
                    progress_bar.progress(60, text=f"Running {k_folds}-Fold Cross Validation...")
                    cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                    
                    # Training final model
                    progress_bar.progress(80, text="Training Final Model on Training Set...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    progress_bar.progress(100, text="Complete!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    
                    # UI Animation: Success Balloons
                    st.balloons()
                    
                    # 8. Performance Metrics
                    st.subheader(f"Results for {model_choice}")
                    
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("K-Fold Mean Accuracy", f"{cv_scores.mean():.2%}")
                    col_m2.metric("Test Set Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                    
                    # Classification Report
                    st.markdown("**Classification Report**")
                    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                    st.dataframe(report_df.style.background_gradient(cmap='Oranges'), use_container_width=True)
                    
                    # Confusion Matrix Interactive Plot
                    st.markdown("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Reds', 
                                       labels=dict(x="Predicted", y="Actual", color="Count"))
                    st.plotly_chart(fig_cm, use_container_width=True)

                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Pipeline failed. Ensure data is fully numeric/encoded and target variable is correctly selected. Error details: {e}")

else:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to begin building your pipeline.")
