import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
from movie_recommender import render_movie_recommender
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="ML Models Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #00ff87, #60efff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .metric-container {
        background: linear-gradient(90deg, #20252c 0%, #2d3436 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: #00ff87;
        text-align: center;
        margin: 0.7rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-container h3 {
        color: #60efff;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #20252c 0%, #2d3436 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: #00ff87;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .feature-input {
        background-color: #2d3436;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        color: #ffffff;
    }
    .stApp {
        background: linear-gradient(to bottom, #1a1f25, #2d3436);
    }
    /* Streamlit elements customization */
    .stTextInput > div > div {
        background-color: #2d3436;
        color: white;
    }
    .stSelectbox > div > div {
        background-color: #2d3436;
        color: white;
    }
    .stNumberInput > div > div {
        background-color: #2d3436;
        color: white;
    }
    /* Headers and text */
    h1, h2, h3, h4, h5, h6 {
        color: #00ff87 !important;
    }
    p {
        color: #e0e0e0 !important;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
    /* Dataframe styling */
    .dataframe {
        background-color: #20252c !important;
        color: #e0e0e0 !important;
    }
    .dataframe th {
        background-color: #2d3436 !important;
        color: #00ff87 !important;
    }
    /* Plotly chart background */
    .js-plotly-plot .plotly {
        background-color: rgba(45, 52, 54, 0.8) !important;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #20252c;
    }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00ff87 0%, #60efff 100%);
        color: #1a1f25;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,255,135,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Application:",
    ["🏠 Home", "🏘️ House Price Prediction", "🎬 Movie Recommendations"]
)

if app_mode == "🏠 Home":
    st.markdown('<h1 class="main-header">Welcome to ML Models Dashboard</h1>', unsafe_allow_html=True)
    
    st.write("""
    ### Choose from our ML-powered tools:
    
    #### 🏘️ House Price Prediction
    - Predict house prices using advanced machine learning models
    - Analyze feature importance and model performance
    - Get detailed insights about your property valuation
    
    #### 🎬 Movie Recommendations
    - Get personalized movie recommendations
    - Discover new movies based on your preferences
    - Explore detailed movie information and ratings
    """)
    
    # Display some statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🏘️ Housing Model Features
        - Multiple ML algorithms
        - Feature importance analysis
        - Real-time predictions
        - Interactive visualizations
        """)
    
    with col2:
        st.markdown("""
        ### 🎬 Movie Recommender Features
        - Content-based filtering
        - Similarity analysis
        - Personalized suggestions
        - Detailed movie info
        """)

elif app_mode == "🏘️ House Price Prediction":
    # ... existing housing price prediction code ...
    # (The existing code from line ~50 onwards remains unchanged)
    
    # Title and description
    st.markdown('<h1 class="main-header">🏠 Boston Housing Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    This application predicts house prices in Boston using machine learning models trained on the famous Boston Housing dataset.
    The dataset contains 13 features about houses and neighborhoods that influence housing prices.
    """)

    # Add before load_data function

    def validate_dataset(df):
        """Validate the input dataset for required features and data types"""
        required_features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 
                            'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']
        
        # Check for required columns
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            return False
        
        # Check for numeric data types
        non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
        if len(non_numeric_cols) > 0:
            st.error(f"❌ Non-numeric columns found: {', '.join(non_numeric_cols)}")
            return False
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            st.warning("⚠️ Dataset contains missing values. They will be handled during preprocessing.")
        
        # Check for negative values in features that should be positive
        positive_features = ['rm', 'age', 'tax', 'ptratio', 'b', 'medv']
        for feature in positive_features:
            if (df[feature] < 0).any():
                st.error(f"❌ Negative values found in {feature} column")
                return False
        
        return True

    def engineer_features(df):
        """Create new features from existing ones"""
        # Room-related features
        df['rooms_per_person'] = df['rm'] / df['ptratio']
        df['living_space'] = df['rm'] ** 2
        
        # Location-based features
        df['location_score'] = (df['dis'] * (1 - df['crim']/df['crim'].max()) * 
                              (1 + df['chas']) / (1 + np.log1p(df['tax'])))
        
        # Socioeconomic features
        df['wealth_index'] = ((1 - df['lstat']/100) * (df['b']/df['b'].max()) * 
                             (1 - df['crim']/df['crim'].max()))
        
        # Environmental features
        df['env_index'] = (1 - df['nox']) * (1 - df['indus']/100)
        
        return df

    # Load and cache data
    # File uploader for dataset
    uploaded_file = st.file_uploader(
        "Upload your housing dataset CSV file",
        type=["csv"],
        help="Upload your own housing dataset in CSV format. If no file is uploaded, the default Boston Housing dataset will be used."
    )

    @st.cache_data
    def load_data(_uploaded_file):
        """Load housing dataset from either uploaded file or default BostonHousing.csv"""
        try:
            if _uploaded_file is not None:
                # Read uploaded file
                data = pd.read_csv(_uploaded_file)
                st.success("✅ Successfully loaded your uploaded dataset!")
            else:
                # Try to load default Boston Housing dataset
                data = pd.read_csv('BostonHousing.csv')
                st.info("ℹ️ Using the default Boston Housing dataset")
        except Exception as e:
            # If both fail, create sample data for demonstration
            st.warning("⚠️ Could not load dataset. Using sample data for demonstration.")
            np.random.seed(42)
            n_samples = 506
            data = pd.DataFrame({
                'crim': np.random.exponential(3, n_samples),
                'zn': np.random.choice([0, 12.5, 25, 50, 85, 90], n_samples),
                'indus': np.random.uniform(0.5, 27, n_samples),
                'chas': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
                'nox': np.random.uniform(0.3, 0.9, n_samples),
                'rm': np.random.normal(6.3, 0.7, n_samples),
                'age': np.random.uniform(2, 100, n_samples),
                'dis': np.random.uniform(1, 12, n_samples),
                'rad': np.random.choice(range(1, 25), n_samples),
                'tax': np.random.uniform(180, 720, n_samples),
                'ptratio': np.random.uniform(12, 22, n_samples),
                'b': np.random.uniform(0, 400, n_samples),
                'lstat': np.random.uniform(2, 38, n_samples),
                'medv': np.random.uniform(5, 50, n_samples)
            })
            # Create some correlation with target
            data['medv'] = (
                50 - data['lstat'] * 0.5 + 
                data['rm'] * 4 + 
                np.random.normal(0, 3, n_samples)
            ).clip(5, 50)
        
        # Handle missing values
        data = data.fillna(data.mean())
        
        # After loading data and before returning:
        if not validate_dataset(data):
            st.error("❌ Dataset validation failed. Please check the requirements.")
            return None

        return data

    # Feature engineering options
    def apply_feature_engineering(data):
        """Apply feature engineering based on user selection"""
        with st.expander("🔧 Feature Engineering Options"):
            if st.checkbox("Enable Feature Engineering", value=False):
                data = engineer_features(data)
                st.success("✅ New features created successfully!")
                st.write("New Features Added:", 
                        ["rooms_per_person", "living_space", "location_score", 
                         "wealth_index", "env_index"])
        return data

    # Feature descriptions
    FEATURE_DESCRIPTIONS = {
        'crim': 'Per capita crime rate by town',
        'zn': 'Proportion of residential land zoned for lots over 25,000 sq.ft.',
        'indus': 'Proportion of non-retail business acres per town',
        'chas': 'Charles River dummy variable (1 if tract bounds river; 0 otherwise)',
        'nox': 'Nitric oxides concentration (parts per 10 million)',
        'rm': 'Average number of rooms per dwelling',
        'age': 'Proportion of owner-occupied units built prior to 1940',
        'dis': 'Weighted distances to employment centres',
        'rad': 'Index of accessibility to radial highways',
        'tax': 'Full-value property-tax rate per $10,000',
        'ptratio': 'Pupil-teacher ratio by town',
        'b': 'Proportion of blacks by town',
        'lstat': '% lower status of the population',
        'medv': 'Median value of owner-occupied homes in $1000s (TARGET)'
    }

    # Load data
    data = load_data(uploaded_file)

    # Check if data loading was successful
    if data is None:
        st.error("❌ Failed to load dataset. Please check your data format and try again.")
        st.stop()

    # Apply feature engineering if enabled
    data = apply_feature_engineering(data)

    # Sidebar for navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Data Overview", "📈 Model Training", "🔮 Price Prediction", "📊 Model Analysis"]
    )

    if page == "🏠 Data Overview":
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-container"><h3>Samples</h3><h2>{}</h2></div>'.format(len(data)), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-container"><h3>Features</h3><h2>{}</h2></div>'.format(len(data.columns)-1), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-container"><h3>Avg Price</h3><h2>${:.1f}K</h2></div>'.format(data['medv'].mean()), unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-container"><h3>Price Range</h3><h2>${:.1f}K-{:.1f}K</h2></div>'.format(data['medv'].min(), data['medv'].max()), unsafe_allow_html=True)
        
        # Data sample
        st.subheader("Data Sample")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Feature descriptions
        st.subheader("Feature Descriptions")
        for feature, description in FEATURE_DESCRIPTIONS.items():
            st.write(f"**{feature}**: {description}")
        
        # Data statistics
        st.subheader("Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
        
        # Enhanced Visualizations
        st.subheader("Data Visualizations")
        
        # Visualization type selector
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Distribution Plot", "Box Plot", "Bar Plot", "Line Plot", "Scatter Plot", "Correlation Heatmap", "Violin Plot"],
            help="Choose different types of visualizations to explore the data"
        )
        
        if viz_type == "Distribution Plot":
            feature = st.selectbox("Select Feature", data.columns)
            fig = px.histogram(
                data, x=feature, 
                nbins=30,
                title=f'Distribution of {feature}',
                color_discrete_sequence=['#00ff87']
            )
            fig.update_layout(
                plot_bgcolor='rgba(45, 52, 54, 0.8)',
                paper_bgcolor='rgba(45, 52, 54, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Box Plot":
            feature = st.selectbox("Select Feature", data.columns)
            fig = px.box(
                data, y=feature,
                title=f'Box Plot of {feature}',
                color_discrete_sequence=['#00ff87']
            )
            fig.update_layout(
                plot_bgcolor='rgba(45, 52, 54, 0.8)',
                paper_bgcolor='rgba(45, 52, 54, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Bar Plot":
            feature = st.selectbox("Select Feature", data.columns)
            agg_func = st.selectbox("Select Aggregation", ["mean", "median", "sum", "count"])
            if st.checkbox("Add Group By"):
                group_by = st.selectbox("Group By Feature", data.columns)
                agg_data = data.groupby(group_by)[feature].agg(agg_func).reset_index()
                fig = px.bar(
                    agg_data, x=group_by, y=feature,
                    title=f'{agg_func.capitalize()} {feature} by {group_by}',
                    color_discrete_sequence=['#00ff87']
                )
            else:
                fig = px.bar(
                    data, x=data.index, y=feature,
                    title=f'Bar Plot of {feature}',
                    color_discrete_sequence=['#00ff87']
                )
            fig.update_layout(
                plot_bgcolor='rgba(45, 52, 54, 0.8)',
                paper_bgcolor='rgba(45, 52, 54, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Line Plot":
            feature = st.selectbox("Select Feature", data.columns)
            fig = px.line(
                data, y=feature,
                title=f'Line Plot of {feature}',
                color_discrete_sequence=['#00ff87']
            )
            fig.update_layout(
                plot_bgcolor='rgba(45, 52, 54, 0.8)',
                paper_bgcolor='rgba(45, 52, 54, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("Select X-axis Feature", data.columns)
            with col2:
                y_feature = st.selectbox("Select Y-axis Feature", data.columns)
            
            color_feature = st.selectbox("Select Color Feature (optional)", ["None"] + list(data.columns))
            
            if color_feature == "None":
                fig = px.scatter(
                    data, x=x_feature, y=y_feature,
                    title=f'Scatter Plot: {x_feature} vs {y_feature}',
                    color_discrete_sequence=['#00ff87']
                )
            else:
                fig = px.scatter(
                    data, x=x_feature, y=y_feature, color=color_feature,
                    title=f'Scatter Plot: {x_feature} vs {y_feature} (colored by {color_feature})',
                    color_continuous_scale='Viridis'
                )
            fig.update_layout(
                plot_bgcolor='rgba(45, 52, 54, 0.8)',
                paper_bgcolor='rgba(45, 52, 54, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Correlation Heatmap":
            corr_matrix = data.corr()
            fig = px.imshow(
                corr_matrix,
                title='Feature Correlation Heatmap',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig.update_layout(
                width=800, height=600,
                plot_bgcolor='rgba(45, 52, 54, 0.8)',
                paper_bgcolor='rgba(45, 52, 54, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Violin Plot":
            feature = st.selectbox("Select Feature", data.columns)
            fig = px.violin(
                data, y=feature,
                title=f'Violin Plot of {feature}',
                box=True,
                points="all",
                color_discrete_sequence=['#00ff87']
            )
            fig.update_layout(
                plot_bgcolor='rgba(45, 52, 54, 0.8)',
                paper_bgcolor='rgba(45, 52, 54, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature vs target scatter plots
        st.subheader("Features vs House Price")
        feature_cols = data.columns.drop('medv').tolist()
        selected_features = st.multiselect("Select features to plot:", feature_cols, default=['rm', 'lstat', 'crim'])
        
        if selected_features:
            fig = make_subplots(
                rows=len(selected_features), cols=1,
                subplot_titles=[f'{feat} vs Price' for feat in selected_features],
                vertical_spacing=0.08
            )
            
            for i, feature in enumerate(selected_features):
                fig.add_trace(
                    go.Scatter(x=data[feature], y=data['medv'], mode='markers',
                              name=feature, opacity=0.6),
                    row=i+1, col=1
                )
                fig.update_xaxes(title_text=feature, row=i+1, col=1)
                fig.update_yaxes(title_text='Price ($1000s)', row=i+1, col=1)
            
            fig.update_layout(height=300*len(selected_features), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    elif page == "📈 Model Training":
        st.header("Model Training & Evaluation")
        
        # Model selection
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model:",
                ["Linear Regression", "Random Forest", "Gradient Boosting"]
            )
            test_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05)
        
        with col2:
            random_state = st.number_input("Random State:", value=42, min_value=1)
            if model_type == "Random Forest":
                n_estimators = st.slider("Number of Trees:", 10, 200, 100, 10)
            elif model_type == "Gradient Boosting":
                n_estimators = st.slider("Number of Trees:", 10, 200, 100, 10)
                learning_rate = st.slider("Learning Rate:", 0.01, 0.3, 0.1, 0.01)
        
        # Advanced model configuration
        with st.expander("🔧 Advanced Model Configuration"):
            # Preprocessing options
            scaler_type = st.selectbox(
                "Select Scaler:",
                ["Standard Scaler", "Robust Scaler", "MinMax Scaler"]
            )
            
            # Feature selection
            use_feature_selection = st.checkbox("Enable Feature Selection")
            if use_feature_selection:
                feature_selection_method = st.selectbox(
                    "Feature Selection Method:",
                    ["SelectKBest", "Recursive Feature Elimination (RFE)"]
                )
                n_features = st.slider("Number of features to select:", 
                                     min_value=1, 
                                     max_value=len(X.columns), 
                                     value=min(10, len(X.columns)))

            # Cross-validation options
            use_cv = st.checkbox("Enable Cross-Validation")
            if use_cv:
                n_folds = st.slider("Number of CV Folds:", 3, 10, 5)

            # Model-specific hyperparameters
            if model_type == "Linear Regression":
                regularization = st.selectbox(
                    "Regularization:",
                    ["None", "Ridge", "Lasso", "ElasticNet"]
                )
                if regularization != "None":
                    alpha = st.slider("Alpha:", 0.0, 1.0, 0.1, 0.01)
                
            elif model_type == "Random Forest":
                max_depth = st.slider("Max Depth:", 3, 30, 10)
                min_samples_split = st.slider("Min Samples Split:", 2, 20, 2)
                
            elif model_type == "Gradient Boosting":
                max_depth = st.slider("Max Depth:", 3, 10, 3)
                subsample = st.slider("Subsample Ratio:", 0.5, 1.0, 0.8, 0.1)

        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Prepare data
                X = data.drop('medv', axis=1)
                y = data['medv']
                
                # Apply scaling
                if scaler_type == "Standard Scaler":
                    scaler = StandardScaler()
                elif scaler_type == "Robust Scaler":
                    scaler = RobustScaler()
                else:
                    scaler = MinMaxScaler()
                
                X_scaled = scaler.fit_transform(X)
                
                # Feature selection
                if use_feature_selection:
                    if feature_selection_method == "SelectKBest":
                        selector = SelectKBest(f_regression, k=n_features)
                        X_selected = selector.fit_transform(X_scaled, y)
                        selected_features = X.columns[selector.get_support()].tolist()
                    else:  # RFE
                        base_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                        selector = RFE(base_estimator, n_features_to_select=n_features)
                        X_selected = selector.fit_transform(X_scaled, y)
                        selected_features = X.columns[selector.get_support()].tolist()
                    
                    st.write("Selected Features:", selected_features)
                else:
                    X_selected = X_scaled
                    selected_features = X.columns.tolist()

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y, test_size=test_size, random_state=random_state
                )
                
                # Initialize model with advanced parameters
                if model_type == "Linear Regression":
                    if regularization == "None":
                        model = LinearRegression()
                    elif regularization == "Ridge":
                        model = Ridge(alpha=alpha)
                    elif regularization == "Lasso":
                        model = Lasso(alpha=alpha)
                    else:
                        model = ElasticNet(alpha=alpha)
                elif model_type == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state
                    )
                else:  # Gradient Boosting
                    model = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        subsample=subsample,
                        random_state=random_state
                    )
                
                # Cross-validation if enabled
                if use_cv:
                    cv_scores = cross_val_score(
                        model, X_selected, y,
                        cv=KFold(n_splits=n_folds, shuffle=True, random_state=42),
                        scoring='r2'
                    )
                    st.write("Cross-Validation Scores:")
                    st.write(f"Mean R² Score: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")

                # Train final model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Save model
                if not os.path.exists('models'):
                    os.makedirs('models')
                model_filename = f'models/{model_type.lower().replace(" ", "_")}_model.joblib'
                joblib.dump(model, model_filename)
                st.success(f"✅ Model saved to {model_filename}")
                
                # Store in session state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.model_type = model_type
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.feature_names = selected_features
                
                # Display metrics
                st.success("✅ Model trained successfully!")
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<div class="metric-container"><h3>R² Score</h3><h2>{r2:.3f}</h2></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-container"><h3>RMSE</h3><h2>{rmse:.2f}</h2></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-container"><h3>MAE</h3><h2>{mae:.2f}</h2></div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="metric-container"><h3>MSE</h3><h2>{mse:.2f}</h2></div>', unsafe_allow_html=True)
                
                # Prediction vs Actual plot
                st.subheader("Prediction vs Actual Values")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test, y=y_pred, mode='markers',
                    name='Predictions', opacity=0.6,
                    marker=dict(color='blue', size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()], 
                    y=[y_test.min(), y_test.max()],
                    mode='lines', name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title='Predicted vs Actual House Prices',
                    xaxis_title='Actual Price ($1000s)',
                    yaxis_title='Predicted Price ($1000s)',
                    width=600, height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Residuals plot
                st.subheader("Residuals Analysis")
                residuals = y_test - y_pred
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_pred, y=residuals, mode='markers',
                    name='Residuals', opacity=0.6,
                    marker=dict(color='green', size=8)
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    title='Residuals Plot',
                    xaxis_title='Predicted Price ($1000s)',
                    yaxis_title='Residuals',
                    width=600, height=500
                )
                st.plotly_chart(fig, use_container_width=True)

    elif app_mode == "🔮 Price Prediction":
        st.header("House Price Prediction")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train a model first in the 'Model Training' section!")
        else:
            st.subheader("Enter House Features")
            
            # Create input fields for all features
            col1, col2, col3 = st.columns(3)
            
            input_values = {}
            features = st.session_state.feature_names
            
            # Organize inputs in columns
            for i, feature in enumerate(features):
                col = [col1, col2, col3][i % 3]
                
                with col:
                    if feature == 'chas':
                        input_values[feature] = st.selectbox(
                            f"{feature.upper()} (Charles River)",
                            [0, 1], 
                            format_func=lambda x: "Yes" if x == 1 else "No"
                        )
                    else:
                        # Get reasonable defaults based on data statistics
                        feature_mean = data[feature].mean()
                        feature_std = data[feature].std()
                        feature_min = data[feature].min()
                        feature_max = data[feature].max()
                        
                        input_values[feature] = st.number_input(
                            f"{feature.upper()} ({FEATURE_DESCRIPTIONS.get(feature, 'Feature')})",
                            value=float(feature_mean),
                            min_value=float(feature_min),
                            max_value=float(feature_max),
                            step=float(feature_std/10),
                            help=FEATURE_DESCRIPTIONS.get(feature, 'Feature description')
                        )
            
            if st.button("💡 Predict Price", type="primary"):
                # Prepare input data
                input_df = pd.DataFrame([input_values])
                
                # Make prediction
                if st.session_state.model_type == "Linear Regression":
                    input_scaled = st.session_state.scaler.transform(input_df)
                    prediction = st.session_state.model.predict(input_scaled)[0]
                else:
                    prediction = st.session_state.model.predict(input_df)[0]
                
                # Display prediction
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<h2>🏠 Predicted House Price</h2>'
                    f'<h1>${prediction:.2f}K</h1>'
                    f'<p>Estimated market value: ${prediction*1000:,.0f}</p>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                # Show confidence interval (rough estimate)
                if 'y_pred' in st.session_state:
                    residuals = st.session_state.y_test - st.session_state.y_pred
                    std_residual = np.std(residuals)
                    
                    lower_bound = prediction - 1.96 * std_residual
                    upper_bound = prediction + 1.96 * std_residual
                    
                    st.info(f"📊 **95% Confidence Interval**: ${lower_bound:.2f}K - ${upper_bound:.2f}K")
                
                # Compare with similar houses
                st.subheader("📈 Price Context")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    percentile = (data['medv'] < prediction).mean() * 100
                    st.metric("Price Percentile", f"{percentile:.1f}%")
                with col2:
                    avg_price = data['medv'].mean()
                    diff_from_avg = ((prediction - avg_price) / avg_price) * 100
                    st.metric("vs Average", f"{diff_from_avg:+.1f}%")
                with col3:
                    median_price = data['medv'].median()
                    diff_from_median = ((prediction - median_price) / median_price) * 100
                    st.metric("vs Median", f"{diff_from_median:+.1f}%")

    elif app_mode == "📊 Model Analysis":
        st.header("Model Analysis & Feature Importance")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train a model first in the 'Model Training' section!")
        else:
            # Feature importance
            st.subheader("Feature Importance Analysis")
            
            model = st.session_state.model
            feature_names = st.session_state.feature_names
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(
                    feature_importance_df, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title='Feature Importance (Tree-based Model)',
                    color='importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = model.coef_
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coefficients,
                    'abs_coefficient': np.abs(coefficients)
                }).sort_values('abs_coefficient', ascending=False)
                
                fig = px.bar(
                    feature_importance_df, 
                    x='coefficient', 
                    y='feature',
                    orientation='h',
                    title='Feature Coefficients (Linear Model)',
                    color='coefficient',
                    color_continuous_scale='RdBu'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Permutation importance (works for all models)
            st.subheader("Permutation Feature Importance")
            
            if st.button("Calculate Permutation Importance"):
                with st.spinner("Calculating permutation importance..."):
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                    
                    if st.session_state.model_type == "Linear Regression":
                        X_test_scaled = st.session_state.scaler.transform(X_test)
                        perm_importance = permutation_importance(
                            model, X_test_scaled, y_test, 
                            n_repeats=10, random_state=42
                        )
                    else:
                        perm_importance = permutation_importance(
                            model, X_test, y_test, 
                            n_repeats=10, random_state=42
                        )
                    
                    perm_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': perm_importance.importances_mean,
                        'std': perm_importance.importances_std
                    }).sort_values('importance', ascending=False)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=perm_df['importance'],
                        y=perm_df['feature'],
                        orientation='h',
                        error_x=dict(type='data', array=perm_df['std']),
                        marker_color='lightblue'
                    ))
                    fig.update_layout(
                        title='Permutation Feature Importance',
                        xaxis_title='Importance Score',
                        yaxis_title='Features',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Model performance summary
            st.subheader("Model Performance Summary")
            
            if 'y_pred' in st.session_state:
                y_test = st.session_state.y_test
                y_pred = st.session_state.y_pred
                
                # Performance metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                metrics_df = pd.DataFrame({
                    'Metric': ['R² Score', 'RMSE', 'MAE', 'MSE'],
                    'Value': [r2, rmse, mae, mse],
                    'Description': [
                        'Coefficient of determination (higher is better)',
                        'Root Mean Squared Error (lower is better)', 
                        'Mean Absolute Error (lower is better)',
                        'Mean Squared Error (lower is better)'
                    ]
                })
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Error distribution
                st.subheader("Error Distribution")
                residuals = y_test - y_pred
                
                fig = px.histogram(
                    x=residuals, 
                    nbins=30,
                    title='Distribution of Prediction Errors'
                )
                fig.update_layout(
                    xaxis_title='Prediction Error',
                    yaxis_title='Frequency'
                )
                st.plotly_chart(fig, use_container_width=True)

elif app_mode == "🎬 Movie Recommendations":
    render_movie_recommender()

# Footer with dark theme
st.markdown("---")
st.markdown("""
<div style='text-align: center; background: linear-gradient(90deg, #20252c 0%, #2d3436 100%); 
     padding: 30px; border-radius: 10px; margin-top: 50px; 
     border: 1px solid rgba(255,255,255,0.1);
     box-shadow: 0 4px 15px rgba(0,0,0,0.2)'>
    <h3 style='color: #00ff87; margin-bottom: 10px;'>🤖 ML Models Dashboard</h3>
    <p style='color: #60efff; margin-bottom: 5px;'>Built with Streamlit & Scikit-learn</p>
    <p style='color: #e0e0e0; font-size: 0.9em;'>Featuring House Price Prediction and Movie Recommendations</p>
</div>
""", unsafe_allow_html=True)