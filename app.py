import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Load the saved model and scaler
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Load the dataset for visualization
df = pd.read_csv('data/diabetes.csv')

# Set page config
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Navigation labels
nav_items = ["Home", "Data Exploration", "Model Prediction", "Model Performance"]

# Initialize selected page in session_state
if "selected_page" not in st.session_state:
    st.session_state.selected_page = nav_items[0]

# Custom CSS for sidebar and image
st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #e74c3c;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #c0392b;
        box-shadow: 0 0 10px rgb(255, 255, 255);
    }
            
    div.stButton > button:first-child:active,
    div.stButton > button:first-child:focus {
        background-color: #c0392b !important;
        color: white !important;
        outline: none !important;
}
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #4b6cb7, #182848) !important;
    }
    .sidebar-title {
        color: white;
        font-size: 24px;
        text-align: center;
        margin-bottom: 25px;
        font-weight: bold;
    }

    .nav-box {
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        color: white;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        background-color: transparent;
    }
    .nav-box:hover {
        background-color: rgba(255,255,255,0.1);
    }
    .nav-box-selected {
        background-color: #e74c3c !important;  /* Red for selected */
        font-weight: bold !important;
        box-shadow: 0 0 10px rgba(231, 76, 60, 0.5) !important;
    }
    
    /* Target the actual button element */
    button[kind="secondary"] {
        background-color: white;
        border: none;
        width: 100%;
    }
    .sidebar-footer {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 30px;
        padding-top: 15px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.sidebar.markdown('<div class="sidebar-title">Diabetes Prediction</div>', unsafe_allow_html=True)

# Navigation labels
nav_items = ["Home", "Data Exploration", "Model Prediction", "Model Performance"]

# Initialize selected page in session_state
if "selected_page" not in st.session_state:
    st.session_state.selected_page = nav_items[0]

# Box-style navigation
for item in nav_items:
    is_selected = item == st.session_state.selected_page
    btn = st.sidebar.button(
        item, 
        key=f"nav_{item}", 
        help=f"Go to {item}",
        type="primary" if is_selected else "secondary"
    )
    
    # Apply custom styling
    st.sidebar.markdown(f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        const btn = document.querySelector('button[kind="{"primary" if is_selected else "secondary"}"]:has(div:contains("{item}"))');
        if (btn) {{
            btn.parentElement.classList.add('{"nav-box-selected" if is_selected else "nav-box"}');
            btn.style.width = '100%';
        }}
    }});
    </script>
    """, unsafe_allow_html=True)
    
    if btn:
        st.session_state.selected_page = item
        st.rerun()

# Get current page
page = st.session_state.selected_page

# Footer note
st.sidebar.markdown("""
<div class="sidebar-footer">
<b>ITBIN-2110-0064</b>: Machine Learning Model Deployment Assignment
</div>
""", unsafe_allow_html=True)

# Home page
if page == "Home":
    st.title("Diabetes Prediction Streamlit Web App")
    
    # Banner image with custom styling using base64
    import base64
    from pathlib import Path

    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded

    try:
        img_bytes = img_to_bytes("Diabetes_Prediction.png")
        st.markdown(
            f"""
            <style>
                .banner-image {{
                    width: 100%;
                    height: 340px;
                    object-fit: cover;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                }}
            </style>
            <img src="data:image/png;base64,{img_bytes}" class="banner-image">
            """, 
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error("Image file not found!")
    
    st.write("""
    This application predicts the likelihood of a patient having diabetes based on various medical measurements.
    """)
    
    
    st.write("""
    ### How to use this app:
    1. **Data Exploration**: View the dataset and explore visualizations
    2. **Model Prediction**: Enter patient data to get a diabetes prediction
    3. **Model Performance**: View the model's evaluation metrics
    """)

    # Display model performance summary
    st.subheader("Model Performance Summary")
    st.write("""
    - **Algorithm**: Support Vector Machine (SVM)
    - **Test Accuracy**: 75.32%
    - **ROC AUC Score**: 81.04%
    - **Precision**: 68.09%
    - **Recall**: 58.18%
    """)

# Data Exploration page
elif page == "Data Exploration":
    st.title("Data Exploration")
    
    st.subheader("Dataset Overview")
    st.write(f"Shape of dataset: {df.shape}")
    st.write("First 5 rows:")
    st.write(df.head())
    
    st.subheader("Data Description")
    st.write(df.describe())
    
    st.subheader("Missing Values")
    st.write("Note: Zero values were treated as missing values and replaced with column means")
    st.write(df.isnull().sum())
    
    st.subheader("Class Distribution")
    st.write(df['Outcome'].value_counts())
    
    st.subheader("Visualizations")
    
    # Histograms
    st.write("### Feature Distributions")
    fig, ax = plt.subplots(figsize=(10, 8))
    df.hist(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation heatmap
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    
    # Outcome distribution
    st.write("### Outcome Distribution")
    fig, ax = plt.subplots()
    df['Outcome'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Diabetes Outcome Distribution')
    ax.set_xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Model Prediction page
elif page == "Model Prediction":
    st.title("Diabetes Prediction")
    
    st.write("""
    Enter the patient's medical details to predict the likelihood of diabetes.
    """)
    
    # Reset form fields when navigating to this page
    if "form_reset" not in st.session_state or st.session_state.selected_page != "Model Prediction":
        st.session_state.form_reset = True
        st.session_state.pregnancies = 0
        st.session_state.glucose = 0
        st.session_state.blood_pressure = 0
        st.session_state.skin_thickness = 0
        st.session_state.insulin = 0
        st.session_state.bmi = 0.0
        st.session_state.diabetes_pedigree = 0.0
        st.session_state.age = 0
    
    # Create input fields with better formatting
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, 
                                    value=st.session_state.pregnancies,
                                    help="Number of times pregnant")
        glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=300, 
                                value=st.session_state.glucose,
                                help="Plasma glucose concentration")
        blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=150, 
                                       value=st.session_state.blood_pressure,
                                       help="Diastolic blood pressure")
        skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, 
                                       value=st.session_state.skin_thickness,
                                       help="Triceps skin fold thickness")
    
    with col2:
        insulin = st.number_input('Insulin (μU/mL)', min_value=0, max_value=1000, 
                                value=st.session_state.insulin,
                                help="2-Hour serum insulin")
        bmi = st.number_input('BMI (kg/m²)', min_value=0.0, max_value=70.0, 
                            value=st.session_state.bmi, step=0.1,
                            help="Body mass index")
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, 
                                          value=st.session_state.diabetes_pedigree, step=0.01,
                                          help="Diabetes pedigree function")
        age = st.number_input('Age (years)', min_value=0, max_value=120, 
                             value=st.session_state.age,
                             help="Age in years")
    
    # Check if all fields are zero
    all_zero = (pregnancies == 0 and 
                glucose == 0 and 
                blood_pressure == 0 and 
                skin_thickness == 0 and 
                insulin == 0 and 
                bmi == 0.0 and 
                diabetes_pedigree == 0.0 and 
                age == 0)
    
    # Only show predict button if not all fields are zero
    if not all_zero:
        if st.button('Predict Diabetes', key='predict_button'):
            with st.spinner('Predicting...'):
                try:
                    # Create input array
                    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                        insulin, bmi, diabetes_pedigree, age]])
                    
                    # Standardize the input
                    input_scaled = scaler.transform(input_data)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)
                    probability = model.predict_proba(input_scaled)[0][1]
                    
                    # Display results
                    st.subheader("Prediction Result")
                    if prediction[0] == 1:
                        st.error(f"**Diabetes Detected** with {probability*100:.2f}% probability")
                        st.write("The model predicts this patient has diabetes. Please consult with a healthcare professional.")
                    else:
                        st.success(f"**No Diabetes Detected** with {(1-probability)*100:.2f}% probability")
                        st.write("The model predicts this patient does not have diabetes.")
                    
                    # Enhanced Probability Meter with Ruler Marks
                    st.write("\n**Probability Meter**")
                    
                    # Create custom HTML/CSS for the enhanced progress bar
                    st.markdown(f"""
                    <style>
                        .progress-container {{
                            width: 100%;
                            background-color: #f0f2f6;
                            border-radius: 8px;
                            margin: 10px 0;
                            position: relative;
                            height: 25px;
                        }}
                        .progress-bar {{
                            width: {probability * 100}%;
                            height: 100%;
                            border-radius: 8px;
                            background-color: #e74c3c;
                            position: absolute;
                        }}
                        .ruler-marks {{
                            position: absolute;
                            width: 100%;
                            height: 100%;
                            display: flex;
                            justify-content: space-between;
                        }}
                        .mark {{
                            width: 1px;
                            height: 10px;
                            background-color: #7f8c8d;
                            position: relative;
                            top: 15px;
                        }}
                        .mark-label {{
                            position: absolute;
                            top: 20px;
                            font-size: 10px;
                            transform: translateX(-50%);
                        }}
                        .risk-zones {{
                            display: flex;
                            justify-content: space-between;
                            width: 100%;
                            margin-top: 25px;
                            font-size: 12px;
                        }}
                        .risk-zone {{
                            text-align: center;
                            padding: 2px 5px;
                            border-radius: 4px;
                        }}
                    </style>
                    
                    <div class="progress-container">
                        <div class="progress-bar"></div>
                        <div class="ruler-marks">
                            {' '.join(['<div class="mark" style="left: {}%"><div class="mark-label">{}</div></div>'.format(i, f"{i}%") for i in range(0, 101, 10)])}
                        </div>
                    </div>
                    
                    <div class="risk-zones">
                        <div class="risk-zone" style="background-color: rgba(46, 204, 113, 0.2); color: #27ae60;">Low (0-30%)</div>
                        <div class="risk-zone" style="background-color: rgba(241, 196, 15, 0.2); color: #f39c12;">Moderate (30-70%)</div>
                        <div class="risk-zone" style="background-color: rgba(231, 76, 60, 0.2); color: #c0392b;">High (70-100%)</div>
                    </div>
                    
                    <div style="text-align: center; margin-top: 5px; font-size: 14px;">
                        Current: <strong>{probability*100:.1f}%</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Interpretation guidance
                    st.subheader("Interpretation Guide")
                    st.write("""
                    - **0-30%**: Low risk
                    - **30-70%**: Moderate risk - recommend follow-up
                    - **70-100%**: High risk - recommend medical consultation
                    """)
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter at least one non-zero value to enable prediction")

# Model Performance page
elif page == "Model Performance":
    st.title("Model Performance Metrics")
    
    st.write("""
    These metrics show how the Support Vector Machine model performs on unseen test data.
    """)
    
    # Model metrics from your training output
    st.subheader("Classification Report")
    st.code("""
              precision    recall  f1-score   support

           0       0.79      0.85      0.82        99
           1       0.68      0.58      0.63        55

    accuracy                           0.75       154
   macro avg       0.73      0.72      0.72       154
weighted avg       0.75      0.75      0.75       154
    """)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = np.array([[84, 15], [23, 32]])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    ax.set_yticklabels(['No Diabetes', 'Diabetes'])
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader("ROC Curve (AUC = 0.8104)")
    fpr = np.array([0.0, 0.15151515, 1.0])
    tpr = np.array([0.0, 0.58181818, 1.0])
    roc_auc = 0.8104
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    # Performance Metrics
    st.subheader("Key Performance Metrics")
    metrics = {
        "Accuracy": "75.32%",
        "Precision": "68.09%",
        "Recall (Sensitivity)": "58.18%",
        "F1 Score": "62.75%",
        "ROC AUC": "81.04%"
    }
    
    for metric, value in metrics.items():
        st.metric(label=metric, value=value)
    
    # Cross-validation results
    st.subheader("Cross-Validation Results")
    st.write("""
    - **Mean Accuracy**: 75.73% (±2.90%)
    - **Mean ROC AUC**: 83.90% (±3.45%)
    - **Mean Precision**: 69.63% (±5.31%)
    - **Mean Recall**: 53.54% (±5.60%)
    """)