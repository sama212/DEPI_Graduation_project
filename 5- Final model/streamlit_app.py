import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .high-risk {
        background-color: rgba(255, 75, 75, 0.1);
        border: 1px solid rgba(255, 75, 75, 0.5);
    }
    .low-risk {
        background-color: rgba(75, 192, 192, 0.1);
        border: 1px solid rgba(75, 192, 192, 0.5);
    }
    .feature-importance {
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        # Load the pickled model
        with open('heart_disease_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load model metadata
        with open('heart_disease_model_metadata.json', 'r') as f:
            model_data = json.load(f)
        
        feature_names = model_data['featureNames']
        feature_importance = model_data['featureImportance']
        config = model_data['config']
        
        return model, feature_names, feature_importance, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

# Function to preprocess input data
def preprocess_input(data, feature_names):
    # Create a DataFrame with the same structure as training data
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Set numerical features
    input_df['age'] = data['age']
    input_df['trestbps'] = data['trestbps']
    input_df['chol'] = data['chol']
    input_df['fbs'] = 1 if data['fbs'] else 0
    input_df['thalch'] = data['thalch']
    input_df['exang'] = 1 if data['exang'] else 0
    input_df['oldpeak'] = data['oldpeak']
    input_df['ca'] = data['ca']
    
    # Set one-hot encoded features
    if data['sex'] == 'Female':
        input_df['sex_Female'] = 1
    else:
        input_df['sex_Male'] = 1
    
    # Chest pain type
    cp_col = f"cp_{data['cp'].replace(' ', ' ')}"
    if cp_col in feature_names:
        input_df[cp_col] = 1
    
    # Resting ECG
    restecg_col = f"restecg_{data['restecg'].replace(' ', ' ')}"
    if restecg_col in feature_names:
        input_df[restecg_col] = 1
    
    # Slope
    slope_col = f"slope_{data['slope']}"
    if slope_col in feature_names:
        input_df[slope_col] = 1
    
    # Thalassemia
    thal_col = f"thal_{data['thal'].replace(' ', ' ')}"
    if thal_col in feature_names:
        input_df[thal_col] = 1
    
    return input_df

# Main app
def main():
    st.markdown('<h1 class="main-header">Heart Disease Risk Prediction</h1>', unsafe_allow_html=True)
    
    # Load model
    model, feature_names, feature_importance, config = load_model()
    
    if model is None:
        st.warning("Please run the train_model.py script first to generate the model files.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Information", "About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Enter patient information to predict heart disease risk.</p>', unsafe_allow_html=True)
        
        # Create two columns for the form
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age', min_value=20, max_value=100, value=50)
            sex = st.radio('Sex', ['Male', 'Female'])
            cp = st.selectbox('Chest Pain Type', 
                             ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
            trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
            chol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
            fbs = st.checkbox('Fasting Blood Sugar > 120 mg/dl')
            
        with col2:
            restecg = st.selectbox('Resting ECG', 
                                  ['normal', 'st-t abnormality', 'lv hypertrophy'])
            thalch = st.number_input('Maximum Heart Rate', min_value=60, max_value=220, value=150)
            exang = st.checkbox('Exercise Induced Angina')
            oldpeak = st.number_input('ST Depression Induced by Exercise', 
                                     min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox('Slope of Peak Exercise ST Segment', 
                                ['upsloping', 'flat', 'downsloping'])
            ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 3, 0)
            thal = st.selectbox('Thalassemia', 
                               ['normal', 'fixed defect', 'reversable defect'])
        
        # Collect all inputs
        patient_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalch': thalch,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        # Prediction button
        if st.button('Predict Heart Disease Risk'):
            # Preprocess input
            features = preprocess_input(patient_data, feature_names)
            
            # Make prediction
            prediction_proba = model.predict_proba(features)[0][1]
            prediction = 1 if prediction_proba >= 0.5 else 0
            
            # Display result
            st.markdown('<h2 class="sub-header">Prediction Result</h2>', unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box high-risk">
                    <h3>⚠️ High Risk of Heart Disease</h3>
                    <p>The model predicts a {prediction_proba:.2%} probability of heart disease.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box low-risk">
                    <h3>✅ Low Risk of Heart Disease</h3>
                    <p>The model predicts a {prediction_proba:.2%} probability of heart disease.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show feature contributions (simplified)
            st.markdown('<h3 class="feature-importance">Key Factors in This Prediction</h3>', unsafe_allow_html=True)
            
            # Get feature importances from the model
            importances = model.feature_importances_
            
            # Create a DataFrame for visualization
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': features.values[0],
                'Importance': importances
            })
            
            # Calculate contribution (value * importance)
            feature_df['Contribution'] = feature_df['Value'] * feature_df['Importance']
            
            # Sort by absolute contribution
            feature_df = feature_df.iloc[(-feature_df['Contribution'].abs()).argsort()]
            
            # Take top 5 features
            top_features = feature_df.head(5)
            
            # Create a horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(
                [name.replace('_', ' ').title() for name in top_features['Feature']], 
                top_features['Contribution']
            )
            
            # Color bars based on positive/negative contribution
            for i, bar in enumerate(bars):
                if top_features['Contribution'].iloc[i] > 0:
                    bar.set_color('#FF4B4B')
                else:
                    bar.set_color('#4BC0C0')
            
            ax.set_xlabel('Contribution to Prediction')
            ax.set_title('Top Factors Influencing This Prediction')
            
            st.pyplot(fig)
            
            st.markdown("""
            <p class="info-text">Note: This is a simplified model and should not replace professional medical advice.</p>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
        
        # Display model configuration
        st.markdown('<h3>Model Configuration</h3>', unsafe_allow_html=True)
        st.write(f"Model Type: Random Forest Classifier")
        st.write(f"Number of Trees: {config['n_estimators']}")
        st.write(f"Maximum Tree Depth: {config['max_depth']}")
        st.write(f"Minimum Samples to Split: {config['min_samples_split']}")
        st.write(f"Feature Selection: {config['max_features']}")
        
        # Display feature importance
        st.markdown('<h3>Feature Importance</h3>', unsafe_allow_html=True)
        
        # Create a bar chart of feature importance
        top_features = feature_importance[:10]  # Top 10 features
        
        feature_names_display = [f"{item['feature'].replace('_', ' ').title()}" for item in top_features]
        importance_values = [item['importance'] for item in top_features]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(feature_names_display, importance_values)
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Most Important Features')
        
        st.pyplot(fig)
        
        # Display anti-overfitting measures
        st.markdown('<h3>Anti-Overfitting Measures</h3>', unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li><strong>Limited Tree Depth:</strong> Restricts how complex each decision tree can become, preventing it from memorizing noise in the training data.</li>
            <li><strong>Minimum Samples Split:</strong> Requires a minimum number of samples before making a decision at a node, reducing the chance of learning from too few examples.</li>
            <li><strong>Feature Subsampling:</strong> Each tree only considers a subset of features when making decisions, increasing diversity among trees.</li>
            <li><strong>Ensemble Method:</strong> Random Forest combines multiple trees, which helps reduce variance and prevents overfitting.</li>
            <li><strong>Cross-Validation:</strong> The model was evaluated using 5-fold cross-validation to ensure stable performance across different subsets of data.</li>
        </ul>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <p>This application predicts the risk of heart disease based on patient clinical data. The model was trained on the Heart Disease UCI dataset, which contains anonymized patient records with various clinical measurements.</p>
        
        <h3>Dataset Features</h3>
        <ul>
            <li><strong>age:</strong> Age in years</li>
            <li><strong>sex:</strong> Gender (Male/Female)</li>
            <li><strong>cp:</strong> Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic)</li>
            <li><strong>trestbps:</strong> Resting blood pressure in mm Hg</li>
            <li><strong>chol:</strong> Serum cholesterol in mg/dl</li>
            <li><strong>fbs:</strong> Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)</li>
            <li><strong>restecg:</strong> Resting electrocardiographic results</li>
            <li><strong>thalch:</strong> Maximum heart rate achieved</li>
            <li><strong>exang:</strong> Exercise induced angina (1 = yes; 0 = no)</li>
            <li><strong>oldpeak:</strong> ST depression induced by exercise relative to rest</li>
            <li><strong>slope:</strong> Slope of the peak exercise ST segment</li>
            <li><strong>ca:</strong> Number of major vessels colored by fluoroscopy (0-3)</li>
            <li><strong>thal:</strong> Thalassemia (normal, fixed defect, reversable defect)</li>
        </ul>
        
        <h3>Model Performance</h3>
        <p>The model was trained using a Random Forest classifier with hyperparameters specifically tuned to prevent overfitting on this small dataset. Cross-validation was used to ensure the model generalizes well to new data.</p>
        
        <h3>Disclaimer</h3>
        <p>This application is for educational purposes only and should not be used for medical diagnosis. Always consult with a healthcare professional for medical advice.</p>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()