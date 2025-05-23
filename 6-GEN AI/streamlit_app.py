import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
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
.ai-explanation {
    background-color: rgba(240, 240, 240, 0.5);
    border: 1px solid rgba(200, 200, 200, 0.5);
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}
h1, h2, h3, h4 {
    color: #333;
}
.stMarkdown ul li {
    margin-bottom: 0.5rem;
}
.stMarkdown h3 {
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# Advanced AI Explanation Functions
@st.cache_data
def generate_advanced_explanation(prediction, probability, patient_data, feature_contributions, feature_names):
    """
    Generate a comprehensive, medically sound explanation of heart disease risk prediction.
    
    This function creates a detailed medical explanation that:
    1. Analyzes specific patient data in medical context
    2. Explains the physiological mechanisms behind risk factors
    3. Provides evidence-based insights on risk modification
    4. Offers personalized recommendations based on the specific risk profile
    
    Args:
        prediction: Binary prediction (0 or 1)
        probability: Prediction probability
        patient_data: Dictionary of patient information
        feature_contributions: DataFrame with feature contributions
        feature_names: List of all feature names
    
    Returns:
        A detailed medical explanation
    """
    # Extract top positive and negative contributors
    positive_contributors = feature_contributions[feature_contributions['Contribution'] > 0].head(3)
    negative_contributors = feature_contributions[feature_contributions['Contribution'] < 0].head(2)
    
    # Create medical context for each significant factor
    medical_context = []
    
    # Age context
    if 'age' in patient_data:
        age = patient_data['age']
        if age > 65:
            medical_context.append(f"At {age} years, cardiovascular aging has likely resulted in arterial stiffening and endothelial dysfunction, which significantly increases atherosclerotic risk.")
        elif age > 45:
            medical_context.append(f"At {age} years, age-related vascular changes are beginning to manifest, with moderate impact on cardiovascular risk.")
        else:
            medical_context.append(f"At {age} years, your age presents a relatively lower cardiovascular risk factor, though family history and other factors remain important.")
    
    # Cholesterol context
    if 'chol' in patient_data:
        chol = patient_data['chol']
        if chol > 240:
            medical_context.append(f"Your total cholesterol of {chol} mg/dL is significantly elevated. This increases risk through accelerated atherosclerotic plaque formation, particularly when LDL fraction is high.")
        elif chol > 200:
            medical_context.append(f"Your total cholesterol of {chol} mg/dL is borderline high, potentially contributing to atherosclerotic processes if LDL/HDL ratio is unfavorable.")
        else:
            medical_context.append(f"Your total cholesterol of {chol} mg/dL is within normal range, which is favorable for cardiovascular health.")
    
    # Blood pressure context
    if 'trestbps' in patient_data:
        bp = patient_data['trestbps']
        if bp >= 140:
            medical_context.append(f"Your resting blood pressure of {bp} mmHg indicates hypertension, which increases cardiac workload and contributes to left ventricular hypertrophy and vascular damage over time.")
        elif bp >= 120:
            medical_context.append(f"Your resting blood pressure of {bp} mmHg is elevated, suggesting early-stage vascular stress that may accelerate atherosclerosis.")
        else:
            medical_context.append(f"Your resting blood pressure of {bp} mmHg is optimal, reducing strain on your cardiovascular system.")
    
    # Heart rate context
    if 'thalch' in patient_data:
        hr = patient_data['thalch']
        if hr > 100:
            medical_context.append(f"Your maximum heart rate of {hr} bpm suggests potential chronotropic competence, though elevated resting heart rate can indicate decreased parasympathetic tone and increased cardiovascular risk.")
        elif hr < 80:
            medical_context.append(f"Your maximum heart rate of {hr} bpm may indicate good cardiovascular conditioning, associated with lower long-term cardiovascular risk.")
    
    # ST depression context
    if 'oldpeak' in patient_data:
        st = patient_data['oldpeak']
        if st >= 2.0:
            medical_context.append(f"Your ST depression of {st} mm during exercise suggests significant myocardial ischemia, indicating reduced coronary perfusion under stress.")
        elif st > 0.5:
            medical_context.append(f"Your ST depression of {st} mm during exercise indicates moderate exercise-induced ischemia, a marker of coronary artery disease.")
    
    # Vessels context
    if 'ca' in patient_data:
        ca = patient_data['ca']
        if ca > 0:
            medical_context.append(f"Fluoroscopy reveals {ca} major vessel(s) with significant calcification, indicating established atherosclerotic disease and reduced coronary reserve.")
    
    # Create a comprehensive explanation
    risk_level = "high" if prediction == 1 else "low"
    probability_percent = f"{probability * 100:.1f}%"
    
    # Construct a detailed prompt for the AI
    system_prompt = """
    You are a cardiologist with expertise in preventive cardiology and risk assessment. 
    Provide a detailed, evidence-based explanation of a patient's heart disease risk assessment.
    Your explanation should:
    1. Be scientifically accurate and reflect current medical understanding
    2. Explain physiological mechanisms behind risk factors
    3. Provide specific, actionable recommendations based on the patient's profile
    4. Use professional medical terminology while remaining accessible
    5. Be logically structured and demonstrate clinical reasoning
    6. Cite specific relationships between risk factors when relevant
    7. Avoid generic advice and instead tailor recommendations to the specific risk profile
    """
    
    user_prompt = f"""
    Patient Risk Assessment: {risk_level} risk of heart disease ({probability_percent} probability)
    
    Patient Data:
    - Age: {patient_data.get('age')}
    - Sex: {patient_data.get('sex')}
    - Chest Pain Type: {patient_data.get('cp')}
    - Resting Blood Pressure: {patient_data.get('trestbps')} mmHg
    - Cholesterol: {patient_data.get('chol')} mg/dl
    - Fasting Blood Sugar > 120 mg/dl: {"Yes" if patient_data.get('fbs') else "No"}
    - Resting ECG: {patient_data.get('restecg')}
    - Maximum Heart Rate: {patient_data.get('thalch')}
    - Exercise Induced Angina: {"Yes" if patient_data.get('exang') else "No"}
    - ST Depression: {patient_data.get('oldpeak')}
    - Slope of Peak Exercise ST Segment: {patient_data.get('slope')}
    - Number of Major Vessels: {patient_data.get('ca')}
    - Thalassemia: {patient_data.get('thal')}
    
    Medical Context:
    {' '.join(medical_context)}
    
    Top Risk-Increasing Factors (with contribution scores):
    {positive_contributors[['Feature', 'Contribution']].to_string(index=False)}
    
    Top Risk-Decreasing Factors (with contribution scores):
    {negative_contributors[['Feature', 'Contribution']].to_string(index=False)}
    
    Provide a comprehensive cardiological assessment explaining this patient's heart disease risk, the physiological significance of the key contributing factors, their interactions, and specific evidence-based recommendations. Include a paragraph on how the patient might modify their risk profile.
    """
    
    # Get API token from environment variable
    api_token = os.environ.get("HF_API_TOKEN")
    
    if not api_token:
        return generate_advanced_fallback(prediction, probability, patient_data, feature_contributions)
    
    # Use a more advanced model for medical explanations
    model = "meta-llama/Llama-2-70b-chat-hf"  # More advanced model
    
    # If the above model is not accessible, fall back to these alternatives
    fallback_models = [
        "google/flan-t5-xxl",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "google/flan-t5-xl"
    ]
    
    # Try the primary model first
    explanation = try_generate_with_model(model, system_prompt, user_prompt, api_token)
    
    # If primary model fails, try fallbacks
    if not explanation:
        for fallback_model in fallback_models:
            explanation = try_generate_with_model(fallback_model, system_prompt, user_prompt, api_token)
            if explanation:
                break
    
    # If all models fail, use the fallback
    if not explanation:
        explanation = generate_advanced_fallback(prediction, probability, patient_data, feature_contributions)
    
    return explanation

def try_generate_with_model(model, system_prompt, user_prompt, api_token):
    """Try to generate an explanation with a specific model"""
    try:
        # API endpoint
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        
        # Headers for the request
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        
        # Payload for the request
        payload = {
            "inputs": user_prompt,
            "parameters": {
                "max_length": 1000,
                "temperature": 0.3,  # Lower temperature for more focused, deterministic responses
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        # Add system prompt if model supports it
        if "llama" in model.lower() or "mistral" in model.lower():
            payload["parameters"]["system_prompt"] = system_prompt
        
        # Make the request
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract the generated text based on model response format
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "generated_text" in result[0]:
                    return result[0]["generated_text"]
                else:
                    return str(result[0])
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                return str(result)
        
        return None
    except Exception:
        return None

def generate_advanced_fallback(prediction, probability, patient_data, feature_contributions):
    """Generate a sophisticated fallback explanation when API calls fail"""
    risk_level = "high" if prediction == 1 else "low"
    probability_percent = f"{probability * 100:.1f}%"
    
    # Extract key risk factors
    top_positive = feature_contributions[feature_contributions['Contribution'] > 0].head(3)
    top_negative = feature_contributions[feature_contributions['Contribution'] < 0].head(2)
    
    # Create detailed medical explanations for common risk factors
    factor_explanations = {
        "age": "Age is a non-modifiable risk factor that affects cardiovascular health through progressive arterial stiffening, endothelial dysfunction, and reduced vascular compliance. Each decade of life approximately doubles coronary heart disease risk.",
        
        "sex_Male": "Male sex is associated with earlier onset of cardiovascular disease, partly due to the protective effects of estrogen in pre-menopausal women. Men typically develop coronary artery disease 7-10 years earlier than women.",
        
        "chol": "Elevated total cholesterol contributes to atherosclerosis through lipid deposition in arterial walls. Each 1% increase in total cholesterol is associated with approximately 2% increase in coronary heart disease risk.",
        
        "trestbps": "Hypertension increases cardiac workload and contributes to left ventricular hypertrophy, endothelial dysfunction, and accelerated atherosclerosis. Each 20 mmHg increase in systolic blood pressure doubles cardiovascular risk.",
        
        "ca": "The number of major vessels with significant stenosis directly correlates with reduced coronary flow reserve and increased risk of ischemic events. Multi-vessel disease significantly worsens prognosis compared to single-vessel disease.",
        
        "thalch": "Maximum heart rate achieved during exercise testing provides insight into chronotropic competence. Reduced heart rate response to exercise (chronotropic incompetence) is associated with increased cardiovascular mortality.",
        
        "cp_asymptomatic": "Asymptomatic presentation can indicate silent ischemia, which carries significant prognostic implications as it may delay diagnosis and treatment. Approximately 25% of myocardial infarctions are clinically silent.",
        
        "oldpeak": "ST-segment depression during exercise represents subendocardial ischemia. The magnitude of ST depression correlates with the severity of coronary artery disease and extent of myocardium at risk.",
        
        "exang": "Exercise-induced angina indicates inadequate myocardial oxygen supply during increased demand, strongly suggesting obstructive coronary artery disease."
    }
    
    # Build a comprehensive explanation
    if prediction == 1:
        explanation = f"""
# Comprehensive Cardiovascular Risk Assessment

## Risk Profile: HIGH ({probability_percent})

Your cardiovascular risk assessment indicates a **high probability ({probability_percent}) of coronary artery disease**. This assessment is based on a comprehensive analysis of your clinical parameters and their physiological impact on cardiovascular function.

### Primary Risk Factors

{build_factor_explanation(top_positive, factor_explanations)}

### Protective Factors

{build_factor_explanation(top_negative, factor_explanations)}

### Clinical Interpretation

The constellation of risk factors in your profile suggests significant atherosclerotic burden and potential coronary flow limitation. The interaction between these factors creates a synergistic effect that amplifies overall cardiovascular risk beyond the sum of individual factors.

### Recommendations

1. **Immediate Consultation**: Given your high-risk profile, consultation with a cardiologist is strongly recommended for comprehensive evaluation, which may include:
   - Stress testing (exercise or pharmacological)
   - Coronary calcium scoring
   - Advanced lipid profiling
   - Consideration of coronary angiography based on symptomatology

2. **Risk Factor Modification**:
   - Implement therapeutic lifestyle changes including Mediterranean or DASH dietary pattern
   - Structured physical activity program (150+ minutes/week of moderate-intensity exercise)
   - Smoking cessation if applicable
   - Weight management targeting BMI < 25 kg/m²

3. **Pharmacological Considerations**:
   - Statin therapy may be indicated for lipid management
   - Antihypertensive therapy if blood pressure exceeds 130/80 mmHg
   - Antiplatelet therapy may be considered based on overall risk profile

4. **Monitoring Strategy**:
   - Regular blood pressure monitoring
   - Periodic lipid profile assessment
   - Annual cardiovascular risk reassessment
   - Vigilant monitoring for evolving symptoms

This assessment is based on statistical modeling and should be integrated with clinical judgment and additional diagnostic testing for comprehensive care planning.
        """
    else:
        explanation = f"""
# Comprehensive Cardiovascular Risk Assessment

## Risk Profile: LOW ({probability_percent})

Your cardiovascular risk assessment indicates a **low probability ({probability_percent}) of coronary artery disease**. This favorable assessment is based on analysis of your clinical parameters and their physiological impact on cardiovascular function.

### Protective Factors

{build_factor_explanation(top_negative, factor_explanations)}

### Residual Risk Factors

{build_factor_explanation(top_positive, factor_explanations)}

### Clinical Interpretation

Your overall risk profile demonstrates a predominance of cardioprotective factors over pathogenic ones. The physiological significance of this balance suggests preserved coronary flow reserve and vascular function. However, even low-risk profiles benefit from preventive strategies to maintain cardiovascular health.

### Recommendations

1. **Preventive Strategy**:
   - Maintain current protective factors
   - Annual primary care evaluation for cardiovascular risk assessment
   - Regular physical activity (150+ minutes/week of moderate-intensity exercise)
   - Heart-healthy dietary pattern rich in fruits, vegetables, whole grains, and lean proteins

2. **Risk Factor Optimization**:
   - Address any modifiable risk factors identified in your profile
   - Maintain blood pressure below 120/80 mmHg
   - Target LDL cholesterol < 100 mg/dL
   - Maintain healthy weight (BMI 18.5-24.9 kg/m²)

3. **Monitoring Approach**:
   - Blood pressure evaluation at least annually
   - Lipid profile assessment every 5 years
   - Blood glucose screening as recommended by guidelines
   - Awareness of cardiovascular symptoms that would warrant reassessment

This favorable assessment should reinforce positive health behaviors while maintaining vigilance for evolving risk factors over time. Cardiovascular risk is dynamic and requires ongoing attention to preventive strategies.
        """
    
    return explanation

def build_factor_explanation(factors_df, explanations_dict):
    """Build detailed explanations for a set of factors"""
    result = ""
    for _, row in factors_df.iterrows():
        feature = row['Feature']
        contribution = row['Contribution']
        
        # Find the best matching explanation
        explanation = None
        for key, value in explanations_dict.items():
            if key in feature:
                explanation = value
                break
        
        if explanation:
            result += f"**{feature.replace('_', ' ').title()}** (Impact: {contribution:.4f}): {explanation}\n\n"
        else:
            result += f"**{feature.replace('_', ' ').title()}** (Impact: {contribution:.4f}): This factor contributes significantly to your overall risk assessment.\n\n"
    
    return result

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
            
            # Take top 5 features for visualization
            top_features = feature_df.head(5)
            
            # Generate advanced AI explanation
            with st.spinner("Generating comprehensive medical assessment..."):
                ai_explanation = generate_advanced_explanation(
                    prediction, 
                    prediction_proba, 
                    patient_data,
                    feature_df,  # Pass the full feature DataFrame
                    feature_names
                )
            
            # Display AI explanation
            st.markdown('<h3 class="sub-header">🩺 Advanced Medical Assessment</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-explanation">{ai_explanation}</div>', unsafe_allow_html=True)
            
            # Show feature contributions (simplified)
            st.markdown('<h3 class="feature-importance">Key Factors in This Prediction</h3>', unsafe_allow_html=True)
            
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
        
        <h3>Advanced AI Explanation</h3>
        <p>This application includes an advanced AI-powered explanation feature that provides comprehensive, medically sound interpretations of the prediction results. The system analyzes the specific risk factors and their physiological significance, offering detailed insights that would typically require consultation with a medical professional.</p>
        
        <h3>Disclaimer</h3>
        <p>This application is for educational purposes only and should not be used for medical diagnosis. Always consult with a healthcare professional for medical advice.</p>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()