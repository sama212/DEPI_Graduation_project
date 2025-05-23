import os
import requests
import streamlit as st
import json

@st.cache_data
def generate_advanced_explanation(prediction, probability, patient_data, feature_contributions, feature_names):
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