import os
import requests
import streamlit as st

# Cache the explanation to avoid repeated API calls for the same input
@st.cache_data
def generate_explanation(prediction, probability, top_features):
    """
    Generate a natural language explanation of the heart disease prediction using Hugging Face.
    
    Args:
        prediction: Binary prediction (0 or 1)
        probability: Prediction probability
        top_features: List of tuples containing (feature_name, contribution_value)
    
    Returns:
        A string containing the explanation
    """
    # Format the features for the prompt
    feature_text = "\n".join([f"- {feature.replace('_', ' ').title()}: {value:.4f}" 
                             for feature, value in top_features])
    
    risk_level = "high" if prediction == 1 else "low"
    probability_percent = f"{probability * 100:.1f}%"
    
    prompt = f"""
    As a medical AI assistant, explain the following heart disease risk prediction to a patient:
    
    Risk Level: {risk_level} risk ({probability_percent} probability)
    
    Top contributing factors (positive values increase risk, negative values decrease risk):
    {feature_text}
    
    Provide a brief, clear explanation of what this means in simple terms. Include:
    1. What the risk level means
    2. An explanation of the top 2-3 factors and how they affect heart disease risk
    3. A brief general recommendation based on the risk level
    
    Keep the explanation under 200 words, conversational, and easy to understand.
    """
    
    # Get API token from environment variable
    api_token = os.environ.get("HF_API_TOKEN")
    
    if not api_token:
        return "Hugging Face API token not found. Please set the HF_API_TOKEN environment variable."
    
    # You can choose different models based on your needs
    # Some good options for medical explanations:
    # - "google/flan-t5-xl" (smaller but effective)
    # - "meta-llama/Llama-2-7b-chat-hf" (if you have access)
    # - "mistralai/Mistral-7B-Instruct-v0.2" (good performance)
    model = "google/flan-t5-xl"  # A good default option that's free to use
    
    # API endpoint
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    
    # Headers for the request
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Payload for the request
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 500,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    
    try:
        # Make the request
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the response
        result = response.json()
        
        # Different models return results in different formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            else:
                return str(result[0])
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        else:
            return str(result)
            
    except Exception as e:
        # Fallback explanation if the API call fails
        return generate_fallback_explanation(prediction, probability, top_features, str(e))

def generate_fallback_explanation(prediction, probability, top_features, error_msg):
    """Generate a fallback explanation when the API call fails."""
    risk_level = "high" if prediction == 1 else "low"
    probability_percent = f"{probability * 100:.1f}%"
    
    # Get the top 3 features
    top_3_features = top_features[:3]
    feature_text = ""
    
    for feature, value in top_3_features:
        feature_name = feature.replace('_', ' ').title()
        if value > 0:
            feature_text += f"- {feature_name} increases your risk\n"
        else:
            feature_text += f"- {feature_name} decreases your risk\n"
    
    # Basic explanation based on risk level
    if prediction == 1:
        explanation = f"""
Your heart disease risk is high ({probability_percent} probability).

Key factors affecting your risk:
{feature_text}

With a high risk level, it's recommended to consult with a healthcare provider for a thorough evaluation and personalized advice.

Note: This is a simplified model and should not replace professional medical advice.

(API Error: {error_msg})
        """
    else:
        explanation = f"""
Your heart disease risk is low ({probability_percent} probability).

Key factors affecting your risk:
{feature_text}

Even with a low risk, maintaining heart-healthy habits is important. Continue with regular check-ups and a healthy lifestyle.

Note: This is a simplified model and should not replace professional medical advice.

(API Error: {error_msg})
        """
    
    return explanation