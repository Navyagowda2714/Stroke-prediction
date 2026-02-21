from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import google.generativeai as genai
import os
import logging

app = Flask(__name__)

# Configure logging for better server-side error visibility
logging.basicConfig(level=logging.INFO)

# --- Load ML Model and Scaler ---
model = None
scaler = None

try:
    # Load your trained ML model
    model = pickle.load(open("model_pickle.pkl", 'rb'))
    app.logger.info("ML model 'model_pickle.pkl' loaded successfully.")
except FileNotFoundError:
    app.logger.error("Error: 'model_pickle.pkl' not found. Ensure your trained model is saved and in the correct path.")
except Exception as e:
    app.logger.error(f"Error loading model_pickle.pkl: {e}")

try:
    # Load your pre-fitted scaler
    scaler = pickle.load(open("scaler.pkl", "rb"))
    app.logger.info("Scaler 'scaler.pkl' loaded successfully.")
except FileNotFoundError:
    app.logger.error("Error: 'scaler.pkl' not found. Ensure your fitted scaler is saved and in the correct path.")
    app.logger.error("You need to fit StandardScaler on your training data and save it (e.g., pickle.dump(scaler, open('scaler.pkl', 'wb'))).")
except Exception as e:
    app.logger.error(f"Error loading scaler.pkl: {e}")


# --- Configure Google Gemini API ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
llm_model = None

if not GEMINI_API_KEY:
    app.logger.warning("WARNING: GOOGLE_API_KEY environment variable not set. LLM functionality might be limited.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        llm_model = genai.GenerativeModel('gemini-pro')
        app.logger.info("Google Gemini API configured successfully.")
    except Exception as e:
        app.logger.error(f"Error configuring Gemini API: {e}", exc_info=True)
        llm_model = None


# --- Helper Functions for LLM Prompt Construction ---

def format_patient_data_for_llm(form_data):
    """
    Converts raw form data into a natural language string for the LLM.
    """
    formatted_str = "The patient has the following characteristics:\n"
    formatted_str += f"- Gender: {form_data.get('gender', 'N/A')}\n"
    formatted_str += f"- Age: {form_data.get('age', 'N/A')} years old\n"
    formatted_str += f"- Hypertension: {'Yes' if form_data.get('hypertension') == '1' else 'No'}\n"
    formatted_str += f"- Heart Disease: {'Yes' if form_data.get('disease') == '1' else 'No'}\n"
    formatted_str += f"- Ever Married: {form_data.get('married', 'N/A')}\n"
    formatted_str += f"- Work Type: {form_data.get('work', 'N/A').replace('_', ' ').title()}\n"
    formatted_str += f"- Residence Type: {form_data.get('residence', 'N/A')}\n"
    formatted_str += f"- Average Glucose Level: {form_data.get('glucose', 'N/A')} mg/dL\n"
    formatted_str += f"- BMI (Body Mass Index): {form_data.get('bmi', 'N/A')}\n"
    formatted_str += f"- Smoking Status: {form_data.get('smoking', 'N/A').replace('_', ' ').title()}\n"
    return formatted_str

def get_risk_level_description(probability):
    """
    Converts probability to risk level description
    """
    if probability < 0.1:
        return "Very Low Risk"
    elif probability < 0.3:
        return "Low Risk"
    elif probability < 0.5:
        return "Moderate Risk"
    elif probability < 0.7:
        return "High Risk"
    else:
        return "Very High Risk"

def get_influential_factors_for_llm(form_data, is_high_risk):
    """
    Identifies and formats potentially influential factors based on input values.
    """
    factors = []
    age = int(form_data.get('age', 0))
    glucose = float(form_data.get('glucose', 0.0))
    bmi = float(form_data.get('bmi', 0.0))
    hypertension = int(form_data.get('hypertension', 0))
    heart_disease = int(form_data.get('disease', 0))
    smoking = form_data.get('smoking', '')

    if is_high_risk:
        if age >= 65:
            factors.append("older age")
        if hypertension == 1:
            factors.append("presence of hypertension (high blood pressure)")
        if heart_disease == 1:
            factors.append("pre-existing heart disease")
        if glucose > 140:
            factors.append("elevated average glucose level")
        if bmi > 30:
            factors.append("a higher Body Mass Index (BMI)")
        if smoking == 'smokes':
            factors.append("active smoking status")
        elif smoking == 'formerly smoked':
            factors.append("a history of smoking")

        if factors:
            return f"Key factors that may contribute to this assessment include: {', '.join(factors)}."
        else:
            return "The model considered various factors in its assessment."
    else:
        if age < 40:
            factors.append("younger age")
        if hypertension == 0:
            factors.append("absence of hypertension")
        if heart_disease == 0:
            factors.append("absence of heart disease")
        if glucose < 100:
            factors.append("healthy average glucose level")
        if bmi < 25:
            factors.append("a healthy Body Mass Index (BMI)")
        if smoking == 'never smoked':
            factors.append("never having smoked")

        if factors:
            return f"Factors contributing to this assessment include: {', '.join(factors)}."
        else:
            return "The model considered various factors in its assessment."


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html", prediction_text=None, llm_explanation=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            error_msg = "ML model or scaler not loaded on server. Please check server logs."
            app.logger.error(error_msg)
            return jsonify({"error": error_msg}), 500

        # Expecting JSON data from the fetch API call
        form_data = request.json
        if not form_data:
            return jsonify({"error": "No input data provided in JSON format."}), 400

        # --- Data Preprocessing: Manual One-Hot Encoding ---
        age = int(form_data.get('age', 0))
        hypertension = int(form_data.get('hypertension', 0))
        heart_disease = int(form_data.get('disease', 0))
        glucose = float(form_data.get('glucose', 0.0))
        bmi = float(form_data.get('bmi', 0.0))

        # Gender (gender_Female, gender_Male, gender_Other)
        gender = form_data.get('gender', 'Other')
        gender_female = int(gender == "Female")
        gender_male = int(gender == "Male")
        gender_other = int(gender == "Other")

        # Ever Married (married_Yes)
        married_yes = int(form_data.get('married', 'No') == "Yes")

        # Work Type
        work = form_data.get('work', 'Private')
        work_type_Govt_job = int(work == "Govt_job")
        work_type_Never_worked = int(work == "Never_worked")
        work_type_Private = int(work == "Private")
        work_type_Self_employed = int(work == "Self-employed")
        work_type_children = int(work == "children")

        # Residence Type
        residence = form_data.get('residence', 'Urban')
        Residence_type_Urban = int(residence == "Urban")

        # Smoking Status
        smoking = form_data.get('smoking', 'Unknown')
        smoking_status_Unknown = int(smoking == 'Unknown')
        smoking_status_formerly_smoked = int(smoking == 'formerly smoked')
        smoking_status_never_smoked = int(smoking == 'never smoked')
        smoking_status_smokes = int(smoking == 'smokes')

        # Construct the final_features array in the EXACT order
        final_features_array = np.array([[
            age, hypertension, heart_disease, glucose, bmi,
            gender_female, gender_male, gender_other,
            married_yes,
            work_type_Govt_job, work_type_Never_worked, work_type_Private, work_type_Self_employed, work_type_children,
            Residence_type_Urban,
            smoking_status_Unknown, smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes
        ]])

        # --- Apply the loaded scaler's transform method ---
        final_features_scaled = scaler.transform(final_features_array)

        # --- ML Prediction with Enhanced Details ---
        prediction_raw = model.predict(final_features_scaled)[0]
        
        # Get probability scores for both classes
        prediction_proba_all = model.predict_proba(final_features_scaled)[0]
        no_stroke_prob = prediction_proba_all[0]  # Probability of no stroke (class 0)
        stroke_prob = prediction_proba_all[1]     # Probability of stroke (class 1)

        # Determine risk level
        is_high_risk = prediction_raw == 1
        risk_level = get_risk_level_description(stroke_prob)
        
        # Create detailed prediction text
        prediction_label = "STROKE RISK DETECTED" if is_high_risk else "LOW STROKE RISK"
        
        # Enhanced prediction display with exact probabilities
        prediction_text_display = f"""
        🏥 STROKE PREDICTION RESULTS:
        
        📊 Prediction: {prediction_label}
        🎯 Risk Level: {risk_level}
        
        📈 Detailed Probabilities:
        • No Stroke: {no_stroke_prob:.1%} ({no_stroke_prob:.4f})
        • Stroke Risk: {stroke_prob:.1%} ({stroke_prob:.4f})
        
        ⚖️ Model Confidence: {max(no_stroke_prob, stroke_prob):.1%}
        """

        # --- LLM Integration ---
        llm_explanation = "AI explanation could not be generated at this time."

        if llm_model:
            formatted_patient_info = format_patient_data_for_llm(form_data)
            influential_factors = get_influential_factors_for_llm(form_data, is_high_risk)

            prompt_text = f"""
            A machine learning model has assessed a patient with the following characteristics:
            {formatted_patient_info}

            The model predicts a **{prediction_label}** with:
            - No Stroke Probability: {no_stroke_prob:.1%}
            - Stroke Risk Probability: {stroke_prob:.1%}
            - Risk Level: {risk_level}
            
            {influential_factors}

            Please provide a concise, easy-to-understand explanation of what these risk factors mean in the context of stroke.
            Additionally, offer general, non-medical advice for stroke prevention or steps a person with these characteristics might consider, emphasizing a healthy lifestyle.
            **Important:** Do not provide any medical diagnosis, treatment, or specific medical advice. Always advise consulting a qualified healthcare professional.
            """
            try:
                response = llm_model.generate_content(prompt_text)
                llm_explanation = response.text
            except Exception as e:
                app.logger.error(f"Error calling LLM API: {e}", exc_info=True)
                llm_explanation = f"Error generating explanation from AI: {e}. Please ensure your API key is correct and try again."
        else:
            llm_explanation = "LLM functionality is disabled due to server configuration issues (e.g., missing API key)."

        # Return comprehensive JSON response
        return jsonify({
            "prediction_text": prediction_text_display,
            "llm_explanation": llm_explanation,
            "is_high_risk": 1 if is_high_risk else 0,
            "detailed_results": {
                "no_stroke_probability": float(no_stroke_prob),
                "stroke_probability": float(stroke_prob),
                "risk_level": risk_level,
                "model_confidence": float(max(no_stroke_prob, stroke_prob)),
                "prediction_label": prediction_label,
                "raw_prediction": int(prediction_raw)
            }
        })

    except Exception as e:
        app.logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)