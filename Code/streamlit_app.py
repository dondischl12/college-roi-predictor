import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained ROI model
@st.cache_data
def load_model():
    try:
        with open('data/processed/roi_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['feature_names']
    except FileNotFoundError:
        st.error("Model not found. Please run the ML pipeline first.")
        return None, None

def main():
    st.title("College ROI Predictor")
    st.write("Predict college return on investment using machine learning")
    
    # Load model
    model, feature_names = load_model()
    
    if model is None:
        st.stop()
    
    st.header("Input School Characteristics")
    
    # Simple input form
    col1, col2 = st.columns(2)
    
    with col1:
        ownership = st.selectbox("School Type", 
                                options=[1, 2, 3], 
                                format_func=lambda x: {1: "Public", 2: "Private Nonprofit", 3: "Private For-Profit"}[x])
        
        total_cost = st.slider("Total Annual Cost", 10000, 80000, 30000, 1000)
        
        sat_average = st.slider("Average SAT Score", 800, 1600, 1200, 10)
        
        completion_rate = st.slider("Graduation Rate", 0.0, 1.0, 0.6, 0.01)
        
        student_size = st.slider("Student Body Size", 500, 50000, 5000, 100)
    
    with col2:
        admission_rate = st.slider("Admission Rate", 0.0, 1.0, 0.5, 0.01)
        
        tuition_in_state = st.slider("In-State Tuition", 5000, 60000, 15000, 500)
        
        tuition_out_state = st.slider("Out-of-State Tuition", 8000, 80000, 25000, 500)
        
        act_midpoint = st.slider("Average ACT Score", 10, 36, 25, 1)
        
        state_encoded = st.slider("State Factor", 0, 50, 25, 1)
    
    # Calculate derived features
    tuition_ratio = tuition_out_state / tuition_in_state if tuition_in_state > 0 else 1
    selectivity_score = 1 - admission_rate
    cost_per_completion = total_cost / (completion_rate + 0.01)
    
    # Prepare input for model
    user_input = np.array([[
        ownership, student_size, admission_rate, sat_average, act_midpoint,
        tuition_in_state, tuition_out_state, total_cost, completion_rate,
        state_encoded, tuition_ratio, selectivity_score, cost_per_completion
    ]])
    
    # Make prediction
    if st.button("Predict ROI", type="primary"):
        prediction = model.predict(user_input)
        roi_percent = prediction[0]
        
        st.header("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted ROI", f"{roi_percent:.1f}%")
        
        with col2:
            payback_years = total_cost * 4 / (35000 * (roi_percent/100)) if roi_percent > 0 else float('inf')
            st.metric("Estimated Payback", f"{payback_years:.1f} years" if payback_years < 50 else "50+ years")
        
        with col3:
            if roi_percent > 3000:
                quality = "Excellent"
                color = "green"
            elif roi_percent > 2000:
                quality = "Good"
                color = "blue"
            elif roi_percent > 1000:
                quality = "Fair"
                color = "orange"
            else:
                quality = "Poor"
                color = "red"
            
            st.markdown(f"**ROI Rating:** :{color}[{quality}]")
        
        # Show interpretation
        st.subheader("What this means:")
        if roi_percent > 3000:
            st.success("This school profile suggests excellent return on investment, similar to top-performing affordable public schools.")
        elif roi_percent > 2000:
            st.info("This represents a good educational investment with solid returns.")
        elif roi_percent > 1000:
            st.warning("Fair return on investment. Consider if the benefits justify the costs.")
        else:
            st.error("This profile suggests poor return on investment. Consider more affordable alternatives.")

if __name__ == "__main__":
    main()
