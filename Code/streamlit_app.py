import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Sample school data for lookup (you could expand this with real data)
SAMPLE_SCHOOLS = {
    "Harvard University": {
        "ownership": 2, "student_size": 23000, "admission_rate": 0.04, "sat_average": 1520,
        "act_midpoint": 34, "tuition_in_state": 54000, "tuition_out_state": 54000,
        "total_cost": 79000, "completion_rate_4yr": 0.97, "state_encoded": 21
    },
    "CUNY Baruch College": {
        "ownership": 1, "student_size": 18000, "admission_rate": 0.39, "sat_average": 1350,
        "act_midpoint": 29, "tuition_in_state": 7000, "tuition_out_state": 18000,
        "total_cost": 13500, "completion_rate_4yr": 0.69, "state_encoded": 32
    },
    "University of California-Berkeley": {
        "ownership": 1, "student_size": 45000, "admission_rate": 0.16, "sat_average": 1430,
        "act_midpoint": 32, "tuition_in_state": 14000, "tuition_out_state": 44000,
        "total_cost": 36000, "completion_rate_4yr": 0.92, "state_encoded": 4
    },
    "New York University": {
        "ownership": 2, "student_size": 51000, "admission_rate": 0.20, "sat_average": 1470,
        "act_midpoint": 33, "tuition_in_state": 58000, "tuition_out_state": 58000,
        "total_cost": 78000, "completion_rate_4yr": 0.85, "state_encoded": 32
    }
}

@st.cache_data
def load_model():
    try:
        with open('data/processed/roi_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['feature_names']
    except FileNotFoundError:
        st.error("Model not found. Please run the ML pipeline first.")
        return None, None

def calculate_realistic_metrics(roi_percent, total_annual_cost):
    """Calculate more realistic payback and rating metrics"""
    # 4-year total education cost
    total_education_cost = total_annual_cost * 4
    
    # More realistic salary assumptions
    high_school_median = 35000
    estimated_college_salary = high_school_median * (1 + roi_percent/100)
    annual_premium = estimated_college_salary - high_school_median
    
    # Payback period in years (how long to recover the 4-year investment)
    if annual_premium > 0:
        payback_years = total_education_cost / annual_premium
    else:
        payback_years = float('inf')
    
    return total_education_cost, payback_years, estimated_college_salary

def get_roi_rating(roi_percent, payback_years):
    """More stringent ROI rating system"""
    if payback_years <= 5 and roi_percent > 1000:
        return "Excellent", "green"
    elif payback_years <= 8 and roi_percent > 500:
        return "Good", "blue" 
    elif payback_years <= 12 and roi_percent > 200:
        return "Fair", "orange"
    elif payback_years <= 20 and roi_percent > 0:
        return "Poor", "red"
    else:
        return "Very Poor", "red"

def main():
    st.title("College ROI Predictor")
    st.write("Predict college return on investment using machine learning")
    
    # Load model
    model, feature_names = load_model()
    
    if model is None:
        st.stop()
    
    # Input method selection
    input_method = st.radio("How would you like to input data?", 
                           ["Manual Input", "School Lookup"])
    
    if input_method == "School Lookup":
        st.header("School Lookup")
        selected_school = st.selectbox("Select a school:", list(SAMPLE_SCHOOLS.keys()))
        
        if selected_school:
            school_data = SAMPLE_SCHOOLS[selected_school]
            st.write(f"**{selected_school}** - Data loaded automatically")
            
            # You can still adjust with sliders
            st.subheader("Adjust values if needed:")
            col1, col2 = st.columns(2)
            
            with col1:
                ownership = st.selectbox("School Type", 
                                        options=[1, 2, 3], 
                                        index=[1, 2, 3].index(school_data["ownership"]),
                                        format_func=lambda x: {1: "Public", 2: "Private Nonprofit", 3: "Private For-Profit"}[x])
                
                total_cost = st.slider("Total Annual Cost", 10000, 90000, school_data["total_cost"], 1000)
                sat_average = st.slider("Average SAT Score", 800, 1600, school_data["sat_average"], 10)
                completion_rate = st.slider("Graduation Rate", 0.0, 1.0, school_data["completion_rate_4yr"], 0.01)
                student_size = st.slider("Student Body Size", 500, 60000, school_data["student_size"], 100)
            
            with col2:
                admission_rate = st.slider("Admission Rate", 0.0, 1.0, school_data["admission_rate"], 0.01)
                tuition_in_state = st.slider("In-State Tuition", 5000, 70000, school_data["tuition_in_state"], 500)
                tuition_out_state = st.slider("Out-of-State Tuition", 8000, 90000, school_data["tuition_out_state"], 500)
                act_midpoint = st.slider("Average ACT Score", 10, 36, school_data["act_midpoint"], 1)
                state_encoded = st.slider("State Factor", 0, 50, school_data["state_encoded"], 1)
    
    else:  # Manual Input
        st.header("Manual Input - School Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ownership = st.selectbox("School Type", 
                                    options=[1, 2, 3], 
                                    format_func=lambda x: {1: "Public", 2: "Private Nonprofit", 3: "Private For-Profit"}[x])
            
            total_cost = st.slider("Total Annual Cost", 10000, 90000, 35000, 1000)
            sat_average = st.slider("Average SAT Score", 800, 1600, 1200, 10)
            completion_rate = st.slider("Graduation Rate", 0.0, 1.0, 0.65, 0.01)
            student_size = st.slider("Student Body Size", 500, 60000, 8000, 100)
        
        with col2:
            admission_rate = st.slider("Admission Rate", 0.0, 1.0, 0.45, 0.01)
            tuition_in_state = st.slider("In-State Tuition", 5000, 70000, 20000, 500)
            tuition_out_state = st.slider("Out-of-State Tuition", 8000, 90000, 35000, 500)
            act_midpoint = st.slider("Average ACT Score", 10, 36, 26, 1)
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
        
        # Calculate realistic metrics
        total_education_cost, payback_years, estimated_salary = calculate_realistic_metrics(roi_percent, total_cost)
        quality, color = get_roi_rating(roi_percent, payback_years)
        
        st.header("Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Predicted ROI", f"{roi_percent:.0f}%")
        
        with col2:
            if payback_years < 50:
                st.metric("Payback Period", f"{payback_years:.1f} years")
            else:
                st.metric("Payback Period", "50+ years")
        
        with col3:
            st.metric("4-Year Total Cost", f"${total_education_cost:,.0f}")
        
        with col4:
            st.metric("Est. Starting Salary", f"${estimated_salary:,.0f}")
        
        # ROI Rating
        st.subheader("Investment Rating")
        st.markdown(f"**Overall Rating:** :{color}[{quality}]")
        
        # More detailed analysis
        st.subheader("Detailed Analysis")
        
        if quality == "Excellent":
            st.success(f"Outstanding investment! You'll recover your {total_education_cost:,.0f} investment in just {payback_years:.1f} years. This profile matches top-performing affordable schools.")
        elif quality == "Good":
            st.info(f"Solid investment. The {payback_years:.1f}-year payback period is reasonable for the {total_education_cost:,.0f} total cost.")
        elif quality == "Fair":
            st.warning(f"Moderate investment. Consider if the {payback_years:.1f}-year payback justifies the {total_education_cost:,.0f} cost. Look for more affordable alternatives.")
        elif quality == "Poor":
            st.error(f"Questionable investment. The {payback_years:.1f}-year payback period is quite long for a {total_education_cost:,.0f} investment.")
        else:
            st.error(f"Poor investment prospects. Consider much more affordable alternatives or different career paths.")
        
        # Comparison context
        st.subheader("For Context")
        st.write("**Excellent schools** typically have:")
        st.write("- Payback periods under 5 years")
        st.write("- ROI over 1000%")
        st.write("- Total costs under $60,000")
        st.write("- High graduation rates (>80%)")

if __name__ == "__main__":
    main()
