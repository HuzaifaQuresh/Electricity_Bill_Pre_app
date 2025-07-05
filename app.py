import streamlit as st
import pandas as pd
import joblib
from electricity_model import ElectricityUsageModel  # Make sure this class is imported

# ========== Load Combined Model ==========
@st.cache_resource
def load_model():
    return joblib.load("electricity_usage_combined_model.pkl")

model = load_model()

# ========== App Title ==========
st.title("🔌 Electricity Unit Prediction App")
st.markdown("Predict next month's electricity usage based on user history.")

# ========== User Inputs ==========
UserID = st.text_input("Enter UserID:", value="26111-0000083")
current_units = st.number_input("Enter current month's units:", min_value=0.0, value=320.0)

# ========== Predict Button ==========
if st.button("Predict"):
    if UserID not in model.user_encoder.classes_:
        st.error("❌ UserID not found in trained model.")
    elif UserID not in model.lag_lookup:
        st.error("❌ No previous unit history found for this UserID.")
    else:
        encoded_id = model.user_encoder.transform([UserID])[0]

        cluster_row = model.df_cluster_map.loc[model.df_cluster_map['UserID'] == UserID, 'Cluster']
        if cluster_row.empty:
            st.error("❌ No cluster info found for this UserID.")
        else:
            cluster = int(cluster_row.values[0])
            prev_units = model.lag_lookup[UserID]
            diff = current_units - prev_units

            # Use dummy values for Month and Year
            dummy_year = 2025
            dummy_month = 6  # Assume June (summer)

            is_summer = int(dummy_month in [5, 6, 7, 8])
            is_winter = int(dummy_month in [12, 1, 2])

            input_df = pd.DataFrame([{
                'UserID_encoded': encoded_id,
                'Year': dummy_year,
                'Month_Num': dummy_month,
                'is_summer': is_summer,
                'is_winter': is_winter,
                'prev_units': current_units,
                'diff_from_last_month': diff,
                'Cluster': cluster
            }])

            predicted = model.model.predict(input_df)[0]
            st.success(f"✅ Predicted Units: **{round(predicted, 2)}**")

            