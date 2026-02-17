import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load("rf_model_alzheimer.pkl")

st.title("Application de prédiction du risque d'Alzheimer")
st.write("Projet Master - Outil d'aide à la décision clinique")

# Inputs utilisateur
age = st.number_input("Age", 50, 100, 65)
mmse = st.number_input("Score MMSE", 0, 30, 25)
bmi = st.number_input("BMI", 15.0, 40.0, 25.0)

if st.button("Prédire le risque"):
    
    patient = pd.DataFrame({
        "Age":[age],
        "Gender":[1],
        "Ethnicity":[1],
        "EducationLevel":[2],
        "BMI":[bmi],
        "Smoking":[0],
        "AlcoholConsumption":[1],
        "PhysicalActivity":[2],
        "DietQuality":[2],
        "SleepQuality":[2],
        "FamilyHistoryAlzheimers":[0],
        "CardiovascularDisease":[0],
        "Diabetes":[0],
        "Depression":[0],
        "HeadInjury":[0],
        "Hypertension":[0],
        "SystolicBP":[120],
        "DiastolicBP":[80],
        "CholesterolTotal":[200],
        "CholesterolLDL":[120],
        "CholesterolHDL":[50],
        "CholesterolTriglycerides":[140],
        "MMSE":[mmse]
    })

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    st.subheader("Résultat")

    if probability < 0.4:
        st.success(f"Risque faible ({round(probability,3)})")
    elif probability < 0.7:
        st.warning(f"Risque modéré ({round(probability,3)})")
    else:
        st.error(f"Risque élevé ({round(probability,3)})")

    st.write("Classe prédite (0=Non Alzheimer, 1=Alzheimer) :", prediction)
