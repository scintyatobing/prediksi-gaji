# KELOMPOK 01
# 11320005 - Scintya Lumban Tobing
# 11320043 - Rut Lumbantoruan
# 11320057 - Feronika Simanjuntak

import streamlit as st
import numpy as np
import pickle

def load_model():
    with open('regresi-prediksi-gaji.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
city_label = data["city_label"]
role_label = data["role_label"]

def tampil_prediksi_gaji():
    st.title("Prediksi Gaji Karyawan IT di pulau Jawa")

    city = (
        "Jakarta",
        "Yogyakarta",
        "Surabaya",
        "Bandung",       
        "Tangerang",
        "Banten",
        "Semarang",
        "Bekasi",
        "Depok",
        "Bogor",
    )

    role = (
        "Software Engineer",
        "Project Manager",
        "Backend Engineer",
        "Android Developer",
        "Data Analyst",
        "Frontend Engineer",
        "UX Writer",
        "Quality Assurance",
        "UI Designer",
        "IT Support",
        "Data Engineer",
        "Fullstack Developer",
        "Tester",
        "Data Scientist",
    )

    city = st.selectbox("City", city)
    role = st.selectbox("Role", role)
    years_experience = st.slider('Tahun pengalaman', 0, 20, 0)

    # expericence = st.slider("Years' Experience", 0, 50, 3)

    ok = st.button("Hitung Gaji")
    if ok:
        X = np.array([[city, role, years_experience ]])
        X[:, 0] = city_label.transform(X[:,0])
        X[:, 1] = role_label.transform(X[:,1])
        X = X.astype(float)

        gaji = regressor.predict(X)
        st.subheader(f"Prediksi gaji adalah Rp{gaji[0]:.0f}/bulan")