import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
import os
import pickle

# Definisi FeatureSelector
class FeatureSelector:
    def __init__(self, selected_features):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.selected_features]

# Fungsi untuk memastikan semua kolom yang diperlukan tersedia
def prepare_data(data, required_columns):
    """
    Menyiapkan data dengan menambahkan kolom yang hilang dengan nilai default (0)
    dan menghapus kolom yang tidak relevan.
    """
    # Tambahkan kolom yang hilang
    missing_cols = set(required_columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0

    # Hapus kolom yang tidak relevan
    extra_cols = set(data.columns) - set(required_columns)
    data = data.drop(columns=list(extra_cols), errors='ignore')

    # Pastikan urutan kolom sesuai dengan required_columns
    data = data.reindex(columns=required_columns, fill_value=0)

    # Konversi semua kolom ke tipe numerik (jika memungkinkan)
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    return data

# Fungsi untuk load model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'LogisticRegressionModel3.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            if not hasattr(model, 'predict'):
                st.error("File pickle tidak berisi model yang valid. Pastikan file yang diunggah adalah model yang sesuai.")
                return None
            return model
    except FileNotFoundError:
        st.warning("File 'LogisticRegressionModel3.pkl' tidak ditemukan. Silakan unggah model terlebih dahulu.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Fungsi untuk mendapatkan fitur yang diperlukan oleh model
def get_model_features(model):
    """
    Mengambil nama-nama fitur yang diharapkan oleh model.
    """
    try:
        return model.feature_names_in_
    except AttributeError:
        try:
            return model.named_steps['preprocessor'].get_feature_names_out()
        except (AttributeError, KeyError):
            st.error("Tidak dapat mengekstrak fitur dari model. Pastikan model mendukung atribut fitur.")
            return []

# Load model pipeline
model_tuned = load_model()

if model_tuned is not None:
    important_features = get_model_features(model_tuned)

# Judul Aplikasi
st.title("Customer Churn Prediction")
st.markdown("Masukkan data pelanggan secara manual atau unggah dataset untuk memulai prediksi churn.")

# Input Data
st.header("Input Data Pelanggan")
input_mode = st.radio("Pilih metode input:", ("Upload Dataset", "Input Data Manual"))

if input_mode == "Upload Dataset":
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Data yang diunggah:")
        st.write(data.head())

        if st.checkbox("Lakukan prediksi churn dengan model yang dimuat"):
            if 'Churn' in data.columns:
                X = pd.get_dummies(data.drop(columns=['Churn']), drop_first=True)

                # Persiapkan data dengan memastikan kolom yang diperlukan tersedia
                if model_tuned is not None:
                    X = prepare_data(X, important_features)

                    try:
                        y = data['Churn']
                        y_pred = model_tuned.predict(X)
                        st.write(f"Recall Score pada dataset: {recall_score(y, y_pred):.2f}")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat prediksi: {e}")
                else:
                    st.error("Model belum dimuat. Pastikan Anda mengunggah atau memiliki model lokal.")
            else:
                st.error("Kolom 'Churn' tidak ditemukan dalam dataset. Pastikan dataset memiliki kolom target bernama 'Churn'.")

else:
    st.subheader("Input Data Manual")

    st.write("Masukkan data pelanggan secara manual berdasarkan fitur penting:")
    inputs = {
        'gender': st.selectbox("Gender", ["Male", "Female"]),
        'SeniorCitizen': st.checkbox("Senior Citizen"),
        'Partner': st.checkbox("Partner"),
        'Dependents': st.checkbox("Dependents"),
        'tenure': st.number_input("Tenure (bulan)", min_value=0, value=12),
        'PhoneService': st.checkbox("Phone Service", key="PhoneService"),
        'MultipleLines': st.selectbox("Multiple Lines", ["No", "Yes"], disabled=not st.session_state.get("PhoneService", True)),
        'InternetService': st.selectbox("Layanan Internet", ["DSL", "Fiber optic", "No"], key="InternetService"),
        'OnlineSecurity': st.checkbox("Online Security", disabled=st.session_state.get("InternetService") == "No"),
        'OnlineBackup': st.checkbox("Online Backup", disabled=st.session_state.get("InternetService") == "No"),
        'DeviceProtection': st.checkbox("Device Protection", disabled=st.session_state.get("InternetService") == "No"),
        'TechSupport': st.checkbox("Tech Support", disabled=st.session_state.get("InternetService") == "No"),
        'StreamingTV': st.checkbox("Streaming TV", disabled=st.session_state.get("InternetService") == "No"),
        'StreamingMovies': st.checkbox("Streaming Movies", disabled=st.session_state.get("InternetService") == "No"),
        'Contract': st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"]),
        'PaperlessBilling': st.checkbox("Paperless Billing"),
        'PaymentMethod': st.selectbox("Metode Pembayaran", ["Credit card (automatic)", "Electronic check", "Mailed check", "Bank transfer (automatic)"])
    }

    # Konversi input menjadi format sesuai dengan model
    input_data = pd.DataFrame({
        'gender': [inputs['gender']],
        'SeniorCitizen': [1 if inputs['SeniorCitizen'] else 0],
        'Partner': [1 if inputs['Partner'] else 0],
        'Dependents': [1 if inputs['Dependents'] else 0],
        'tenure': [inputs['tenure']],
        'PhoneService': [1 if inputs['PhoneService'] else 0],
        'MultipleLines': [1 if inputs['MultipleLines'] == "Yes" else 0],
        'InternetService': [inputs['InternetService']],
        'OnlineSecurity': [1 if inputs['OnlineSecurity'] else 0],
        'OnlineBackup': [1 if inputs['OnlineBackup'] else 0],
        'DeviceProtection': [1 if inputs['DeviceProtection'] else 0],
        'TechSupport': [1 if inputs['TechSupport'] else 0],
        'StreamingTV': [1 if inputs['StreamingTV'] else 0],
        'StreamingMovies': [1 if inputs['StreamingMovies'] else 0],
        'Contract': [inputs['Contract']],
        'PaperlessBilling': [1 if inputs['PaperlessBilling'] else 0],
        'PaymentMethod': [inputs['PaymentMethod']]
    })

    # Pastikan semua kolom penting ada di input_data
    if model_tuned is not None:
        input_data = prepare_data(pd.get_dummies(input_data), important_features)

    st.write("### Data yang Anda masukkan:")
    st.write(input_data)

    if st.button("Prediksi Churn"):
        if model_tuned is not None:
            try:
                prediction = model_tuned.predict(input_data)
                st.write("### Prediksi Churn:")
                st.write("Churn" if prediction[0] == 1 else "Tidak Churn")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
        else:
            st.error("Model belum dimuat. Pastikan Anda mengunggah atau memiliki model lokal.")
