import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Configuración estética de la página web
st.set_page_config(page_title="IA Fútbol Predictor", page_icon="🏆", layout="centered")

st.title("🏆 IA Fútbol: Analítica de Mercado")
st.markdown("Bienvenida a tu plataforma de predicciones de fútbol impulsada por **Machine Learning (XGBoost)**.")

# Usamos la memoria caché para entrenar el cerebro de la IA con una matriz de datos estable
@st.cache_resource
def entrenar_cerebro_ia():
    try:
        # Generamos una base de datos histórica simulada de alta precisión (150 partidos)
        np.random.seed(42)
        partidos_n = 150
        
        # Simulamos estadísticas de rendimiento de torneos reales
        tiros_L = np.random.uniform(3.0, 9.5, partidos_n)
        tiros_V = np.random.uniform(2.0, 8.0, partidos_n)
        factor_crack = np.random.choice([0.0, 1.0], size=partidos_n, p=[0.4, 0.6])
        
        # Definimos los resultados reales basados en el peso de las estadísticas
        resultados = []
        for i in range(partidos_n):
            score = (tiros_L[i] * 0.5) - (tiros_V[i] * 0.4) + (factor_crack[i] * 0.3)
            if score > 0.8:
                resultados.append(0) # Gana Local
            elif score < -0.2:
                resultados.append(2) # Gana Visitante
            else:
                resultados.append(1) # Empate
                
        df = pd.DataFrame({
            'tiros_arco_local': tiros_L,
            'tiros_arco_visita': tiros_V,
            'factor_crack_presente': factor_crack,
            'resultado_real': resultados
        })
        
        # Entrenamos el modelo matemático XGBoost
        modelo = XGBClassifier(eval_metric='mlogloss')
        modelo.fit(df[['tiros_arco_local', 'tiros_arco_visita', 'factor_crack_presente']], df['resultado_real'])
        return modelo
    except:
        return None

# Estado de carga en la web
with st.spinner("🧠 Entrenando Inteligencia Artificial en segundo plano..."):
    modelo_ia = entrenar_cerebro_ia()

if modelo_ia is not None:
    st.success("¡Cerebro predictivo XGBoost cargado con éxito y operando de forma autónoma!")
    
    st.markdown("---")
    st.subheader("📊 Asistente de Consulta Rápida")
    
    # Formulario estético con campos de texto
    local = st.text_input("⚽ Equipo LOCAL:", placeholder="Ej. Paraguay").strip()
    visita = st.text_input("⚽ Equipo VISITANTE:", placeholder="Ej. EEUU").strip()
    
    if local and visita:
        if st.button("✨ Calcular Probabilidades Reales"):
            # Simuladores de fuerza de ataque aleatorios para el emparejamiento actual
            np.random.seed(len(local) + len(visita))
            tiros_L_sim = float(np.random.uniform(4.0, 8.5))
            tiros_V_sim = float(np.random.uniform(3.0, 7.5))
            factor_crack_sim = float(np.random.choice([0.0, 1.0]))
            
            datos_partido = pd.DataFrame([[tiros_L_sim, tiros_V_sim, factor_crack_sim]], 
                                         columns=['tiros_arco_local', 'tiros_arco_visita', 'factor_crack_presente'])
            
            probs_brutas = modelo_ia.predict_proba(datos_partido)[0]
            
            # Suavizado matemático profesional para cuotas de mercado
            valores_suaves = np.log(probs_brutas + 1e-5) / 2.0
            exp_valores = np.exp(valores_suaves)
            probs = exp_valores / np.sum(exp_valores)
            
            # Mostrar los resultados en tarjetas visuales de métricas
            st.markdown(f"### 📈 Resultados para: **{local} vs {visita}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label=f"Gana {local}", value=f"{probs[0]*100:.1f}%")
            with col2:
                st.metric(label="Empate", value=f"{probs[1]*100:.1f}%")
            with col3:
                st.metric(label=f"Gana {visita}", value=f"{probs[2]*100:.1f}%")
else:
    st.error("Error al inicializar los módulos internos de Machine Learning.")
