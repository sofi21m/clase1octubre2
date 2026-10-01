import os
import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from openai import OpenAI
import openai

Expert = " "
profile_imgenh = " "

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_image
    except FileNotFoundError:
        return "Error: La imagen no se encontró en la ruta especificada."

# Configuración de la página
st.set_page_config(page_title='Tablero Inteligente', layout="centered")
st.title('🎨 Tablero Mágico e Inteligente')

# Barra lateral para herramientas y configuración
with st.sidebar:
    st.subheader("🛠️ Herramientas de Arte")
    
    # 1. Selección de colores múltiple para el niño
    st.subheader("🌈 Elige un color para pintar")
    colores_divertidos = {
        "🔴 Rojo": "#FF0000",
        "🟡 Amarillo": "#FFD700",
        "🟢 Verde": "#00FF00",
        "🔵 Azul": "#0000FF",
        "🟣 Morado": "#8A2BE2",
        "💖 Rosado": "#FF69B4",
        "🟧 Naranja": "#FFA500",
        "🖤 Negro": "#000000",
        "⚪ Blanco": "#FFFFFF",
    }
    
    opcion_color = st.radio("Paleta rápida:", list(colores_divertidos.keys()))
    color_base = colores_divertidos[opcion_color]
    stroke_color = st.color_picker("O personaliza el color:", value=color_base)

    # 2. Tamaño del pincel
    stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 8)
    
    # 3. Herramientas de dibujo
    drawing_mode = st.selectbox(
        "Herramienta:",
        ("freedraw", "line", "rect", "circle"),
        format_func=lambda x: {
            "freedraw": "✏️ Lápiz libre",
            "line": "📏 Línea recta",
            "rect": "⬛ Rectángulo",
            "circle": "⚪ Círculo"
        }[x]
    )
    
    bg_color = st.color_picker("Color de fondo del papel", "#FFFFFF")

    st.markdown("---")
    st.subheader("Acerca de:")
    st.write("Dibuja en el lienzo usando los colores que quieras. La IA interpretará tu boceto y creará un cuento.")

st.subheader("👇 Dibuja tu boceto en el panel")

# Creación del lienzo
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=350,
    width=500,
    drawing_mode=drawing_mode,
    key="canvas_inteligente",
)

# Sección para descargar la imagen dibujada
if canvas_result.image_data is not None:
    # Convertir el arreglo del lienzo a imagen PNG
    img_array = np.array(canvas_result.image_data).astype(np.uint8)
    drawing_image = Image.fromarray(img_array)
    
    # Guardar en memoria para descarga
    buffer = io.BytesIO()
    drawing_image.save(buffer, format="PNG")
    bytes_imagen = buffer.getvalue()
    
    st.download_button(
        label="📥 Descargar mi dibujo (PNG)",
        data=bytes_imagen,
        file_name="mi_dibujo_magico.png",
        mime="image/png"
    )

st.markdown("---")

# Clave API y Botón de análisis
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
os.environ['OPENAI_API_KEY'] = ke
api_key = os.environ.get('OPENAI_API_KEY')

analyze_button = st.button("✨ Analizar la imagen y crear cuento", type="primary")

# Proceso de análisis e integración con OpenAI
if canvas_result.image_data is not None and api_key and analyze_button:
    with st.spinner("Analizando tu dibujo y escribiendo una historia..."):
        # Guardar la imagen localmente para el procesamiento
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('img.png')
        
        # Codificar la imagen a base64
        base64_image = encode_image_to_base64("img.png")
            
        prompt_text = "write a history for children based in the image in spanish"
    
        try:
            client = OpenAI(api_key=api_key)
            full_response = ""
            message_placeholder = st.empty()
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )
            
            if response.choices[0].message.content is not None:
                full_response = response.choices[0].message.content
                message_placeholder.markdown(full_response)
                
            if Expert == profile_imgenh:
                st.session_state.mi_respuesta = full_response

        except Exception as e:
            st.error(f"Ocurrió un error al procesar la solicitud: {e}")

else:
    if analyze_button and not api_key:
        st.warning("Por favor ingresa tu API key de OpenAI para poder continuar.")
