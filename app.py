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
st.title('🎨 Tablero Mágico: De Boceto a Imagen')

# Barra lateral para herramientas
with st.sidebar:
    st.subheader("🛠️ Herramientas de Arte")
    
    # Seleccionar colores
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

    stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 8)
    
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

# Descarga del dibujo original
if canvas_result.image_data is not None:
    img_array = np.array(canvas_result.image_data).astype(np.uint8)
    drawing_image = Image.fromarray(img_array)
    
    buffer = io.BytesIO()
    drawing_image.save(buffer, format="PNG")
    bytes_imagen = buffer.getvalue()
    
    st.download_button(
        label="📥 Descargar mi dibujo original (PNG)",
        data=bytes_imagen,
        file_name="mi_boceto.png",
        mime="image/png"
    )

st.markdown("---")

ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
os.environ['OPENAI_API_KEY'] = ke
api_key = os.environ.get('OPENAI_API_KEY')

analyze_button = st.button("✨ Transformar boceto en una Ilustración", type="primary")

if canvas_result.image_data is not None and api_key and analyze_button:
    with st.spinner("Interpretando el boceto y generando la ilustración con DALL-E..."):
        # 1. Guardar la imagen localmente
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('img.png')
        
        base64_image = encode_image_to_base64("img.png")
            
        try:
            client = OpenAI(api_key=api_key)
            
            # Paso 1: Pedir a GPT-4o-mini que describa detalladamente la imagen para DALL-E
            prompt_vision = "Describe en detalle lo que hay en este dibujo infantil para usarlo como prompt de generación de imagen artística para niños en DALL-E. Sé muy claro con los objetos, formas y colores."
            
            vision_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_vision},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            
            descripcion_boceto = vision_response.choices[0].message.content
            
            # Paso 2: Usar DALL-E 3 para crear la nueva imagen con esa descripción
            prompt_dalle = f"A colorful and high quality children's book illustration based on this description: {descripcion_boceto}"
            
            image_response = client.images.generate(
                model="dalle-3",
                prompt=prompt_dalle,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            # Obtener la URL de la imagen generada
            url_imagen_generada = image_response.data[0].url
            
            # Mostrar la imagen en Streamlit
            st.subheader("🖼️ ¡Mira tu dibujo convertido en arte!")
            st.image(url_imagen_generada, caption="Ilustración generada a partir de tu boceto")

        except Exception as e:
            st.error(f"Ocurrió un error al procesar la solicitud: {e}")

else:
    if analyze_button and not api_key:
        st.warning("Por favor ingresa tu API key de OpenAI para poder continuar.")
