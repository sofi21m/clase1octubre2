import os
import io
import urllib.parse
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from google import genai
from google.genai import types

# Configuración principal de la página
st.set_page_config(page_title='Tablero Mágico con Gemini', layout="centered")
st.title('🎨 Tablero Mágico con Google Gemini')

# Barra lateral con herramientas de dibujo
with st.sidebar:
    st.subheader("🛠️ Herramientas de Arte")
    
    # Selección de colores
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
    stroke_color = st.color_picker("O personaliza el color aquí:", value=color_base)

    # Grosor del pincel
    stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 8)
    
    # Herramienta de dibujo
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
    st.write("Dibuja tu boceto en el panel. **Gemini** interpretará la imagen y un generador la transformará en una ilustración mágica.")

st.subheader("👇 Dibuja tu boceto en el panel")

# Creación del lienzo interactivo
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=350,
    width=500,
    drawing_mode=drawing_mode,
    key="canvas_gemini",
)

# Permite descargar el boceto original
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

# Clave API de Gemini
gemini_api_key = st.text_input('Ingresa tu API Key de Google Gemini', type="password")

analyze_button = st.button("✨ Transformar boceto con Gemini", type="primary")

if canvas_result.image_data is not None and gemini_api_key and analyze_button:
    with st.spinner("Gemini está analizando tu dibujo y creando la ilustración..."):
        try:
            # 1. Convertir la imagen del lienzo a bytes PNG
            input_numpy_array = np.array(canvas_result.image_data).astype('uint8')
            pil_image = Image.fromarray(input_numpy_array).convert('RGB')
            
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            # 2. Inicializar el cliente de Gemini
            client = genai.Client(api_key=gemini_api_key)

            # 3. Preparar los datos compatibles para Gemini
            image_part = types.Part.from_bytes(
                data=img_bytes,
                mime_type='image/png'
            )

            prompt_vision = (
                "Describe en inglés de forma corta y muy detallada los objetos, personajes, colores y "
                "escenario de este dibujo infantil para crear un prompt de imagen en alta calidad estilo libro ilustrado para niños."
            )

            # 4. Enviar a Gemini 2.5 Flash
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image_part, prompt_vision]
            )

            descripcion_boceto = response.text
            st.success("¡Gemini entendió tu dibujo!")

            # 5. Generar la ilustración con Pollinations.ai usando la descripción de Gemini
            prompt_final = f"A beautiful children's book illustration, vibrant colors, fantasy style: {descripcion_boceto}"
            prompt_encoded = urllib.parse.quote(prompt_final)
            url_imagen = f"https://image.pollinations.ai/prompt/{prompt_encoded}?width=1024&height=1024&nologo=true"

            # Mostrar resultado final
            st.subheader("🖼️ ¡Mira tu dibujo convertido en arte!")
            st.image(url_imagen, caption="Ilustración generada a partir de la interpretación de Gemini")

        except Exception as e:
            st.error(f"Ocurrió un error al procesar con Gemini: {e}")

else:
    if analyze_button and not gemini_api_key:
        st.warning("Por favor ingresa tu API key de Google Gemini para continuar.")
