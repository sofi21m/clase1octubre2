import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from openai import OpenAI

Expert = " "
profile_imgenh = " "

def encode_image_to_base64(img_bytes):
    return base64.b64encode(img_bytes).decode("utf-8")

# Configuración principal de la página
st.set_page_config(page_title='Tablero Inteligente', layout="centered")
st.title('🎨 Tablero Inteligente: De Boceto a Cuento')

# Barra lateral con herramientas de dibujo
with st.sidebar:
    st.subheader("🛠️ Herramientas de Arte")
    
    # 1. Selección de colores múltiple para niños
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

    # 2. Grosor del pincel
    stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 8)
    
    # 3. Herramienta de dibujo
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
    st.write("Dibuja tu boceto en el panel. La IA interpretará la imagen y creará un cuento infantil inspirado en tu dibujo.")

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
    key="canvas_cuento",
)

# Extracción segura de la imagen del lienzo para evitar fallos al cargar
image_data = getattr(canvas_result, "image_data", None)

# Permite descargar el boceto original
if image_data is not None:
    img_array = np.array(image_data).astype(np.uint8)
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

# Clave API de OpenAI
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
os.environ['OPENAI_API_KEY'] = ke
api_key = os.environ.get('OPENAI_API_KEY')

analyze_button = st.button("✨ Analizar la imagen y crear cuento", type="primary")

# Verificación y generación de cuento
if analyze_button:
    if not api_key:
        st.warning("Por favor ingresa tu API key de OpenAI para continuar.")
    elif image_data is None:
        st.warning("Por favor dibuja algo en el lienzo antes de continuar.")
    else:
        with st.spinner("Analizando tu dibujo y escribiendo una historia..."):
            try:
                # 1. Convertir la imagen a bytes PNG y luego a Base64
                input_numpy_array = np.array(image_data).astype('uint8')
                pil_image = Image.fromarray(input_numpy_array).convert('RGB')
                
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                base64_image = encode_image_to_base64(img_byte_arr.getvalue())

                # 2. Prompt indicándole a GPT que redacte el cuento en español
                prompt_text = "Escribe un cuento corto e ilustrativo para niños basado en este dibujo, en idioma español."

                # 3. Llamada al cliente de OpenAI
                client = OpenAI(api_key=api_key)
                
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
                
                # 4. Mostrar el cuento generado en la pantalla
                cuento = response.choices[0].message.content
                
                st.subheader("📖 Tu Cuento Mágico")
                st.markdown(cuento)

                if Expert == profile_imgenh:
                    st.session_state.mi_respuesta = cuento

            except Exception as e:
                st.error(f"Ocurrió un error al procesar la solicitud: {e}")
