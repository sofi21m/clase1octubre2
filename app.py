import sys
import io
import base64

import numpy as np
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from openai import OpenAI


# =========================================================
# CONFIGURACIÓN UTF-8
# =========================================================

if sys.version_info[0] >= 3:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Tablero Inteligente",
    page_icon="🎨",
    layout="centered"
)


# =========================================================
# TÍTULO
# =========================================================

st.title("🎨 Tablero Mágico: De Boceto a Imagen")

st.write(
    "Dibuja un boceto y la inteligencia artificial "
    "lo convertirá en una ilustración."
)


# =========================================================
# BARRA LATERAL
# =========================================================

with st.sidebar:

    st.subheader("🛠️ Herramientas de Arte")

    # -----------------------------------------------------
    # COLORES
    # -----------------------------------------------------

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
        "⚪ Blanco": "#FFFFFF"
    }

    opcion_color = st.radio(
        "Paleta rápida:",
        list(colores_divertidos.keys())
    )

    color_base = colores_divertidos[opcion_color]

    stroke_color = st.color_picker(
        "O personaliza el color aquí:",
        value=color_base
    )


    # -----------------------------------------------------
    # GROSOR
    # -----------------------------------------------------

    stroke_width = st.slider(
        "Selecciona el ancho de línea",
        1,
        30,
        8
    )


    # -----------------------------------------------------
    # HERRAMIENTA
    # -----------------------------------------------------

    drawing_mode = st.selectbox(
        "Herramienta:",

        (
            "freedraw",
            "line",
            "rect",
            "circle"
        ),

        format_func=lambda x: {

            "freedraw": "✏️ Lápiz libre",

            "line": "📏 Línea recta",

            "rect": "⬛ Rectángulo",

            "circle": "⚪ Círculo"

        }[x]
    )


    # -----------------------------------------------------
    # FONDO
    # -----------------------------------------------------

    bg_color = st.color_picker(
        "Color de fondo del papel",
        "#FFFFFF"
    )


    st.markdown("---")


    # -----------------------------------------------------
    # INFORMACIÓN
    # -----------------------------------------------------

    st.subheader("Acerca de:")

    st.write(
        "Dibuja tu boceto en el panel. "
        "La IA interpretará la imagen y "
        "la transformará en una ilustración."
    )


# =========================================================
# ÁREA DE DIBUJO
# =========================================================

st.subheader("👇 Dibuja tu boceto en el panel")


canvas_result = st_canvas(

    fill_color="rgba(255, 165, 0, 0.3)",

    stroke_width=stroke_width,

    stroke_color=stroke_color,

    background_color=bg_color,

    height=350,

    width=500,

    drawing_mode=drawing_mode,

    display_toolbar=True,

    key="canvas_openai"
)


# =========================================================
# OBTENER IMAGEN DEL CANVAS
# =========================================================

image_data = getattr(
    canvas_result,
    "image_data",
    None
)


# =========================================================
# DESCARGAR BOCETO
# =========================================================

if image_data is not None:

    img_array = np.array(
        image_data
    ).astype(np.uint8)


    drawing_image = Image.fromarray(
        img_array
    )


    buffer = io.BytesIO()


    drawing_image.save(
        buffer,
        format="PNG"
    )


    bytes_imagen = buffer.getvalue()


    st.download_button(

        label="📥 Descargar mi dibujo original (PNG)",

        data=bytes_imagen,

        file_name="mi_boceto.png",

        mime="image/png"
    )


# =========================================================
# SEPARADOR
# =========================================================

st.markdown("---")


# =========================================================
# API KEY
# =========================================================

api_key_input = st.text_input(
    "Ingresa tu Clave de OpenAI",
    type="password",
    placeholder="sk-..."
)


# =========================================================
# BOTÓN
# =========================================================

analyze_button = st.button(
    "✨ Transformar boceto en una Ilustración",
    type="primary"
)


# =========================================================
# PROCESAMIENTO
# =========================================================

if analyze_button:

    clean_api_key = (
        api_key_input.strip()
        if api_key_input
        else ""
    )


    # -----------------------------------------------------
    # COMPROBAR API KEY
    # -----------------------------------------------------

    if not clean_api_key:

        st.warning(
            "Por favor ingresa tu API key de OpenAI para continuar."
        )


    # -----------------------------------------------------
    # COMPROBAR DIBUJO
    # -----------------------------------------------------

    elif image_data is None:

        st.warning(
            "Por favor dibuja algo en el lienzo antes de continuar."
        )


    else:

        try:

            # =================================================
            # CLIENTE OPENAI
            # =================================================

            client = OpenAI(
                api_key=clean_api_key
            )


            # =================================================
            # CONVERTIR DIBUJO A PNG
            # =================================================

            input_numpy_array = np.array(
                image_data
            ).astype("uint8")


            pil_image = Image.fromarray(
                input_numpy_array
            ).convert("RGB")


            img_byte_arr = io.BytesIO()


            pil_image.save(
                img_byte_arr,
                format="PNG"
            )


            image_bytes = img_byte_arr.getvalue()


            # =================================================
            # BASE64
            # =================================================

            base64_image = base64.b64encode(
                image_bytes
            ).decode("utf-8")


            # =================================================
            # PASO 1
            # LA IA INTERPRETA EL BOCETO
            # =================================================

            with st.spinner(
                "🔎 Interpretando tu boceto..."
            ):

                prompt_vision = """
Analiza el dibujo realizado por el usuario.

Describe claramente qué objeto, personaje,
animal, lugar o escena representa.

Ten en cuenta que es un dibujo hecho a mano
y puede ser simple o tener pocos detalles.

Describe:

1. Objeto principal.
2. Forma y características.
3. Elementos secundarios.
4. Colores visibles.
5. Posición de los elementos.
6. Posible contexto de la escena.

La descripción será utilizada posteriormente
para crear una ilustración de libro infantil.
"""


                vision_response = client.chat.completions.create(

                    model="gpt-4o-mini",

                    messages=[

                        {
                            "role": "user",

                            "content": [

                                {
                                    "type": "text",

                                    "text": prompt_vision
                                },

                                {
                                    "type": "image_url",

                                    "image_url": {

                                        "url":
                                        f"data:image/png;base64,{base64_image}"

                                    }
                                }

                            ]
                        }

                    ],

                    max_tokens=400
                )


                descripcion_boceto = (
                    vision_response
                    .choices[0]
                    .message
                    .content
                )


            # =================================================
            # MOSTRAR INTERPRETACIÓN
            # =================================================

            st.subheader(
                "🔎 Interpretación del boceto"
            )

            st.write(
                descripcion_boceto
            )


            # =================================================
            # PASO 2
            # CREAR PROMPT PARA LA IMAGEN
            # =================================================

            prompt_imagen = f"""
Create a vibrant and charming children's
book illustration based on the following
description:

{descripcion_boceto}

Style:
- Children's picture book
- Colorful
- Friendly
- Whimsical
- Clean shapes
- Soft lighting
- Detailed but easy to understand
- Professional illustration
- Attractive composition
- Keep the main subject clearly recognizable

Transform the simple hand-drawn concept
into a polished illustrated scene.
"""


            # =================================================
            # PASO 3
            # GENERAR IMAGEN
            # =================================================

            with st.spinner(
                "🎨 Creando tu ilustración..."
            ):

                image_response = client.images.generate(

                    model="gpt-image-2",

                    prompt=prompt_imagen,

                    size="1024x1024",

                    quality="medium",

                    n=1
                )


            # =================================================
            # OBTENER IMAGEN EN BASE64
            # =================================================

            image_base64 = (
                image_response
                .data[0]
                .b64_json
            )


            image_bytes_generated = base64.b64decode(
                image_base64
            )


            # =================================================
            # MOSTRAR RESULTADO
            # =================================================

            st.subheader(
                "🖼️ ¡Mira tu dibujo convertido en arte!"
            )


            st.image(
                image_bytes_generated,

                caption="Ilustración generada con IA",

                use_container_width=True
            )


            # =================================================
            # BOTÓN DESCARGAR ILUSTRACIÓN
            # =================================================

            st.download_button(

                label="📥 Descargar ilustración",

                data=image_bytes_generated,

                file_name="ilustracion_boceto.png",

                mime="image/png"
            )


        # =====================================================
        # ERRORES
        # =====================================================

        except Exception as e:

            st.error(
                "Ocurrió un error al procesar la solicitud."
            )

            st.code(
                str(e)
            )
