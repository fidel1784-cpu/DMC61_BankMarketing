import streamlit as st

st.set_page_config(
    page_title="Bank Marketing",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Caso de Estudio: Bank Marketing")
st.subheader("Análisis de campañas de marketing bancario")

st.write(
    """
    Esta aplicación tiene como objetivo analizar la información de una campaña
    de marketing realizada por una entidad bancaria.
    """
)

st.divider()

st.info("Seleccione una opción en el menú lateral para comenzar.")

st.sidebar.title("Menú principal")

opcion = st.sidebar.selectbox(
    "Seleccione una opción:",
    [
        "Inicio",
        "Carga de datos",
        "Análisis exploratorio",
        "Visualizaciones",
        "Conclusiones"
    ]
)

st.sidebar.divider()
st.sidebar.write("Elaborado por: Fidel Bringas")
