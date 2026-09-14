import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO


# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing",
    page_icon="🏦",
    layout="wide"
)


# ---------------------------------------------------------
# CLASE PARA TRABAJAR CON EL DATASET
# ---------------------------------------------------------
class AnalizadorBankMarketing:

    def __init__(self, dataframe):
        self.df = dataframe

    def obtener_dimensiones(self):
        filas, columnas = self.df.shape
        return filas, columnas

    def obtener_vista_previa(self, cantidad=5):
        return self.df.head(cantidad)

    def obtener_info(self):
        buffer = StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()


# ---------------------------------------------------------
# FUNCIÓN PARA CARGAR EL ARCHIVO CSV
# ---------------------------------------------------------
def cargar_dataset(archivo):
    try:
        # Detecta automáticamente si el separador es coma o punto y coma
        dataframe = pd.read_csv(archivo, sep=None, engine="python")
        return dataframe, None
    except Exception as error:
        return None, str(error)


# ---------------------------------------------------------
# MENÚ LATERAL
# ---------------------------------------------------------
st.sidebar.title("🏦 Menú principal")

opcion = st.sidebar.selectbox(
    "Seleccione un módulo:",
    [
        "Home",
        "Carga del dataset",
        "Análisis EDA"
    ]
)

st.sidebar.divider()
st.sidebar.write("**Autor:** Fidel Napoleón Bringas Salazar")
st.sidebar.write("**Año:** 2026")


# ---------------------------------------------------------
# MÓDULO 1: HOME
# ---------------------------------------------------------
if opcion == "Home":

    st.title("🏦 Bank Marketing: Análisis Exploratorio de Datos")

    st.subheader("Presentación del proyecto")

    st.write(
        """
        Este proyecto presenta una aplicación interactiva desarrollada en
        Python y Streamlit para realizar el Análisis Exploratorio de Datos
        del dataset **BankMarketing.csv**.
        """
    )

    st.info(
        """
        **Objetivo del análisis:** explorar las características de los clientes
        y los resultados de una campaña de marketing bancario, identificando
        distribuciones, relaciones entre variables, valores faltantes y
        hallazgos relevantes mediante estadística descriptiva y visualizaciones.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Datos del autor")
        st.write("**Nombre:** Fidel Napoleón Bringas Salazar")
        st.write("**Curso / Especialización:** Python for Analytics")
        st.write("**Año:** 2026")

    with col2:
        st.subheader("🛠️ Tecnologías utilizadas")
        st.write("- Python")
        st.write("- Streamlit")
        st.write("- Pandas y NumPy")
        st.write("- Matplotlib y Seaborn")
        st.write("- Programación Orientada a Objetos")

    st.subheader("📊 Descripción del dataset")

    st.write(
        """
        Bank Marketing contiene información relacionada con campañas de
        marketing directo realizadas por una institución bancaria. Incluye
        características de los clientes, datos de contacto y el resultado
        de la campaña, representado principalmente por la variable `y`.
        """
    )

    st.warning(
        "Este proyecto realiza análisis exploratorio y no construye modelos predictivos."
    )


# ---------------------------------------------------------
# MÓDULO 2: CARGA DEL DATASET
# ---------------------------------------------------------
elif opcion == "Carga del dataset":

    st.title("📂 Carga del dataset")

    st.write(
        """
        Cargue el archivo **BankMarketing.csv** para visualizar sus datos
        y habilitar el módulo de análisis exploratorio.
        """
    )

    archivo = st.file_uploader(
        "Seleccione el archivo CSV",
        type=["csv"]
    )

    if archivo is None:
        st.warning("Debe cargar el archivo BankMarketing.csv para continuar.")

    else:
        df, error = cargar_dataset(archivo)

        if error is not None:
            st.error(f"No se pudo cargar el archivo. Detalle: {error}")

        else:
            st.session_state["dataset"] = df
            analizador = AnalizadorBankMarketing(df)
            filas, columnas = analizador.obtener_dimensiones()

            st.success("El dataset fue cargado correctamente.")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Número de filas", f"{filas:,}")

            with col2:
                st.metric("Número de columnas", columnas)

            st.subheader("Vista previa del dataset")
            st.dataframe(
                analizador.obtener_vista_previa(),
                use_container_width=True
            )

            st.subheader("Nombres de las columnas")
            st.write(list(df.columns))


# ---------------------------------------------------------
# MÓDULO 3: ANÁLISIS EDA
# ---------------------------------------------------------
elif opcion == "Análisis EDA":

    st.title("📈 Análisis Exploratorio de Datos")

    if "dataset" not in st.session_state:
        st.warning(
            "Primero debe ingresar al módulo 'Carga del dataset' y cargar el archivo CSV."
        )
        st.stop()

    df = st.session_state["dataset"]

    st.success(
        f"Dataset disponible: {df.shape[0]:,} filas y {df.shape[1]} columnas."
    )

    # Función personalizada para clasificar las variables
    def clasificar_variables(dataframe):
        variables_numericas = dataframe.select_dtypes(
            include=np.number
        ).columns.tolist()

        variables_categoricas = dataframe.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        return variables_numericas, variables_categoricas

    numericas, categoricas = clasificar_variables(df)

    tab1, tab2 = st.tabs([
        "1️⃣ Información general",
        "2️⃣ Clasificación de variables"
    ])

    # -----------------------------------------------------
    # ÍTEM 1: INFORMACIÓN GENERAL DEL DATASET
    # -----------------------------------------------------
    with tab1:

        st.header("Ítem 1: Información general del dataset")

        st.write(
            """
            Este análisis permite conocer la estructura del dataset,
            los tipos de datos de sus variables y la cantidad de valores
            nulos presentes.
            """
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Filas", f"{df.shape[0]:,}")

        with col2:
            st.metric("Columnas", df.shape[1])

        with col3:
            st.metric("Valores nulos", int(df.isnull().sum().sum()))

        st.subheader("Información obtenida con .info()")

        buffer = StringIO()
        df.info(buf=buffer)
        informacion = buffer.getvalue()

        st.code(informacion, language="text")

        st.subheader("Tipos de datos")

        tabla_tipos = pd.DataFrame({
            "Variable": df.columns,
            "Tipo de dato": df.dtypes.astype(str).values
        })

        st.dataframe(
            tabla_tipos,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Conteo de valores nulos")

        tabla_nulos = pd.DataFrame({
            "Variable": df.columns,
            "Valores nulos": df.isnull().sum().values
        })

        tabla_nulos["Porcentaje"] = (
            tabla_nulos["Valores nulos"] / len(df) * 100
        ).round(2)

        st.dataframe(
            tabla_nulos,
            use_container_width=True,
            hide_index=True
        )

        if df.isnull().sum().sum() == 0:
            st.success(
                "No se identificaron valores nulos en el dataset."
            )
        else:
            st.warning(
                "El dataset contiene valores nulos que deberán analizarse."
            )

    # -----------------------------------------------------
    # ÍTEM 2: CLASIFICACIÓN DE VARIABLES
    # -----------------------------------------------------
    with tab2:

        st.header("Ítem 2: Clasificación de variables")

        st.write(
            """
            Mediante una función personalizada se clasifican las variables
            del dataset en numéricas y categóricas, según el tipo de dato
            reconocido por Pandas.
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Variables numéricas",
                len(numericas)
            )

            st.subheader("Lista de variables numéricas")

            tabla_numericas = pd.DataFrame({
                "N.º": range(1, len(numericas) + 1),
                "Variable numérica": numericas
            })

            st.dataframe(
                tabla_numericas,
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.metric(
                "Variables categóricas",
                len(categoricas)
            )

            st.subheader("Lista de variables categóricas")

            tabla_categoricas = pd.DataFrame({
                "N.º": range(1, len(categoricas) + 1),
                "Variable categórica": categoricas
            })

            st.dataframe(
                tabla_categoricas,
                use_container_width=True,
                hide_index=True
            )

        st.subheader("Resumen de la clasificación")

        resumen_variables = pd.DataFrame({
            "Clasificación": [
                "Variables numéricas",
                "Variables categóricas"
            ],
            "Cantidad": [
                len(numericas),
                len(categoricas)
            ]
        })

        st.bar_chart(
            resumen_variables.set_index("Clasificación")
        )

        st.info(
            f"El dataset contiene {len(numericas)} variables numéricas "
            f"y {len(categoricas)} variables categóricas."
        )
