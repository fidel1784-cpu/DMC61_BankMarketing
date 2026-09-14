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

         tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1️⃣ Información general",
        "2️⃣ Clasificación",
        "3️⃣ Estadísticas",
        "4️⃣ Valores faltantes",
        "5️⃣ Distribución numérica",
        "6️⃣ Variables categóricas"
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

    # -----------------------------------------------------
    # ÍTEM 3: ESTADÍSTICAS DESCRIPTIVAS
    # -----------------------------------------------------
    with tab3:

        st.header("Ítem 3: Estadísticas descriptivas")

        st.write(
            """
            Las estadísticas descriptivas permiten resumir el comportamiento
            de las variables numéricas mediante medidas de tendencia central
            y dispersión.
            """
        )

        estadisticas = df[numericas].describe().T

        estadisticas["median"] = df[numericas].median()
        estadisticas["variance"] = df[numericas].var()

        estadisticas = estadisticas.rename(columns={
            "count": "Cantidad",
            "mean": "Media",
            "std": "Desviación estándar",
            "min": "Mínimo",
            "25%": "Percentil 25",
            "50%": "Percentil 50",
            "75%": "Percentil 75",
            "max": "Máximo",
            "median": "Mediana",
            "variance": "Varianza"
        })

        st.subheader("Resumen estadístico de variables numéricas")

        st.dataframe(
            estadisticas.round(2),
            use_container_width=True
        )

        st.subheader("Interpretación de medias, medianas y dispersión")

        variable_interpretar = st.selectbox(
            "Seleccione una variable numérica:",
            numericas,
            key="variable_estadisticas"
        )

        media = np.mean(df[variable_interpretar])
        mediana = np.median(df[variable_interpretar])
        desviacion = np.std(df[variable_interpretar])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Media", f"{media:.2f}")

        with col2:
            st.metric("Mediana", f"{mediana:.2f}")

        with col3:
            st.metric("Desviación estándar", f"{desviacion:.2f}")

        if media > mediana:
            forma_distribucion = (
                "La media es mayor que la mediana, lo que puede indicar "
                "una distribución con valores altos o sesgo hacia la derecha."
            )
        elif media < mediana:
            forma_distribucion = (
                "La media es menor que la mediana, lo que puede indicar "
                "una distribución con sesgo hacia la izquierda."
            )
        else:
            forma_distribucion = (
                "La media y la mediana son similares, lo que sugiere "
                "una distribución aproximadamente equilibrada."
            )

        st.info(
            f"Para la variable '{variable_interpretar}', la media es "
            f"{media:.2f}, la mediana es {mediana:.2f} y la desviación "
            f"estándar es {desviacion:.2f}. {forma_distribucion}"
        )


    # -----------------------------------------------------
    # ÍTEM 4: ANÁLISIS DE VALORES FALTANTES
    # -----------------------------------------------------
    with tab4:

        st.header("Ítem 4: Análisis de valores faltantes")

        st.write(
            """
            Este análisis identifica la cantidad y el porcentaje de valores
            faltantes en cada variable. Esto permite evaluar si se requiere
            un proceso posterior de limpieza o imputación.
            """
        )

        valores_faltantes = df.isnull().sum()

        tabla_faltantes = pd.DataFrame({
            "Variable": valores_faltantes.index,
            "Cantidad de faltantes": valores_faltantes.values,
            "Porcentaje": (
                valores_faltantes.values / len(df) * 100
            ).round(2)
        })

        st.dataframe(
            tabla_faltantes,
            use_container_width=True,
            hide_index=True
        )

        total_faltantes = int(valores_faltantes.sum())
        variables_con_faltantes = int((valores_faltantes > 0).sum())

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Total de valores faltantes",
                total_faltantes
            )

        with col2:
            st.metric(
                "Variables con faltantes",
                variables_con_faltantes
            )

        if total_faltantes > 0:

            st.subheader("Visualización de valores faltantes")

            datos_grafico = tabla_faltantes[
                tabla_faltantes["Cantidad de faltantes"] > 0
            ]

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.barplot(
                data=datos_grafico,
                x="Cantidad de faltantes",
                y="Variable",
                color="#2E86C1",
                ax=ax
            )

            ax.set_title("Valores faltantes por variable")
            ax.set_xlabel("Cantidad de valores faltantes")
            ax.set_ylabel("Variable")

            st.pyplot(fig)
            plt.close(fig)

            st.warning(
                f"Se encontraron {total_faltantes} valores faltantes "
                f"distribuidos en {variables_con_faltantes} variables. "
                "Estas variables deben revisarse antes de realizar "
                "interpretaciones definitivas."
            )

        else:
            st.success(
                """
                No se encontraron valores faltantes reales en el dataset.
                Por ello, no es necesario aplicar eliminación ni imputación
                y no corresponde generar un gráfico de faltantes.
                """
            )

            st.info(
                """
                Algunas variables categóricas pueden contener la palabra
                `unknown`. Esta es una categoría registrada en el archivo
                y no es reconocida por Pandas como un valor nulo.
                """
            )

    # -----------------------------------------------------
    # ÍTEM 5: DISTRIBUCIÓN DE VARIABLES NUMÉRICAS
    # -----------------------------------------------------
    with tab5:

        st.header("Ítem 5: Distribución de variables numéricas")

        st.write(
            """
            Los histogramas permiten observar la frecuencia, concentración,
            dispersión y posible asimetría de una variable numérica.
            """
        )

        variable_numerica = st.selectbox(
            "Seleccione una variable numérica:",
            numericas,
            key="variable_histograma"
        )

        numero_intervalos = st.slider(
            "Seleccione el número de intervalos del histograma:",
            min_value=5,
            max_value=60,
            value=30,
            step=5
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.histplot(
            data=df,
            x=variable_numerica,
            bins=numero_intervalos,
            kde=True,
            color="#2E86C1",
            ax=ax
        )

        ax.axvline(
            df[variable_numerica].mean(),
            color="red",
            linestyle="--",
            label="Media"
        )

        ax.axvline(
            df[variable_numerica].median(),
            color="green",
            linestyle="--",
            label="Mediana"
        )

        ax.set_title(
            f"Distribución de la variable {variable_numerica}"
        )
        ax.set_xlabel(variable_numerica)
        ax.set_ylabel("Frecuencia")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        asimetria = df[variable_numerica].skew()

        if asimetria > 0.5:
            interpretacion = (
                "La distribución presenta asimetría positiva; es decir, "
                "existen algunos valores altos que extienden la distribución "
                "hacia la derecha."
            )
        elif asimetria < -0.5:
            interpretacion = (
                "La distribución presenta asimetría negativa; es decir, "
                "la distribución se extiende hacia los valores menores."
            )
        else:
            interpretacion = (
                "La distribución es relativamente simétrica, debido a que "
                "su coeficiente de asimetría se encuentra cercano a cero."
            )

        st.info(
            f"La variable '{variable_numerica}' tiene una media de "
            f"{df[variable_numerica].mean():.2f}, una mediana de "
            f"{df[variable_numerica].median():.2f} y una asimetría de "
            f"{asimetria:.2f}. {interpretacion}"
        )


    # -----------------------------------------------------
    # ÍTEM 6: ANÁLISIS DE VARIABLES CATEGÓRICAS
    # -----------------------------------------------------
    with tab6:

        st.header("Ítem 6: Análisis de variables categóricas")

        st.write(
            """
            Este análisis muestra el conteo y la proporción de las categorías
            presentes en una variable seleccionada por el usuario.
            """
        )

        variable_categorica = st.selectbox(
            "Seleccione una variable categórica:",
            categoricas,
            key="variable_categorica"
        )

        conteo_categorias = (
            df[variable_categorica]
            .value_counts(dropna=False)
            .reset_index()
        )

        conteo_categorias.columns = [
            "Categoría",
            "Cantidad"
        ]

        conteo_categorias["Proporción (%)"] = (
            conteo_categorias["Cantidad"] / len(df) * 100
        ).round(2)

        categoria_principal = conteo_categorias.iloc[0]["Categoría"]
        cantidad_principal = conteo_categorias.iloc[0]["Cantidad"]
        proporcion_principal = conteo_categorias.iloc[0]["Proporción (%)"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Número de categorías",
                df[variable_categorica].nunique(dropna=False)
            )

        with col2:
            st.metric(
                "Categoría más frecuente",
                str(categoria_principal)
            )

        with col3:
            st.metric(
                "Proporción principal",
                f"{proporcion_principal:.2f}%"
            )

        st.subheader("Conteos y proporciones")

        st.dataframe(
            conteo_categorias,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Gráfico de barras")

        # Se muestran como máximo las 15 categorías más frecuentes
        datos_grafico = conteo_categorias.head(15)

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            data=datos_grafico,
            x="Cantidad",
            y="Categoría",
            color="#7D3C98",
            ax=ax
        )

        ax.set_title(
            f"Distribución de la variable {variable_categorica}"
        )
        ax.set_xlabel("Cantidad de registros")
        ax.set_ylabel(variable_categorica)

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"En la variable '{variable_categorica}', la categoría más "
            f"frecuente es '{categoria_principal}', con "
            f"{cantidad_principal:,} registros, equivalentes al "
            f"{proporcion_principal:.2f}% del total."
        )
