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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "1️⃣ Información general",
    "2️⃣ Clasificación",
    "3️⃣ Estadísticas",
    "4️⃣ Valores faltantes",
    "5️⃣ Distribución numérica",
    "6️⃣ Variables categóricas",
    "7️⃣ Numérico vs categórico",
    "8️⃣ Categórico vs categórico",
    "9️⃣ Análisis dinámico",
    "🔟 Hallazgos clave"
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
    # -----------------------------------------------------
    # ÍTEM 7: ANÁLISIS BIVARIADO NUMÉRICO VS CATEGÓRICO
    # -----------------------------------------------------
    with tab7:

        st.header("Ítem 7: Análisis bivariado — numérico vs categórico")

        st.write(
            """
            Este análisis compara la distribución de una variable numérica
            entre los diferentes grupos de una variable categórica.
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            variable_num_bivariada = st.selectbox(
                "Seleccione la variable numérica:",
                numericas,
                index=numericas.index("age") if "age" in numericas else 0,
                key="numerica_bivariada"
            )

        with col2:
            indice_y = categoricas.index("y") if "y" in categoricas else 0

            variable_cat_bivariada = st.selectbox(
                "Seleccione la variable categórica:",
                categoricas,
                index=indice_y,
                key="categorica_bivariada"
            )

        resumen_bivariado = (
            df.groupby(variable_cat_bivariada)[variable_num_bivariada]
            .agg(["count", "mean", "median", "std"])
            .round(2)
            .reset_index()
        )

        resumen_bivariado.columns = [
            variable_cat_bivariada,
            "Cantidad",
            "Media",
            "Mediana",
            "Desviación estándar"
        ]

        st.subheader("Resumen por categoría")

        st.dataframe(
            resumen_bivariado,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Comparación mediante diagrama de cajas")

        fig, ax = plt.subplots(figsize=(11, 6))

        sns.boxplot(
            data=df,
            x=variable_cat_bivariada,
            y=variable_num_bivariada,
            color="#5DADE2",
            ax=ax
        )

        ax.set_title(
            f"{variable_num_bivariada} según {variable_cat_bivariada}"
        )
        ax.set_xlabel(variable_cat_bivariada)
        ax.set_ylabel(variable_num_bivariada)
        ax.tick_params(axis="x", rotation=45)

        st.pyplot(fig)
        plt.close(fig)

        categoria_media_mayor = resumen_bivariado.loc[
            resumen_bivariado["Media"].idxmax()
        ]

        categoria_media_menor = resumen_bivariado.loc[
            resumen_bivariado["Media"].idxmin()
        ]

        st.info(
            f"La categoría '{categoria_media_mayor[variable_cat_bivariada]}' "
            f"presenta la mayor media de {variable_num_bivariada}, con "
            f"{categoria_media_mayor['Media']:.2f}. La categoría "
            f"'{categoria_media_menor[variable_cat_bivariada]}' presenta "
            f"la menor media, con {categoria_media_menor['Media']:.2f}."
        )


    # -----------------------------------------------------
    # ÍTEM 8: ANÁLISIS BIVARIADO CATEGÓRICO VS CATEGÓRICO
    # -----------------------------------------------------
    with tab8:

        st.header("Ítem 8: Análisis bivariado — categórico vs categórico")

        st.write(
            """
            La tabla de contingencia permite comparar las frecuencias y
            proporciones existentes entre dos variables categóricas.
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            primera_categorica = st.selectbox(
                "Seleccione la primera variable:",
                categoricas,
                index=categoricas.index("education")
                if "education" in categoricas else 0,
                key="primera_categorica"
            )

        opciones_segunda = [
            variable for variable in categoricas
            if variable != primera_categorica
        ]

        with col2:
            indice_segunda = (
                opciones_segunda.index("y")
                if "y" in opciones_segunda else 0
            )

            segunda_categorica = st.selectbox(
                "Seleccione la segunda variable:",
                opciones_segunda,
                index=indice_segunda,
                key="segunda_categorica"
            )

        tabla_contingencia = pd.crosstab(
            df[primera_categorica],
            df[segunda_categorica]
        )

        tabla_porcentajes = pd.crosstab(
            df[primera_categorica],
            df[segunda_categorica],
            normalize="index"
        ) * 100

        st.subheader("Tabla de frecuencias")

        st.dataframe(
            tabla_contingencia,
            use_container_width=True
        )

        st.subheader("Proporciones por fila (%)")

        st.dataframe(
            tabla_porcentajes.round(2),
            use_container_width=True
        )

        st.subheader("Mapa de calor de proporciones")

        fig, ax = plt.subplots(figsize=(11, 6))

        sns.heatmap(
            tabla_porcentajes,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(
            f"Relación entre {primera_categorica} y {segunda_categorica}"
        )
        ax.set_xlabel(segunda_categorica)
        ax.set_ylabel(primera_categorica)

        st.pyplot(fig)
        plt.close(fig)

        combinacion_principal = tabla_contingencia.stack().idxmax()
        frecuencia_principal = int(tabla_contingencia.stack().max())

        st.info(
            f"La combinación más frecuente corresponde a "
            f"'{primera_categorica} = {combinacion_principal[0]}' y "
            f"'{segunda_categorica} = {combinacion_principal[1]}', con "
            f"{frecuencia_principal:,} registros."
        )

    # -----------------------------------------------------
    # ÍTEM 9: ANÁLISIS BASADO EN PARÁMETROS
    # -----------------------------------------------------
    with tab9:

        st.header("Ítem 9: Análisis basado en parámetros seleccionados")

        st.write(
            """
            Este módulo permite al usuario seleccionar columnas y aplicar
            filtros de manera interactiva para generar un análisis dinámico.
            """
        )

        columnas_seleccionadas = st.multiselect(
            "Seleccione las columnas que desea visualizar:",
            options=df.columns.tolist(),
            default=["age", "job", "education", "duration", "y"]
        )

        variable_filtro = st.selectbox(
            "Seleccione una variable categórica para filtrar:",
            categoricas,
            index=categoricas.index("y") if "y" in categoricas else 0,
            key="variable_filtro"
        )

        categorias_disponibles = sorted(
            df[variable_filtro].dropna().astype(str).unique().tolist()
        )

        categorias_elegidas = st.multiselect(
            f"Seleccione los valores de {variable_filtro}:",
            options=categorias_disponibles,
            default=categorias_disponibles,
            key="categorias_filtro"
        )

        if len(categorias_elegidas) == 0:
            st.warning("Seleccione al menos una categoría para continuar.")

        elif len(columnas_seleccionadas) == 0:
            st.warning("Seleccione al menos una columna para visualizar.")

        else:
            df_filtrado = df[
                df[variable_filtro].astype(str).isin(categorias_elegidas)
            ]

            st.success(
                f"El filtro devuelve {df_filtrado.shape[0]:,} registros "
                f"y se muestran {len(columnas_seleccionadas)} columnas."
            )

            st.dataframe(
                df_filtrado[columnas_seleccionadas].head(100),
                use_container_width=True
            )

            numericas_seleccionadas = [
                columna for columna in columnas_seleccionadas
                if columna in numericas
            ]

            if len(numericas_seleccionadas) > 0:

                st.subheader("Resumen de columnas numéricas seleccionadas")

                st.dataframe(
                    df_filtrado[numericas_seleccionadas]
                    .describe()
                    .T
                    .round(2),
                    use_container_width=True
                )

                variable_grafico = st.selectbox(
                    "Seleccione una variable para el gráfico dinámico:",
                    numericas_seleccionadas,
                    key="grafico_dinamico"
                )

                fig, ax = plt.subplots(figsize=(10, 5))

                sns.histplot(
                    data=df_filtrado,
                    x=variable_grafico,
                    bins=30,
                    kde=True,
                    color="#17A589",
                    ax=ax
                )

                ax.set_title(
                    f"Distribución dinámica de {variable_grafico}"
                )
                ax.set_xlabel(variable_grafico)
                ax.set_ylabel("Frecuencia")

                st.pyplot(fig)
                plt.close(fig)

                st.info(
                    f"La variable '{variable_grafico}' tiene una media de "
                    f"{df_filtrado[variable_grafico].mean():.2f} dentro "
                    f"del conjunto de datos filtrado."
                )

            else:
                st.info(
                    """
                    No se seleccionaron variables numéricas. Seleccione al
                    menos una para generar estadísticas y un histograma.
                    """
                )


    # -----------------------------------------------------
    # ÍTEM 10: HALLAZGOS CLAVE
    # -----------------------------------------------------
    with tab10:

        st.header("Ítem 10: Hallazgos clave")

        st.write(
            """
            Esta sección resume los principales resultados obtenidos mediante
            el análisis exploratorio del dataset Bank Marketing.
            """
        )

        if "y" in df.columns:

            total_clientes = len(df)
            total_si = int((df["y"] == "yes").sum())
            total_no = int((df["y"] == "no").sum())
            tasa_respuesta = total_si / total_clientes * 100

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Clientes analizados", f"{total_clientes:,}")

            with col2:
                st.metric("Respuesta positiva", f"{total_si:,}")

            with col3:
                st.metric("Respuesta negativa", f"{total_no:,}")

            with col4:
                st.metric("Tasa de respuesta positiva", f"{tasa_respuesta:.2f}%")

            st.subheader("Resultado general de la campaña")

            resultados_campana = (
                df["y"]
                .value_counts()
                .rename_axis("Respuesta")
                .reset_index(name="Cantidad")
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.barplot(
                data=resultados_campana,
                x="Respuesta",
                y="Cantidad",
                hue="Respuesta",
                palette={
                    "no": "#E74C3C",
                    "yes": "#2ECC71"
                },
                legend=False,
                ax=ax
            )

            ax.set_title("Respuesta a la campaña bancaria")
            ax.set_xlabel("Respuesta")
            ax.set_ylabel("Cantidad de clientes")

            st.pyplot(fig)
            plt.close(fig)

            edad_si = df.loc[df["y"] == "yes", "age"].mean()
            edad_no = df.loc[df["y"] == "no", "age"].mean()

            duracion_si = df.loc[df["y"] == "yes", "duration"].mean()
            duracion_no = df.loc[df["y"] == "no", "duration"].mean()

            tasa_por_trabajo = (
                df.assign(
                    respuesta_positiva=(df["y"] == "yes").astype(int)
                )
                .groupby("job")["respuesta_positiva"]
                .agg(["mean", "count"])
                .reset_index()
            )

            tasa_por_trabajo["Tasa positiva (%)"] = (
                tasa_por_trabajo["mean"] * 100
            ).round(2)

            tasa_por_trabajo = tasa_por_trabajo.sort_values(
                "Tasa positiva (%)",
                ascending=False
            )

            st.subheader("Tasa de respuesta positiva por ocupación")

            fig, ax = plt.subplots(figsize=(10, 6))

            sns.barplot(
                data=tasa_por_trabajo,
                x="Tasa positiva (%)",
                y="job",
                color="#3498DB",
                ax=ax
            )

            ax.set_title("Respuesta positiva según ocupación")
            ax.set_xlabel("Tasa de respuesta positiva (%)")
            ax.set_ylabel("Ocupación")

            st.pyplot(fig)
            plt.close(fig)

            mejor_trabajo = tasa_por_trabajo.iloc[0]

            st.subheader("Principales insights derivados del EDA")

            st.markdown(
                f"""
                1. La campaña obtuvo **{total_si:,} respuestas positivas**,
                   equivalentes al **{tasa_respuesta:.2f}%** del total.

                2. La mayoría de los registros corresponde a respuestas
                   negativas: **{total_no:,} clientes**.

                3. La edad promedio de quienes respondieron positivamente fue
                   de **{edad_si:.2f} años**, frente a **{edad_no:.2f} años**
                   entre quienes respondieron negativamente.

                4. La duración promedio del contacto fue de
                   **{duracion_si:.2f} segundos** en las respuestas positivas
                   y de **{duracion_no:.2f} segundos** en las negativas.

                5. La ocupación con mayor proporción de respuestas positivas
                   fue **{mejor_trabajo['job']}**, con una tasa de
                   **{mejor_trabajo['Tasa positiva (%)']:.2f}%**.
                """
            )

            st.warning(
                """
                Los resultados describen asociaciones observadas en el
                dataset y no demuestran relaciones de causa y efecto.
                Además, este proyecto no desarrolla modelos predictivos.
                """
            )

        else:
            st.error(
                "No se encontró la variable objetivo 'y' en el dataset."
            )
