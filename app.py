import hashlib
from io import BytesIO, StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


# =========================================================
# CONFIGURACIÓN Y PERSONALIZACIÓN
# =========================================================
AUTOR = "Fidel Napoleón Bringas Salazar"
CURSO = "Python for Analytics"
ANIO = 2026

COLUMNAS_ESPERADAS = [
    "age", "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "day_of_week",
    "duration", "campaign", "pdays", "previous", "poutcome",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed", "y"
]

NUMERICAS_ESPERADAS = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]

st.set_page_config(
    page_title="Bank Marketing",
    page_icon="🏦",
    layout="wide"
)

sns.set_theme(style="whitegrid")


# =========================================================
# FUNCIÓN PERSONALIZADA
# =========================================================
def clasificar_variables(dataframe):
    """Clasifica según los tipos reconocidos por Pandas."""
    numericas = dataframe.select_dtypes(
        include=np.number
    ).columns.tolist()

    categoricas = [
        columna for columna in dataframe.columns
        if columna not in numericas
    ]

    return numericas, categoricas


# =========================================================
# PROGRAMACIÓN ORIENTADA A OBJETOS
# =========================================================
class AnalizadorBankMarketing:
    """Encapsula clasificación, estadísticas y visualizaciones."""

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def obtener_dimensiones(self):
        return self.df.shape

    def obtener_vista_previa(self, cantidad=5):
        return self.df.head(cantidad)

    def obtener_info(self):
        buffer = StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def clasificar_variables(self):
        return clasificar_variables(self.df)

    def valores_numericos(self, variable):
        valores = pd.to_numeric(
            self.df[variable], errors="coerce"
        )
        return valores.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

    def obtener_modas(self, variable):
        valores = self.df[variable].dropna()

        if valores.empty:
            return []

        frecuencias = valores.value_counts()

        # Si ningún valor se repite, no hay una moda informativa.
        if frecuencias.max() <= 1:
            return []

        return frecuencias[
            frecuencias == frecuencias.max()
        ].index.tolist()

    def resumen_variable(self, variable):
        valores = self.valores_numericos(variable)

        if valores.empty:
            return None

        arreglo = valores.to_numpy(dtype=float)

        return {
            "cantidad": len(arreglo),
            "media": float(np.mean(arreglo)),
            "mediana": float(np.median(arreglo)),
            "desviacion": (
                float(np.std(arreglo, ddof=1))
                if len(arreglo) > 1 else np.nan
            ),
            "modas": self.obtener_modas(variable)
        }

    def estadisticas_descriptivas(self, columnas):
        if not columnas:
            return pd.DataFrame()

        resumen = self.df[columnas].describe().T
        resumen["variance"] = self.df[columnas].var(ddof=1)

        resumen["Moda(s)"] = [
            ", ".join(str(valor) for valor in self.obtener_modas(c))
            or "Sin moda informativa"
            for c in resumen.index
        ]

        return resumen.rename(columns={
            "count": "Cantidad válida",
            "mean": "Media",
            "std": "Desviación estándar",
            "min": "Mínimo",
            "25%": "Percentil 25",
            "50%": "Mediana",
            "75%": "Percentil 75",
            "max": "Máximo",
            "variance": "Varianza"
        })

    def tabla_faltantes(self):
        total = len(self.df)
        nulos = self.df.isna().sum()
        desconocidos = pd.Series(
            0, index=self.df.columns, dtype="int64"
        )

        _, categoricas = self.clasificar_variables()

        for columna in categoricas:
            desconocidos[columna] = int(
                self.df[columna]
                .astype("string")
                .str.strip()
                .str.lower()
                .eq("unknown")
                .fillna(False)
                .sum()
            )

        return pd.DataFrame({
            "Variable": self.df.columns,
            "Nulos": nulos.values,
            "Nulos (%)": (
                nulos.values / total * 100
            ).round(2),
            "Unknown": desconocidos.values,
            "Unknown (%)": (
                desconocidos.values / total * 100
            ).round(2)
        })

    def conteos_categoricos(self, variable):
        valores = (
            self.df[variable]
            .astype("string")
            .fillna("(Nulo)")
        )

        tabla = valores.value_counts().rename_axis(
            "Categoría"
        ).reset_index(name="Cantidad")

        tabla["Proporción (%)"] = (
            tabla["Cantidad"] / len(valores) * 100
        )

        return tabla

    def comparar_grupos(self, numerica, categorica):
        return (
            self.df.groupby(
                categorica, dropna=False, observed=True
            )[numerica]
            .agg(["count", "mean", "median", "std"])
            .rename(columns={
                "count": "Cantidad válida",
                "mean": "Media",
                "median": "Mediana",
                "std": "Desviación estándar"
            })
        )

    def crear_histograma(
        self, variable, intervalos=30, mostrar_kde=True
    ):
        valores = self.valores_numericos(variable)
        fig, ax = plt.subplots(figsize=(9, 4))

        if valores.empty:
            ax.text(
                0.5, 0.5, "Sin datos numéricos válidos",
                ha="center", va="center", transform=ax.transAxes
            )
            return fig

        sns.histplot(
            x=valores,
            bins=intervalos,
            kde=(
                mostrar_kde
                and len(valores) > 2
                and valores.nunique() > 1
            ),
            color="#2E86C1",
            ax=ax
        )

        ax.axvline(
            valores.mean(), color="red",
            linestyle="--", label="Media"
        )
        ax.axvline(
            valores.median(), color="green",
            linestyle="--", label="Mediana"
        )

        ax.set_title(f"Distribución de {variable}")
        ax.set_xlabel(variable)
        ax.set_ylabel("Frecuencia")
        ax.legend()
        fig.tight_layout()

        return fig

    def crear_barras(
        self, tabla, x, y, titulo, color="#7D3C98"
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            tabla[y].astype(str),
            tabla[x],
            color=color
        )
        ax.invert_yaxis()
        ax.set_title(titulo)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.tight_layout()
        return fig

    def crear_boxplot(self, numerica, categorica):
        datos = self.df[[numerica, categorica]].copy()
        datos[categorica] = (
            datos[categorica]
            .astype("string")
            .fillna("(Nulo)")
        )
        datos = datos.dropna(subset=[numerica])

        fig, ax = plt.subplots(figsize=(10, 5))

        if datos.empty:
            ax.text(
                0.5, 0.5, "Sin datos válidos",
                ha="center", va="center", transform=ax.transAxes
            )
        else:
            sns.boxplot(
                data=datos,
                x=categorica,
                y=numerica,
                color="#5DADE2",
                ax=ax
            )

        ax.set_title(f"{numerica} según {categorica}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    def crear_mapa_calor(self, tabla, titulo):
        ancho = min(15, max(7, tabla.shape[1] * 0.8))
        alto = min(12, max(4, tabla.shape[0] * 0.45))

        fig, ax = plt.subplots(figsize=(ancho, alto))

        sns.heatmap(
            tabla,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            cbar_kws={"label": "Porcentaje por fila"},
            ax=ax
        )

        ax.set_title(titulo)
        fig.tight_layout()
        return fig


# =========================================================
# FUNCIONES DE APOYO
# =========================================================
def mostrar_figura(figura):
    st.pyplot(figura)
    plt.close(figura)


def cargar_dataset(contenido):
    """Lee y valida sin imputar datos ni eliminar filas."""
    dataframe = None

    for codificacion in ("utf-8-sig", "cp1252"):
        try:
            dataframe = pd.read_csv(
                BytesIO(contenido),
                sep=None,
                engine="python",
                encoding=codificacion
            )
            break
        except UnicodeDecodeError:
            continue

    if dataframe is None:
        raise ValueError("No se pudo interpretar la codificación.")

    if dataframe.empty:
        raise ValueError("El CSV no contiene filas de datos.")

    dataframe.columns = dataframe.columns.astype(str).str.strip()

    if dataframe.columns.duplicated().any():
        raise ValueError("Existen nombres de columnas duplicados.")

    faltan = [
        columna for columna in COLUMNAS_ESPERADAS
        if columna not in dataframe.columns
    ]

    if faltan:
        raise ValueError(
            "Faltan columnas del caso Bank Marketing: "
            + ", ".join(faltan)
        )

    for columna in NUMERICAS_ESPERADAS:
        original = dataframe[columna]
        numerica = pd.to_numeric(original, errors="coerce")

        invalidos = original.notna() & numerica.isna()

        if invalidos.any():
            raise ValueError(
                f"La columna '{columna}' contiene "
                f"{int(invalidos.sum())} valores no numéricos."
            )

        if np.isinf(numerica.dropna().to_numpy(dtype=float)).any():
            raise ValueError(
                f"La columna '{columna}' contiene valores infinitos."
            )

        dataframe[columna] = numerica

    _, categoricas = clasificar_variables(dataframe)

    for columna in categoricas:
        dataframe[columna] = (
            dataframe[columna].astype("string").str.strip()
        )
        dataframe[columna] = dataframe[columna].replace(
            "", pd.NA
        )

    dataframe["y"] = dataframe["y"].str.lower()

    etiquetas_invalidas = (
        dataframe["y"].notna()
        & ~dataframe["y"].isin(["yes", "no"])
    )

    if etiquetas_invalidas.any():
        raise ValueError(
            "La variable 'y' solo debe contener yes, no o nulos."
        )

    return dataframe


def mostrar_estadisticas(analizador, variable):
    resumen = analizador.resumen_variable(variable)

    if resumen is None:
        st.warning("Esta variable no tiene datos numéricos válidos.")
        return

    columnas = st.columns(4)
    columnas[0].metric("Media", f"{resumen['media']:.2f}")
    columnas[1].metric("Mediana", f"{resumen['mediana']:.2f}")

    desviacion = resumen["desviacion"]
    columnas[2].metric(
        "Desviación estándar",
        f"{desviacion:.2f}" if pd.notna(desviacion) else "No disponible"
    )

    modas = resumen["modas"]

    if not modas:
        texto_moda = "Sin moda"
    elif len(modas) == 1:
        texto_moda = f"{float(modas[0]):.2f}"
    else:
        texto_moda = f"{len(modas)} modas"

    columnas[3].metric("Moda", texto_moda)

    if len(modas) > 1:
        st.write("Valores modales:", modas)

    st.info(
        f"En '{variable}', el promedio es {resumen['media']:.2f} "
        f"y la mediana es {resumen['mediana']:.2f}. "
        "La mediana divide los valores ordenados en dos mitades. "
        "La desviación estándar resume la dispersión en las mismas "
        "unidades de la variable. Comparar media y mediana no basta "
        "para determinar la forma completa de la distribución."
    )

    st.caption(
        "La desviación estándar y la varianza utilizan ddof=1 "
        "tanto en los indicadores como en las tablas."
    )


# =========================================================
# CARGA PERSISTENTE DURANTE LA SESIÓN
# =========================================================
st.sidebar.title("🏦 Menú principal")

opcion = st.sidebar.selectbox(
    "Seleccione un módulo:",
    ["Home", "Carga del dataset", "Análisis EDA", "Conclusiones finales"]
)

# Permanece visible en todos los módulos para evitar perder
# el estado del cargador al cambiar de sección.
archivo = st.sidebar.file_uploader(
    "Cargar BankMarketing.csv",
    type=["csv"],
    key="archivo_bankmarketing"
)

if archivo is None:
    st.session_state.pop("dataset", None)
    st.session_state.pop("firma_csv", None)
    st.session_state.pop("error_csv", None)

else:
    contenido = archivo.getvalue()
    firma = hashlib.sha256(contenido).hexdigest()

    if firma != st.session_state.get("firma_csv"):
        st.session_state.pop("dataset", None)
        st.session_state.pop("error_csv", None)

        # Evita conservar filtros de otro archivo.
        for clave in list(st.session_state.keys()):
            if clave.startswith("eda_"):
                del st.session_state[clave]

        st.session_state["firma_csv"] = firma

        try:
            st.session_state["dataset"] = cargar_dataset(contenido)
        except Exception as error:
            st.session_state["error_csv"] = str(error)

if "dataset" in st.session_state:
    st.sidebar.success("CSV cargado y validado.")
elif "error_csv" in st.session_state:
    st.sidebar.error("No se pudo validar el CSV.")

st.sidebar.divider()
st.sidebar.write(f"**Autor:** {AUTOR}")
st.sidebar.write(f"**Curso:** {CURSO}")
st.sidebar.write(f"**Año:** {ANIO}")


# =========================================================
# HOME
# =========================================================
if opcion == "Home":
    st.title("🏦 Bank Marketing: Análisis Exploratorio de Datos")
    st.subheader("Presentación del proyecto")

    st.write(
        "Aplicación interactiva para explorar el archivo "
        "BankMarketing.csv mediante estadística descriptiva, "
        "comparación de grupos y visualizaciones."
    )

    st.info(
        "Objetivo: identificar características de los registros, "
        "calidad de los datos y asociaciones con el resultado de "
        "la campaña para apoyar decisiones de análisis."
    )

    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader("Datos del autor")
        st.write(f"**Nombre:** {AUTOR}")
        st.write(f"**Curso / Especialización:** {CURSO}")
        st.write(f"**Año:** {ANIO}")

    with derecha:
        st.subheader("Tecnologías")
        st.write(
            "Python, Streamlit, Pandas, NumPy, Matplotlib, "
            "Seaborn y Programación Orientada a Objetos."
        )

    st.subheader("Descripción del dataset")
    st.write(
        "El archivo reúne características de clientes, contactos "
        "de campañas bancarias e indicadores económicos. La variable "
        "'y' registra el resultado: yes o no."
    )

    st.warning(
        "El proyecto realiza EDA. No construye modelos predictivos "
        "ni demuestra relaciones causales."
    )
    st.write("Para empezar, carga el CSV en la barra lateral.")
    st.stop()


# =========================================================
# VALIDACIÓN PREVIA A CUALQUIER ANÁLISIS
# =========================================================
if "dataset" not in st.session_state:
    st.title(opcion)
    st.warning("Primero carga un CSV válido en la barra lateral.")

    if "error_csv" in st.session_state:
        st.error(st.session_state["error_csv"])

    st.stop()

df = st.session_state["dataset"]
analizador = AnalizadorBankMarketing(df)
numericas, categoricas = analizador.clasificar_variables()


# =========================================================
# CARGA DEL DATASET
# =========================================================
if opcion == "Carga del dataset":
    st.title("📂 Carga del dataset")
    st.success("El archivo fue cargado y validado correctamente.")

    filas, columnas = analizador.obtener_dimensiones()
    col1, col2 = st.columns(2)
    col1.metric("Número de filas", f"{filas:,}")
    col2.metric("Número de columnas", columnas)

    st.subheader("Primeras cinco filas")
    st.dataframe(
        analizador.obtener_vista_previa(),
        use_container_width=True
    )

    st.subheader("Nombres de las columnas")
    st.write(df.columns.tolist())

    st.caption(
        "No se han imputado valores ni eliminado registros. "
        "Se recortaron espacios de los textos, se normalizó 'y' "
        "y se validaron las columnas numéricas."
    )
    st.stop()


# =========================================================
# RESULTADOS COMUNES PARA HALLAZGOS Y CONCLUSIONES
# =========================================================
validos_y = df["y"].isin(["yes", "no"])
base_resultados = df.loc[validos_y].copy()
n_validos = len(base_resultados)

n_si = int(base_resultados["y"].eq("yes").sum())
n_no = int(base_resultados["y"].eq("no").sum())
tasa_si = n_si / n_validos * 100 if n_validos else np.nan

ocupaciones = pd.DataFrame()

if n_validos:
    ocupaciones = (
        base_resultados.assign(
            positiva=base_resultados["y"].eq("yes").astype(int)
        )
        .groupby("job", dropna=False, observed=True)["positiva"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(columns={
            "job": "Ocupación",
            "size": "Registros",
            "sum": "Positivos",
            "mean": "Tasa positiva (%)"
        })
    )

    ocupaciones["Ocupación"] = (
        ocupaciones["Ocupación"].astype("string").fillna("(Nulo)")
    )
    ocupaciones["Tasa positiva (%)"] *= 100
    ocupaciones = ocupaciones.sort_values(
        "Tasa positiva (%)", ascending=False
    )


# =========================================================
# CINCO CONCLUSIONES ORIENTADAS A DECISIONES
# =========================================================
if opcion == "Conclusiones finales":
    st.title("📝 Conclusiones finales")

    if not n_validos:
        st.warning("No hay resultados válidos en 'y'.")
        st.stop()

    st.write(
        "Conclusiones calculadas con el archivo cargado. "
        "Las tasas utilizan registros con y igual a yes o no."
    )

    st.markdown(
        f"**1. Resultado general y evaluación de la campaña.** "
        f"Se registraron {n_si:,} resultados positivos de "
        f"{n_validos:,} resultados válidos ({tasa_si:.2f}%). "
        "Esta tasa sirve como referencia para comparar campañas; "
        "antes de decidir aumentar el presupuesto deben incorporarse "
        "los costos de contacto y el valor de las contrataciones."
    )

    if not ocupaciones.empty:
        principal = ocupaciones.iloc[0]

        st.markdown(
            f"**2. Diferencias por ocupación.** "
            f"El grupo '{principal['Ocupación']}' presenta la tasa "
            f"observada más alta: {principal['Tasa positiva (%)']:.2f}%, "
            f"con {int(principal['Registros']):,} registros. "
            "Conviene investigar las diferencias entre segmentos "
            "considerando sus tamaños, antes de priorizar uno. "
            "La tasa por sí sola no demuestra mayor rentabilidad."
        )

    edad_grupos = base_resultados.groupby("y")["age"].mean()
    edad_si = edad_grupos.get("yes", np.nan)
    edad_no = edad_grupos.get("no", np.nan)

    if pd.notna(edad_si) and pd.notna(edad_no):
        st.markdown(
            f"**3. Edad y segmentación.** "
            f"La edad media es {edad_si:.2f} años en el grupo yes "
            f"y {edad_no:.2f} en el grupo no. "
            "Se deben revisar también las medianas y la dispersión; "
            "estas medias no justifican por sí solas fijar un límite "
            "de edad para seleccionar clientes."
        )
    else:
        st.markdown(
            "**3. Edad y segmentación.** No hay datos suficientes "
            "para comparar la edad media de ambos resultados. "
            "Es necesario completar la información antes de segmentar."
        )

    duracion_grupos = base_resultados.groupby("y")["duration"].mean()
    duracion_si = duracion_grupos.get("yes", np.nan)
    duracion_no = duracion_grupos.get("no", np.nan)

    if pd.notna(duracion_si) and pd.notna(duracion_no):
        st.markdown(
            f"**4. Revisión de los contactos.** "
            f"La duración media es {duracion_si:.2f} en yes "
            f"y {duracion_no:.2f} en no, en las unidades del archivo. "
            "Conviene revisar la calidad y el contenido de los contactos. "
            "La asociación observada no demuestra que prolongarlos "
            "provoque un resultado positivo."
        )
    else:
        st.markdown(
            "**4. Revisión de los contactos.** Faltan datos para "
            "comparar la duración entre ambos grupos. "
            "Conviene revisar su registro antes de tomar decisiones."
        )

    calidad = analizador.tabla_faltantes()
    nulos_total = int(calidad["Nulos"].sum())
    unknown_total = int(calidad["Unknown"].sum())

    st.markdown(
        f"**5. Calidad de información.** "
        f"Se detectaron {nulos_total:,} celdas nulas y "
        f"{unknown_total:,} celdas con 'unknown'. "
        "Antes de decidir cómo tratar esas observaciones, debe "
        "revisarse su significado y concentración por variable. "
        "La ausencia de nulos no garantiza información completa."
    )

    st.warning(
        "Las conclusiones describen este dataset. No prueban "
        "causalidad ni garantizan resultados futuros."
    )
    st.stop()


# =========================================================
# ANÁLISIS EDA: DIEZ ÍTEMS
# =========================================================
st.title("📈 Análisis Exploratorio de Datos")
st.success(f"Dataset disponible: {len(df):,} filas y {df.shape[1]} columnas.")

tabs = st.tabs([
    "1. Información",
    "2. Clasificación",
    "3. Estadísticas",
    "4. Faltantes",
    "5. Distribución",
    "6. Categóricas",
    "7. Num. vs cat.",
    "8. Cat. vs cat.",
    "9. Dinámico",
    "10. Hallazgos"
])


# ---------------------------------------------------------
# ÍTEM 1
# ---------------------------------------------------------
with tabs[0]:
    st.header("Ítem 1: Información general del dataset")
    st.write("Estructura, tipos de datos y conteo de nulos.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", f"{len(df):,}")
    col2.metric("Columnas", df.shape[1])
    col3.metric("Celdas nulas", int(df.isna().sum().sum()))

    st.subheader("Información obtenida con .info()")
    st.code(analizador.obtener_info(), language="text")

    st.subheader("Tipos de datos")
    st.dataframe(
        pd.DataFrame({
            "Variable": df.columns,
            "Tipo": df.dtypes.astype(str).values
        }),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Nulos por variable")
    st.dataframe(
        analizador.tabla_faltantes()[["Variable", "Nulos", "Nulos (%)"]],
        hide_index=True,
        use_container_width=True
    )


# ---------------------------------------------------------
# ÍTEM 2
# ---------------------------------------------------------
with tabs[1]:
    st.header("Ítem 2: Clasificación de variables")
    st.write(
        "Una función personalizada, utilizada por la clase, "
        "clasifica las columnas según su tipo de dato."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Numéricas", len(numericas))
        st.dataframe(
            pd.DataFrame({"Variable numérica": numericas}),
            hide_index=True
        )

    with col2:
        st.metric("Categóricas", len(categoricas))
        st.dataframe(
            pd.DataFrame({"Variable categórica": categoricas}),
            hide_index=True
        )

    resumen_tipos = pd.DataFrame({
        "Tipo": ["Numéricas", "Categóricas"],
        "Cantidad": [len(numericas), len(categoricas)]
    })
    st.bar_chart(resumen_tipos.set_index("Tipo"))

    st.caption(
        "La clasificación técnica no sustituye al diccionario de datos: "
        "una variable numérica puede contener códigos especiales."
    )


# ---------------------------------------------------------
# ÍTEM 3
# ---------------------------------------------------------
with tabs[2]:
    st.header("Ítem 3: Estadísticas descriptivas")
    st.write("Resumen con .describe(), media, mediana, moda y dispersión.")

    st.dataframe(
        analizador.estadisticas_descriptivas(numericas).round(2),
        use_container_width=True
    )

    variable = st.selectbox(
        "Variable a interpretar:",
        numericas,
        key="eda_estadisticas"
    )
    mostrar_estadisticas(analizador, variable)


# ---------------------------------------------------------
# ÍTEM 4
# ---------------------------------------------------------
with tabs[3]:
    st.header("Ítem 4: Análisis de valores faltantes")
    st.write(
        "Se distinguen los nulos reconocidos por Pandas de "
        "los textos 'unknown', que también requieren revisión."
    )

    calidad = analizador.tabla_faltantes()
    st.dataframe(
        calidad, hide_index=True, use_container_width=True
    )

    col1, col2 = st.columns(2)
    col1.metric("Celdas nulas", int(calidad["Nulos"].sum()))
    col2.metric("Celdas unknown", int(calidad["Unknown"].sum()))

    afectados = calidad[
        (calidad["Nulos"] > 0) | (calidad["Unknown"] > 0)
    ]

    if afectados.empty:
        st.info(
            "No se detectaron nulos ni textos 'unknown'. "
            "Esto no descarta otros problemas de calidad."
        )
    else:
        st.bar_chart(
            afectados.set_index("Variable")[["Nulos", "Unknown"]]
        )
        st.warning(
            "Revisar las variables afectadas antes de decidir "
            "si corresponde imputar, conservar o excluir observaciones. "
            "La aplicación no realiza estos cambios automáticamente."
        )


# ---------------------------------------------------------
# ÍTEM 5
# ---------------------------------------------------------
with tabs[4]:
    st.header("Ítem 5: Distribución de variables numéricas")
    st.write(
        "El histograma muestra concentraciones y colas; "
        "las líneas indican media y mediana."
    )

    variable = st.selectbox(
        "Variable numérica:",
        numericas,
        key="eda_histograma"
    )

    intervalos = st.slider(
        "Número de intervalos:",
        min_value=5,
        max_value=60,
        value=30,
        step=5,
        key="eda_intervalos"
    )

    mostrar_kde = st.checkbox(
        "Mostrar curva de densidad",
        value=True,
        key="eda_kde"
    )

    mostrar_figura(
        analizador.crear_histograma(variable, intervalos, mostrar_kde)
    )

    valores = analizador.valores_numericos(variable)
    asimetria = valores.skew()

    if pd.isna(asimetria):
        st.info("No hay datos suficientes para interpretar la asimetría.")
    else:
        if asimetria > 0.5:
            comentario = "Se observa asimetría positiva."
        elif asimetria < -0.5:
            comentario = "Se observa asimetría negativa."
        else:
            comentario = (
                "La asimetría calculada es cercana a cero; "
                "esto no implica normalidad."
            )

        st.info(
            f"Asimetría de '{variable}': {asimetria:.2f}. "
            f"{comentario} El umbral de 0.5 es una guía descriptiva."
        )

    st.caption(
        "Antes de interpretar extremos, comprueba si corresponden "
        "a mediciones o a códigos especiales del diccionario."
    )


# ---------------------------------------------------------
# ÍTEM 6
# ---------------------------------------------------------
with tabs[5]:
    st.header("Ítem 6: Análisis de variables categóricas")
    st.write("Conteos, proporciones y categoría modal.")

    variable = st.selectbox(
        "Variable categórica:",
        categoricas,
        key="eda_categorica"
    )

    conteos = analizador.conteos_categoricos(variable)
    st.dataframe(
        conteos.round(2),
        hide_index=True,
        use_container_width=True
    )

    mostrar_figura(
        analizador.crear_barras(
            conteos.head(20),
            "Cantidad",
            "Categoría",
            f"Frecuencias de {variable}"
        )
    )

    if not conteos.empty:
        maximo = conteos["Cantidad"].max()
        modales = conteos.loc[
            conteos["Cantidad"] == maximo, "Categoría"
        ].tolist()

        st.info(
            f"Categoría(s) más frecuente(s): {', '.join(modales)}. "
            f"Cada una tiene {int(maximo):,} registros "
            f"({maximo / len(df) * 100:.2f}%)."
        )

    if len(conteos) > 20:
        st.caption("El gráfico muestra las 20 categorías más frecuentes.")


# ---------------------------------------------------------
# ÍTEM 7
# ---------------------------------------------------------
with tabs[6]:
    st.header("Ítem 7: Numérico vs categórico")
    st.write("Comparación de distribución y estadísticos por grupo.")

    col1, col2 = st.columns(2)

    with col1:
        variable_num = st.selectbox(
            "Variable numérica:",
            numericas,
            key="eda_bivariada_num"
        )

    with col2:
        variable_cat = st.selectbox(
            "Variable categórica:",
            categoricas,
            index=categoricas.index("y"),
            key="eda_bivariada_cat"
        )

    comparacion = analizador.comparar_grupos(
        variable_num, variable_cat
    )

    st.dataframe(comparacion.round(2), use_container_width=True)

    mostrar_figura(
        analizador.crear_boxplot(variable_num, variable_cat)
    )

    medias_validas = comparacion["Media"].dropna()

    if not medias_validas.empty:
        st.info(
            f"Las medias de '{variable_num}' oscilan entre "
            f"{medias_validas.min():.2f} y "
            f"{medias_validas.max():.2f} entre los grupos. "
            "Revisa también medianas, tamaños y dispersión. "
            "La comparación es descriptiva, no una prueba de causalidad."
        )


# ---------------------------------------------------------
# ÍTEM 8
# ---------------------------------------------------------
with tabs[7]:
    st.header("Ítem 8: Categórico vs categórico")
    st.write(
        "Tabla de contingencia y porcentajes dentro de cada fila."
    )

    col1, col2 = st.columns(2)

    with col1:
        primera = st.selectbox(
            "Primera variable:",
            categoricas,
            index=categoricas.index("education"),
            key="eda_cat_primera"
        )

    opciones_segunda = [c for c in categoricas if c != primera]

    with col2:
        segunda = st.selectbox(
            "Segunda variable:",
            opciones_segunda,
            index=(
                opciones_segunda.index("y")
                if "y" in opciones_segunda else 0
            ),
            key="eda_cat_segunda"
        )

    serie_primera = df[primera].astype("string").fillna("(Nulo)")
    serie_segunda = df[segunda].astype("string").fillna("(Nulo)")

    frecuencias = pd.crosstab(serie_primera, serie_segunda)
    porcentajes = frecuencias.div(
        frecuencias.sum(axis=1), axis=0
    ) * 100

    st.subheader("Frecuencias")
    st.dataframe(frecuencias, use_container_width=True)

    st.subheader("Porcentajes por fila")
    st.dataframe(porcentajes.round(2), use_container_width=True)

    mostrar_figura(
        analizador.crear_mapa_calor(
            porcentajes,
            f"{primera} frente a {segunda}"
        )
    )

    st.info(
        "Cada fila suma aproximadamente 100%. Los porcentajes permiten "
        "comparar grupos de distinto tamaño, pero los grupos pequeños "
        "pueden mostrar tasas extremas con pocos casos."
    )


# ---------------------------------------------------------
# ÍTEM 9
# ---------------------------------------------------------
with tabs[8]:
    st.header("Ítem 9: Análisis dinámico")
    st.write("Selecciona columnas y categorías para explorar un subconjunto.")

    seleccionadas = st.multiselect(
        "Columnas a visualizar:",
        options=df.columns.tolist(),
        default=["age", "job", "education", "duration", "y"],
        key="eda_columnas"
    )

    filtro = st.selectbox(
        "Variable categórica para filtrar:",
        categoricas,
        index=categoricas.index("y"),
        key="eda_filtro"
    )

    serie_filtro = df[filtro].astype("string").fillna("(Nulo)")
    opciones = sorted(serie_filtro.unique().tolist())

    elegidas = st.multiselect(
        f"Valores de {filtro}:",
        options=opciones,
        default=opciones,
        key=f"eda_valores_{filtro}"
    )

    if not seleccionadas or not elegidas:
        st.warning("Selecciona al menos una columna y una categoría.")

    else:
        filtrado = df.loc[serie_filtro.isin(elegidas)].copy()

        if filtrado.empty:
            st.warning("No hay registros para esta selección.")
        else:
            st.success(
                f"{len(filtrado):,} registros y "
                f"{len(seleccionadas)} columnas seleccionadas."
            )

            st.caption("Vista previa: hasta 100 registros.")
            st.dataframe(
                filtrado[seleccionadas].head(100),
                use_container_width=True
            )

            analizador_filtrado = AnalizadorBankMarketing(filtrado)
            numericas_elegidas = [
                c for c in seleccionadas if c in numericas
            ]

            if numericas_elegidas:
                st.dataframe(
                    analizador_filtrado.estadisticas_descriptivas(
                        numericas_elegidas
                    ).round(2),
                    use_container_width=True
                )

                variable = st.selectbox(
                    "Variable del histograma filtrado:",
                    numericas_elegidas,
                    key="eda_grafico_filtrado"
                )

                mostrar_figura(
                    analizador_filtrado.crear_histograma(variable)
                )

                mostrar_estadisticas(analizador_filtrado, variable)

            else:
                variable = st.selectbox(
                    "Variable del gráfico categórico filtrado:",
                    seleccionadas,
                    key="eda_grafico_cat_filtrado"
                )

                conteos = analizador_filtrado.conteos_categoricos(variable)

                mostrar_figura(
                    analizador_filtrado.crear_barras(
                        conteos.head(20),
                        "Cantidad",
                        "Categoría",
                        f"{variable}: datos filtrados"
                    )
                )

            st.caption(
                "Las estadísticas y gráficos utilizan todos los registros "
                "filtrados, no solo las 100 filas de la vista previa."
            )


# ---------------------------------------------------------
# ÍTEM 10
# ---------------------------------------------------------
with tabs[9]:
    st.header("Ítem 10: Hallazgos clave")
    st.write(
        "Resumen del dataset completo. Los filtros del ítem 9 "
        "no modifican estos resultados."
    )

    if not n_validos:
        st.warning("No existen valores yes/no válidos en la variable y.")

    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Registros totales", f"{len(df):,}")
        col2.metric("Resultados positivos", f"{n_si:,}")
        col3.metric("Resultados negativos", f"{n_no:,}")
        col4.metric("Tasa positiva", f"{tasa_si:.2f}%")

        st.caption(
            f"Denominador de la tasa: {n_validos:,} resultados válidos. "
            f"Resultados nulos excluidos: {len(df) - n_validos:,}."
        )

        resultados = pd.DataFrame({
            "Respuesta": ["no", "yes"],
            "Cantidad": [n_no, n_si]
        })

        mostrar_figura(
            analizador.crear_barras(
                resultados,
                "Cantidad",
                "Respuesta",
                "Resultado general de la campaña",
                color="#2E86C1"
            )
        )

        st.subheader("Tasa positiva por ocupación")
        st.dataframe(
            ocupaciones.round(2),
            hide_index=True,
            use_container_width=True
        )

        mostrar_figura(
            analizador.crear_barras(
                ocupaciones,
                "Tasa positiva (%)",
                "Ocupación",
                "Resultados positivos dentro de cada ocupación",
                color="#17A589"
            )
        )

        edad = base_resultados.groupby("y")["age"].agg(
            ["mean", "median"]
        )
        duracion = base_resultados.groupby("y")["duration"].mean()

        st.subheader("Hallazgos derivados del EDA")
        st.write(
            f"1. Se registran {n_si:,} resultados positivos: "
            f"{tasa_si:.2f}% de los resultados válidos."
        )
        st.write(
            f"2. Se registran {n_no:,} resultados negativos: "
            f"{n_no / n_validos * 100:.2f}%."
        )

        st.write("3. Edad media y mediana por resultado:")
        st.dataframe(
            edad.rename(columns={
                "mean": "Media",
                "median": "Mediana"
            }).round(2)
        )

        st.write("4. Duración media por resultado:")
        st.dataframe(
            duracion.rename("Duración media").to_frame().round(2)
        )

        principal = ocupaciones.iloc[0]
        st.write(
            f"5. La ocupación con mayor tasa observada es "
            f"'{principal['Ocupación']}': "
            f"{principal['Tasa positiva (%)']:.2f}%, "
            f"con {int(principal['Registros']):,} registros."
        )

        st.warning(
            "Una tasa alta en un grupo pequeño requiere cautela. "
            "Los resultados no demuestran causalidad ni equivalen "
            "a predicciones."
        )

        st.info(
            "Consulta el módulo 'Conclusiones finales' del menú "
            "para ver cinco conclusiones orientadas a decisiones."
        )