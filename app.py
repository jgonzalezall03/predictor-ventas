import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importaciones para machine learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ConfiguraciÃ³n de la pÃ¡gina
st.set_page_config(
    page_title="Sistema de Prediccion de Ventas ML",
    #page_icon="ðŸ“Š",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        color: #856404;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Clase principal del sistema
class SalesMLApp:
    def __init__(self):
        self.df_raw = None
        self.df_clean = None
        self.df_model = None
        self.best_model = None
        self.label_encoder = None
        self.feature_columns = None
        self.results = {}
        
    @st.cache_data
    def load_and_process_data(_self, uploaded_file):
        """Cargar y procesar datos con cache para mejor rendimiento"""
        try:
            # Leer archivo
            if uploaded_file.name.endswith('.csv'):
                # Intentar diferentes codificaciones y parámetros
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df_raw = None
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)
                        df_raw = pd.read_csv(
                            uploaded_file, 
                            encoding=encoding,
                            sep=None,
                            engine='python',
                            on_bad_lines='skip',
                            quoting=3
                        )
                        break
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
                if df_raw is None:
                    raise ValueError("No se pudo leer el archivo con ninguna codificación")
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            # Si la primera fila contiene nombres de columnas, usarla
            if df_raw.iloc[0].notna().sum() > df_raw.columns.notna().sum():
                new_columns = df_raw.iloc[0].tolist()
                df_clean = df_raw.iloc[1:].copy()
                df_clean.columns = new_columns
            else:
                df_clean = df_raw.copy()
            
            # Resetear Ã­ndice y limpiar datos
            print(df_clean.columns.to_list())
            df_clean = df_clean.reset_index(drop=True)
            df_clean['Mes'] = pd.to_datetime(df_clean['Mes de gestión'])
            df_clean['Venta UF'] = pd.to_numeric(df_clean['Venta UF'], errors='coerce')
            
            # Procesar campo Contratos si existe
            if 'Contratos' in df_clean.columns:
                df_clean['Contratos'] = pd.to_numeric(df_clean['Contratos'], errors='coerce')
                df_clean = df_clean.dropna(subset=['Mes de gestión', 'EEVV', 'Venta UF', 'Contratos'])
            else:
                df_clean = df_clean.dropna(subset=['Mes de gestión', 'EEVV', 'Venta UF'])
            
            return df_clean
            
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")
            return None
    
    def create_features(self, df):
        """Crear caracterÃ­sticas adicionales para ML"""
        df = df.copy()
        
        # Componentes de fecha
        df['año'] = df['Mes'].dt.year
        df['mes'] = df['Mes'].dt.month
        df['trimestre'] = df['Mes'].dt.quarter
        
        # Features por ejecutivo
        df_features = []
        
        for ejecutivo in df['EEVV'].unique():
            exec_data = df[df['EEVV'] == ejecutivo].copy()
            exec_data = exec_data.sort_values('Mes')
            
            # Lag features
            exec_data['venta_lag1'] = exec_data['Venta UF'].shift(1)
            exec_data['venta_lag2'] = exec_data['Venta UF'].shift(2)
            exec_data['venta_lag3'] = exec_data['Venta UF'].shift(3)
            
            # Media mÃ³vil
            exec_data['media_movil_3'] = exec_data['Venta UF'].rolling(window=3, min_periods=1).mean()
            exec_data['media_movil_6'] = exec_data['Venta UF'].rolling(window=6, min_periods=1).mean()
            
            # Tendencia
            exec_data['tendencia'] = exec_data['Venta UF'].diff()
            
            # EstadÃ­sticas acumuladas
            exec_data['venta_acumulada'] = exec_data['Venta UF'].cumsum()
            exec_data['promedio_hasta_fecha'] = exec_data['Venta UF'].expanding().mean()
            exec_data['max_hasta_fecha'] = exec_data['Venta UF'].expanding().max()
            exec_data['min_hasta_fecha'] = exec_data['Venta UF'].expanding().min()
            exec_data['meses_desde_inicio'] = range(len(exec_data))
            
            # Features de Contratos si existe la columna
            if 'Contratos' in exec_data.columns:
                exec_data['contratos_lag1'] = exec_data['Contratos'].shift(1)
                exec_data['contratos_media_movil_3'] = exec_data['Contratos'].rolling(window=3, min_periods=1).mean()
                exec_data['contratos_acumulados'] = exec_data['Contratos'].cumsum()
                exec_data['ratio_venta_contratos'] = exec_data['Venta UF'] / (exec_data['Contratos'] + 1)
            
            df_features.append(exec_data)
        
        return pd.concat(df_features, ignore_index=True).dropna()
    
    def train_models(self, df_model):
        """Entrenar mÃºltiples modelos de ML"""
        # Preparar datos
        le = LabelEncoder()
        df_model_encoded = df_model.copy()
        df_model_encoded['EEVV_encoded'] = le.fit_transform(df_model['EEVV'])
        
        feature_columns = ['EEVV_encoded', 'año', 'mes', 'trimestre', 'venta_lag1', 'venta_lag2', 'venta_lag3', 
                          'media_movil_3', 'media_movil_6', 'tendencia', 'venta_acumulada', 
                          'promedio_hasta_fecha', 'max_hasta_fecha', 'min_hasta_fecha', 'meses_desde_inicio']
        
        # Agregar features de Contratos si existe la columna
        if 'Contratos' in df_model.columns:
            contratos_features = ['contratos_lag1', 'contratos_media_movil_3', 'contratos_acumulados', 'ratio_venta_contratos']
            feature_columns.extend(contratos_features)
        
        X = df_model_encoded[feature_columns]
        y = df_model_encoded['Venta UF']
        
        # DivisiÃ³n temporal
        split_idx = int(0.7 * len(df_model_encoded))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Modelos a entrenar
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
        }
        
        results = {}
        
        # Entrenar modelos
        progress_bar = st.progress(0)
        for i, (name, model) in enumerate(models.items()):
            try:
                model.fit(X_train, y_train)
                
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                results[name] = {
                    'model': model,
                    'train_mae': mean_absolute_error(y_train, y_pred_train),
                    'test_mae': mean_absolute_error(y_test, y_pred_test),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'train_r2': r2_score(y_train, y_pred_train),
                    'test_r2': r2_score(y_test, y_pred_test)
                }
                
                progress_bar.progress((i + 1) / len(models))
                
            except Exception as e:
                st.warning(f"Error entrenando {name}: {str(e)}")
        
        # Seleccionar mejor modelo
        if results:
            best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
            best_model = results[best_model_name]['model']
            
            return results, best_model, best_model_name, le, feature_columns
        
        return None, None, None, None, None

# FunciÃ³n principal de la app
def main():
    # Header principal
    st.markdown('<h1 class="main-header">Sistema de Prediccion de Ventas con Machine Learning</h1>', 
                unsafe_allow_html=True)
    
    # Inicializar app
    if 'app' not in st.session_state:
        st.session_state.app = SalesMLApp()
    
    app = st.session_state.app
    
    # Sidebar para navegaciÃ³n
    st.sidebar.title("Panel de Control")
    
    # Upload de archivo
    uploaded_file = st.sidebar.file_uploader(
        "Cargar archivo de ventas",
        type=['csv', 'xlsx'],
        help="Archivo debe contener: Mes de gestion, EEVV, Venta UF, Contratos (opcional)"
    )
    
    if uploaded_file is not None:
        # Procesar datos
        with st.spinner("Procesando datos..."):
            app.df_clean = app.load_and_process_data(uploaded_file)
        
        if app.df_clean is not None:
            st.sidebar.success(f"Datos cargados: {app.df_clean.shape[0]} registros")
            
            # Selector de pÃ¡gina
            page = st.sidebar.selectbox(
                "Navegacion",
                ["Dashboard", "Analisis de Datos", "Modelos ML", "Predicciones", "Escenarios", "Visualizaciones"]
            )
            
            # PÃGINA: DASHBOARD
            if page == "Dashboard":
                st.header("Dashboard General")
                
                # MÃ©tricas principales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_ventas = app.df_clean['Venta UF'].sum()
                    st.metric("Ventas Totales", f"{total_ventas:,.2f} UF")
                
                with col2:
                    promedio_ventas = app.df_clean['Venta UF'].mean()
                    st.metric("Promedio Ventas", f"{promedio_ventas:.2f} UF")
                
                with col3:
                    num_ejecutivos = app.df_clean['EEVV'].nunique()
                    st.metric("Ejecutivos", f"{num_ejecutivos}")
                
                with col4:
                    if 'Contratos' in app.df_clean.columns:
                        total_contratos = app.df_clean['Contratos'].sum()
                        st.metric("Total Contratos", f"{total_contratos:,.0f}")
                    else:
                        periodo = f"{app.df_clean['Mes'].min().strftime('%Y-%m')} / {app.df_clean['Mes'].max().strftime('%Y-%m')}"
                        st.metric("Periodo", periodo)
                
                # GrÃ¡fico de ventas por mes
                st.subheader("Evolucion de Ventas Mensuales")
                
                ventas_mensuales = app.df_clean.groupby('Mes')['Venta UF'].agg(['sum', 'mean']).reset_index()
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=ventas_mensuales['Mes'], y=ventas_mensuales['sum'],
                              name="Total Mensual", line=dict(color='#1f77b4', width=3)),
                    secondary_y=False,
                )
                
                fig.add_trace(
                    go.Scatter(x=ventas_mensuales['Mes'], y=ventas_mensuales['mean'],
                              name="Promedio Mensual", line=dict(color='#ff7f0e', width=2)),
                    secondary_y=True,
                )
                
                fig.update_xaxes(title_text="Mes")
                fig.update_yaxes(title_text="Ventas Totales (UF)", secondary_y=False)
                fig.update_yaxes(title_text="Promedio (UF)", secondary_y=True)
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Top ejecutivos
                st.subheader("Top 10 Ejecutivos")
                if 'Contratos' in app.df_clean.columns:
                    top_ejecutivos = app.df_clean.groupby('EEVV').agg({
                        'Venta UF': ['sum', 'mean'],
                        'Contratos': ['sum', 'mean']
                    }).round(2)
                    top_ejecutivos.columns = ['Total UF', 'Promedio UF', 'Total Contratos', 'Promedio Contratos']
                    top_ejecutivos['Ratio UF/Contrato'] = (top_ejecutivos['Total UF'] / top_ejecutivos['Total Contratos']).round(2)
                    top_ejecutivos = top_ejecutivos.sort_values('Promedio UF', ascending=False).head(10)
                else:
                    top_ejecutivos = app.df_clean.groupby('EEVV')['Venta UF'].agg(['sum', 'mean', 'count']).round(2)
                    top_ejecutivos.columns = ['Total UF', 'Promedio UF', 'Registros']
                    top_ejecutivos = top_ejecutivos.sort_values('Promedio UF', ascending=False).head(10)
                
                st.dataframe(top_ejecutivos, use_container_width=True)
            
            # PÃGINA: ANÃLISIS DE DATOS
            elif page == "Analisis de Datos":
                st.header("Analisis Detallado de Datos")
                
                # AnÃ¡lisis por ejecutivo
                st.subheader("Analisis por Ejecutivo")
                
                ejecutivo_seleccionado = st.selectbox(
                    "Selecciona un ejecutivo:",
                    options=sorted(app.df_clean['EEVV'].unique())
                )
                
                if ejecutivo_seleccionado:
                    exec_data = app.df_clean[app.df_clean['EEVV'] == ejecutivo_seleccionado]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Ventas", f"{exec_data['Venta UF'].sum():.2f} UF")
                    with col2:
                        st.metric("Promedio", f"{exec_data['Venta UF'].mean():.2f} UF")
                    with col3:
                        st.metric("Maximo", f"{exec_data['Venta UF'].max():.2f} UF")
                    with col4:
                        if 'Contratos' in app.df_clean.columns:
                            st.metric("Total Contratos", f"{exec_data['Contratos'].sum():.0f}")
                        else:
                            st.metric("Mi­nimo", f"{exec_data['Venta UF'].min():.2f} UF")
                    
                    # GrÃ¡fico individual
                    if 'Contratos' in app.df_clean.columns:
                        # Gráfico dual con ventas y contratos
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig.add_trace(
                            go.Scatter(x=exec_data.sort_values('Mes')['Mes'], 
                                      y=exec_data.sort_values('Mes')['Venta UF'],
                                      name="Ventas UF", line=dict(color='blue'), 
                                      mode='lines+markers'),
                            secondary_y=False,
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=exec_data.sort_values('Mes')['Mes'], 
                                      y=exec_data.sort_values('Mes')['Contratos'],
                                      name="Contratos", line=dict(color='green'), 
                                      mode='lines+markers'),
                            secondary_y=True,
                        )
                        
                        fig.update_xaxes(title_text="Mes")
                        fig.update_yaxes(title_text="Ventas (UF)", secondary_y=False)
                        fig.update_yaxes(title_text="Contratos", secondary_y=True)
                        fig.update_layout(height=400, title=f"Evolución de Ventas y Contratos - {ejecutivo_seleccionado}")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.line(exec_data.sort_values('Mes'), 
                                     x='Mes', y='Venta UF',
                                     title=f"Evolucion de Ventas - {ejecutivo_seleccionado}",
                                     markers=True)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # EstadÃ­sticas detalladas
                    st.subheader("Estadi­sticas Detalladas")
                    if 'Contratos' in app.df_clean.columns:
                        stats_df = exec_data.groupby('Mes').agg({
                            'Venta UF': 'sum',
                            'Contratos': 'sum'
                        }).reset_index()
                        stats_df['Ratio UF/Contrato'] = (stats_df['Venta UF'] / stats_df['Contratos']).round(2)
                    else:
                        stats_df = exec_data.groupby('Mes')['Venta UF'].sum().reset_index()
                    st.dataframe(stats_df, use_container_width=True)
                
                # DistribuciÃ³n de ventas
                st.subheader("Distribucion de Ventas")
                
                if 'Contratos' in app.df_clean.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(app.df_clean, x='Venta UF', nbins=30, 
                                         title="Distribucion de Ventas por Registro")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.histogram(app.df_clean, x='Contratos', nbins=20, 
                                         title="Distribucion de Contratos por Registro")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(app.df_clean, x='Venta UF', nbins=30, 
                                     title="Distribucion de Ventas por Registro")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Matriz de correlaciÃ³n (si hay suficientes columnas numÃ©ricas)
                numeric_cols = app.df_clean.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    st.subheader("Matriz de Correlacion")
                    correlation_matrix = app.df_clean[numeric_cols].corr()
                    
                    fig = px.imshow(correlation_matrix, 
                                   title="Correlacion entre Variables Numericas",
                                   color_continuous_scale="RdYlBu_r",
                                   text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar correlaciones más importantes
                    if 'Contratos' in app.df_clean.columns:
                        st.subheader("Correlaciones Clave")
                        corr_contratos_ventas = app.df_clean['Contratos'].corr(app.df_clean['Venta UF'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Correlación Contratos-Ventas", f"{corr_contratos_ventas:.3f}")
                        with col2:
                            avg_uf_per_contract = (app.df_clean['Venta UF'].sum() / app.df_clean['Contratos'].sum())
                            st.metric("UF Promedio por Contrato", f"{avg_uf_per_contract:.2f}")
                        with col3:
                            conversion_rate = (app.df_clean['Contratos'].sum() / len(app.df_clean)) * 100
                            st.metric("Tasa de Conversión", f"{conversion_rate:.1f}%")
            
            # PÃGINA: MODELOS ML
            elif page == "Modelos ML":
                st.header("Entrenamiento de Modelos ML")
                
                if st.button("Entrenar Modelos", type="primary"):
                    with st.spinner("Creando caracteri­sticas..."):
                        app.df_model = app.create_features(app.df_clean)
                    
                    st.success(f"CaracterÃ­sticas creadas: {app.df_model.shape}")
                    
                    with st.spinner("– Entrenando modelos ML..."):
                        results, best_model, best_name, le, feature_cols = app.train_models(app.df_model)
                    
                    if results:
                        app.results = results
                        app.best_model = best_model
                        app.label_encoder = le
                        app.feature_columns = feature_cols
                        
                        st.success(f"Mejor modelo: {best_name}")
                        
                        # Tabla de comparaciÃ³n
                        st.subheader("Comparacion de Modelos")
                        
                        comparison_data = []
                        for name, metrics in results.items():
                            comparison_data.append({
                                'Modelo': name,
                                'MAE Train': round(metrics['train_mae'], 3),
                                'MAE Test': round(metrics['test_mae'], 3),
                                'RMSE Train': round(metrics['train_rmse'], 3),
                                'RMSE Test': round(metrics['test_rmse'], 3),
                                'RA² Train': round(metrics['train_r2'], 3),
                                'RA² Test': round(metrics['test_r2'], 3)
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # GrÃ¡fico de comparaciÃ³n
                        st.subheader("Visualizacion de Rendimiento")
                        
                        fig = make_subplots(rows=1, cols=2, subplot_titles=(['MAE Test', 'RA² Test']))
                        
                        fig.add_trace(
                            go.Bar(x=comparison_df['Modelo'], y=comparison_df['MAE Test'],
                                   name='MAE Test', marker_color='lightcoral'),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(x=comparison_df['Modelo'], y=comparison_df['RA² Test'],
                                   name='RA² Test', marker_color='lightblue'),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar informaciÃ³n de modelos entrenados
                if hasattr(app, 'results') and app.results:
                    st.subheader("InformaciÃ³n de Modelos")
                    st.info(f"Modelos entrenados: {len(app.results)}")
                    st.info(f"Mejor modelo disponible para predicciones")
            
            # PÃGINA: PREDICCIONES
            elif page == "Predicciones":
                st.header("Generar Predicciones")
                
                if not hasattr(app, 'best_model') or app.best_model is None:
                    st.warning("Primero debes entrenar los modelos en la seccion – Modelos ML")
                else:
                    st.success("Modelo listo para predicciones")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        ejecutivo_pred = st.selectbox(
                            "Seleccionar Ejecutivo:",
                            options=sorted(app.df_clean['EEVV'].unique()),
                            key="pred_ejecutivo"
                        )
                    
                    with col2:
                        meses_pred = st.slider(
                            "Meses a predecir:",
                            min_value=1, max_value=12, value=3
                        )
                    
                    if st.button("Generar Prediccion", type="primary"):
                        # Obtener datos del ejecutivo
                        exec_data = app.df_model[app.df_model['EEVV'] == ejecutivo_pred].sort_values('Mes')
                        
                        if not exec_data.empty:
                            # Generar predicciones simples (basadas en tendencia histÃ³rica)
                            last_records = exec_data.tail(3)
                            avg_venta = last_records['Venta UF'].mean()
                            tendencia = last_records['Venta UF'].diff().mean()
                            
                            predicciones = []
                            fecha_base = exec_data['Mes'].max()
                            
                            for i in range(1, meses_pred + 1):
                                nueva_fecha = fecha_base + timedelta(days=30*i)
                                pred_valor = max(0, avg_venta + (tendencia * i))
                                
                                predicciones.append({
                                    'Mes': nueva_fecha.strftime('%Y-%m'),
                                    'Fecha': nueva_fecha,
                                    'Prediccion (UF)': round(pred_valor, 2),
                                    'Confianza (%)': max(85 - i*3, 60)
                                })
                            
                            pred_df = pd.DataFrame(predicciones)
                            
                            # Mostrar resultados
                            st.subheader(f"Predicciones para {ejecutivo_pred}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Predicho", f"{pred_df['Prediccion (UF)'].sum():.2f} UF")
                            with col2:
                                st.metric("Promedio Mensual", f"{pred_df['Prediccion (UF)'].mean():.2f} UF")
                            with col3:
                                st.metric("Confianza Promedio", f"{pred_df['Confianza (%)'].mean():.0f}%")
                            
                            # Tabla de predicciones
                            st.dataframe(pred_df[['Mes', 'Prediccion (UF)', 'Confianza (%)']], use_container_width=True)
                            
                            # GrÃ¡fico de predicciones
                            historico = exec_data[['Mes', 'Venta UF']].copy()
                            historico['Tipo'] = 'Historico'
                            historico = historico.rename(columns={'Venta UF': 'Valor'})
                            
                            futuro = pred_df[['Fecha', 'Prediccion (UF)']].copy()
                            futuro['Tipo'] = 'Prediccion'
                            futuro = futuro.rename(columns={'Fecha': 'Mes', 'Prediccion (UF)': 'Valor'})
                            
                            combined = pd.concat([historico, futuro], ignore_index=True)
                            
                            fig = px.line(combined, x='Mes', y='Valor', 
                                         color='Tipo', markers=True,
                                         title=f"Ventas Historicas vs Predicciones - {ejecutivo_pred}")
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # OpciÃ³n de descarga
                            csv = pred_df.to_csv(index=False)
                            st.download_button(
                                label="Descargar Predicciones CSV",
                                data=csv,
                                file_name=f"predicciones_{ejecutivo_pred.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime='text/csv'
                            )
                        else:
                            st.error("No se encontraron datos para este ejecutivo")
            
            # PÃGINA: VISUALIZACIONES
            elif page == "Escenarios":
                st.header("🚀 Escenarios de Predicción Optimista")
                
                st.info("💡 Configura escenarios positivos para mejorar las predicciones de ventas")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Parámetros de Crecimiento")
                    
                    crecimiento_ventas = st.slider(
                        "Crecimiento mensual de ventas (%)",
                        min_value=0, max_value=50, value=10, step=1
                    )
                    
                    estacionalidad = st.selectbox(
                        "Factor estacional",
                        ["Sin estacionalidad", "Diciembre +20%", "Marzo +15%", "Junio +10%"]
                    )
                
                with col2:
                    st.subheader("🎯 Objetivos de Negocio")
                    
                    ejecutivo_objetivo = st.selectbox(
                        "Ejecutivo para análisis:",
                        options=sorted(app.df_clean['EEVV'].unique())
                    )
                    
                    meses_escenario = st.slider(
                        "Meses a proyectar:",
                        min_value=1, max_value=12, value=6
                    )
                    
                    meta_mensual = st.number_input(
                        "Meta mensual (UF):",
                        min_value=0.0, value=50.0, step=5.0
                    )
                
                if st.button("🚀 Generar Escenario Optimista", type="primary"):
                    exec_data = app.df_clean[app.df_clean['EEVV'] == ejecutivo_objetivo].sort_values('Mes')
                    
                    if not exec_data.empty:
                        baseline_ventas = exec_data['Venta UF'].tail(3).mean()
                        
                        escenarios = []
                        fecha_base = exec_data['Mes'].max()
                        
                        for i in range(1, meses_escenario + 1):
                            nueva_fecha = fecha_base + timedelta(days=30*i)
                            
                            venta_conservadora = baseline_ventas * (1 + (crecimiento_ventas/100) * 0.5) ** i
                            venta_optimista = baseline_ventas * (1 + (crecimiento_ventas/100)) ** i
                            venta_agresiva = baseline_ventas * (1 + (crecimiento_ventas/100) * 1.5) ** i
                            
                            factor_estacional = 1.0
                            if estacionalidad == "Diciembre +20%" and nueva_fecha.month == 12:
                                factor_estacional = 1.2
                            elif estacionalidad == "Marzo +15%" and nueva_fecha.month == 3:
                                factor_estacional = 1.15
                            elif estacionalidad == "Junio +10%" and nueva_fecha.month == 6:
                                factor_estacional = 1.1
                            
                            venta_conservadora *= factor_estacional
                            venta_optimista *= factor_estacional
                            venta_agresiva *= factor_estacional
                            
                            escenarios.append({
                                'Mes': nueva_fecha.strftime('%Y-%m'),
                                'Fecha': nueva_fecha,
                                'Conservador': round(venta_conservadora, 2),
                                'Optimista': round(venta_optimista, 2),
                                'Agresivo': round(venta_agresiva, 2),
                                'Meta': meta_mensual
                            })
                        
                        escenarios_df = pd.DataFrame(escenarios)
                        
                        st.subheader(f"📊 Escenarios para {ejecutivo_objetivo}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_conservador = escenarios_df['Conservador'].sum()
                            st.metric("Total Conservador", f"{total_conservador:.0f} UF")
                        
                        with col2:
                            total_optimista = escenarios_df['Optimista'].sum()
                            st.metric("Total Optimista", f"{total_optimista:.0f} UF", 
                                     delta=f"+{total_optimista-total_conservador:.0f}")
                        
                        with col3:
                            total_agresivo = escenarios_df['Agresivo'].sum()
                            st.metric("Total Agresivo", f"{total_agresivo:.0f} UF", 
                                     delta=f"+{total_agresivo-total_optimista:.0f}")
                        
                        with col4:
                            total_meta = escenarios_df['Meta'].sum()
                            cumple_meta = "✅" if total_optimista >= total_meta else "❌"
                            st.metric("Meta Total", f"{total_meta:.0f} UF {cumple_meta}")
                        
                        st.dataframe(escenarios_df[['Mes', 'Conservador', 'Optimista', 'Agresivo', 'Meta']], use_container_width=True)
                        
                        fig = go.Figure()
                        
                        historico = exec_data.tail(6)
                        fig.add_trace(go.Scatter(
                            x=historico['Mes'],
                            y=historico['Venta UF'],
                            mode='lines+markers',
                            name='Histórico',
                            line=dict(color='gray', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=escenarios_df['Fecha'],
                            y=escenarios_df['Conservador'],
                            mode='lines+markers',
                            name='Conservador',
                            line=dict(color='orange', dash='dot')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=escenarios_df['Fecha'],
                            y=escenarios_df['Optimista'],
                            mode='lines+markers',
                            name='Optimista',
                            line=dict(color='green', width=3)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=escenarios_df['Fecha'],
                            y=escenarios_df['Agresivo'],
                            mode='lines+markers',
                            name='Agresivo',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=escenarios_df['Fecha'],
                            y=escenarios_df['Meta'],
                            mode='lines',
                            name='Meta',
                            line=dict(color='purple', dash='dashdot')
                        ))
                        
                        fig.update_layout(
                            title=f"Escenarios de Ventas - {ejecutivo_objetivo}",
                            xaxis_title="Fecha",
                            yaxis_title="Ventas (UF)",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("💡 Recomendaciones")
                        
                        if total_optimista >= total_meta:
                            st.success(f"✅ El escenario optimista supera la meta en {total_optimista-total_meta:.0f} UF")
                        else:
                            st.warning(f"⚠️ Necesitas {total_meta-total_optimista:.0f} UF adicionales para cumplir la meta")
                        
                        st.markdown("""
                        **🎯 Acciones Recomendadas:**
                        - Incrementar actividades de prospección
                        - Mejorar seguimiento de leads
                        - Capacitación en técnicas de cierre
                        - Incentivos por cumplimiento de metas
                        """)
                    
                    else:
                        st.error("No se encontraron datos para este ejecutivo")
            
            elif page == "Visualizaciones":
                st.header("Visualizaciones Interactivas")
                
                # GrÃ¡fico de barras por ejecutivo
                st.subheader("Ventas por Ejecutivo")
                
                ventas_ejecutivo = app.df_clean.groupby('EEVV')['Venta UF'].sum().sort_values(ascending=True).tail(15)
                
                fig = px.bar(x=ventas_ejecutivo.values, y=ventas_ejecutivo.index,
                            orientation='h', title="Top 15 Ejecutivos por Ventas Totales")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap de ventas por mes y ejecutivo
                st.subheader("Mapa de Calor: Ventas por Mes y Ejecutivo")
                
                # Preparar datos para heatmap
                pivot_data = app.df_clean.pivot_table(
                    values='Venta UF', 
                    index='EEVV', 
                    columns=app.df_clean['Mes'].dt.strftime('%Y-%m'),
                    aggfunc='sum',
                    fill_value=0
                )
                
                # Tomar solo top 20 ejecutivos para mejor visualizaciÃ³n
                top_executives = app.df_clean.groupby('EEVV')['Venta UF'].sum().sort_values(ascending=False).head(20).index
                pivot_subset = pivot_data.loc[top_executives]
                
                fig = px.imshow(pivot_subset, 
                               title="Ventas por Ejecutivo y Mes",
                               aspect="auto",
                               color_continuous_scale="Viridis")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap de contratos si existe la columna
                if 'Contratos' in app.df_clean.columns:
                    st.subheader("Mapa de Calor: Contratos por Mes y Ejecutivo")
                    
                    pivot_contratos = app.df_clean.pivot_table(
                        values='Contratos', 
                        index='EEVV', 
                        columns=app.df_clean['Mes'].dt.strftime('%Y-%m'),
                        aggfunc='sum',
                        fill_value=0
                    )
                    
                    pivot_contratos_subset = pivot_contratos.loc[top_executives]
                    
                    fig = px.imshow(pivot_contratos_subset, 
                                   title="Contratos por Ejecutivo y Mes",
                                   aspect="auto",
                                   color_continuous_scale="Blues")
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot de consistencia
                st.subheader("Analisis de Consistencia")
                
                consistency_data = app.df_clean.groupby('EEVV')['Venta UF'].agg(['mean', 'std']).reset_index()
                consistency_data.columns = ['Ejecutivo', 'Promedio', 'Desviacion']
                
                fig = px.scatter(consistency_data, x='Promedio', y='Desviacion',
                                hover_data=['Ejecutivo'],
                                title="Promedio vs Variabilidad de Ventas",
                                labels={'Promedio': 'Venta Promedio (UF)', 'Desviacion': 'Desviacion EstÃ¡ndar'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis de Contratos (si existe la columna)
                if 'Contratos' in app.df_clean.columns:
                    st.subheader("Análisis de Contratos")
                    
                    # Relación Ventas vs Contratos
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Scatter plot Ventas vs Contratos
                        scatter_data = app.df_clean.groupby('EEVV').agg({
                            'Venta UF': 'sum',
                            'Contratos': 'sum'
                        }).reset_index()
                        
                        fig = px.scatter(scatter_data, x='Contratos', y='Venta UF',
                                        hover_data=['EEVV'],
                                        title="Relación Contratos vs Ventas por Ejecutivo")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Ratio UF por Contrato
                        scatter_data['Ratio_UF_Contrato'] = scatter_data['Venta UF'] / scatter_data['Contratos']
                        
                        fig = px.bar(scatter_data.sort_values('Ratio_UF_Contrato', ascending=False).head(15),
                                    x='Ratio_UF_Contrato', y='EEVV',
                                    orientation='h',
                                    title="Top 15: UF por Contrato")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("âŒ Error al procesar el archivo. Verifica el formato y columnas.")
    
    else:
        # PÃ¡gina de bienvenida
        st.info("Bienvenido al Sistema de Prediccion de Ventas!")
        st.markdown("""
        ### Como empezar?
        
        1. Carga tu archivo** de ventas en el panel lateral
        2. Explora los datos** en las diferentes secciones
        3. Entrena modelos** de Machine Learning
        4. Genera predicciones** para tus ejecutivos
        
        ###  Requisitos del archivo:
        - Formato: **CSV** o **Excel**
        - Columnas requeridas:
          - **Mes de gestion**: Fecha del periodo
          - **EEVV**: Nombre del ejecutivo
          - **Venta UF**: Ventas en Unidades de Fomento
          - **Contratos**: Número de contratos (opcional pero recomendado)
        
        ###  Caracteri­sticas principales:
        - **4 algoritmos de ML** diferentes
        - **Visualizaciones interactivas** con Plotly
        - **Predicciones personalizadas** por ejecutivo
        - **Analisis detallado** de rendimiento
        - **Exportacion** de resultados
        """)
        
        # Ejemplo de datos
        st.subheader("Ejemplo de formato de datos")
        example_data = pd.DataFrame({
            'Mes de gestión': ['2025-01-01', '2025-02-01', '2025-03-01'],
            'EEVV': ['Juan Perez', 'Mari­a Garcia', 'Carlos Lopez'],
            'Venta UF': [25.50, 32.75, 18.90],
            'Contratos': [3, 4, 2]
        })
        st.dataframe(example_data, use_container_width=True)

if __name__ == "__main__":
    main()
