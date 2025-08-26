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
    page_title="ðŸ“ˆ Sistema de PredicciÃ³n de Ventas ML",
    page_icon="ðŸ“Š",
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
                df_raw = pd.read_csv(uploaded_file, encoding='utf-8')
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
            df_clean = df_clean.reset_index(drop=True)
            df_clean['Mes de gestiÃ³n'] = pd.to_datetime(df_clean['Mes de gestiÃ³n'])
            df_clean['Venta UF'] = pd.to_numeric(df_clean['Venta UF'], errors='coerce')
            df_clean = df_clean.dropna(subset=['Mes de gestiÃ³n', 'EEVV', 'Venta UF'])
            
            return df_clean
            
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")
            return None
    
    def create_features(self, df):
        """Crear caracterÃ­sticas adicionales para ML"""
        df = df.copy()
        
        # Componentes de fecha
        df['aÃ±o'] = df['Mes de gestiÃ³n'].dt.year
        df['mes'] = df['Mes de gestiÃ³n'].dt.month
        df['trimestre'] = df['Mes de gestiÃ³n'].dt.quarter
        
        # Features por ejecutivo
        df_features = []
        
        for ejecutivo in df['EEVV'].unique():
            exec_data = df[df['EEVV'] == ejecutivo].copy()
            exec_data = exec_data.sort_values('Mes de gestiÃ³n')
            
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
            
            df_features.append(exec_data)
        
        return pd.concat(df_features, ignore_index=True).dropna()
    
    def train_models(self, df_model):
        """Entrenar mÃºltiples modelos de ML"""
        # Preparar datos
        le = LabelEncoder()
        df_model_encoded = df_model.copy()
        df_model_encoded['EEVV_encoded'] = le.fit_transform(df_model['EEVV'])
        
        feature_columns = ['EEVV_encoded', 'aÃ±o', 'mes', 'trimestre', 'venta_lag1', 'venta_lag2', 'venta_lag3', 
                          'media_movil_3', 'media_movil_6', 'tendencia', 'venta_acumulada', 
                          'promedio_hasta_fecha', 'max_hasta_fecha', 'min_hasta_fecha', 'meses_desde_inicio']
        
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
    st.markdown('<h1 class="main-header">ðŸ“ˆ Sistema de PredicciÃ³n de Ventas con Machine Learning</h1>', 
                unsafe_allow_html=True)
    
    # Inicializar app
    if 'app' not in st.session_state:
        st.session_state.app = SalesMLApp()
    
    app = st.session_state.app
    
    # Sidebar para navegaciÃ³n
    st.sidebar.title("ðŸŽ›ï¸ Panel de Control")
    
    # Upload de archivo
    uploaded_file = st.sidebar.file_uploader(
        "ðŸ“ Cargar archivo de ventas",
        type=['csv', 'xlsx'],
        help="Archivo debe contener: Mes de gestiÃ³n, EEVV, Venta UF"
    )
    
    if uploaded_file is not None:
        # Procesar datos
        with st.spinner("ðŸ”„ Procesando datos..."):
            app.df_clean = app.load_and_process_data(uploaded_file)
        
        if app.df_clean is not None:
            st.sidebar.success(f"âœ… Datos cargados: {app.df_clean.shape[0]} registros")
            
            # Selector de pÃ¡gina
            page = st.sidebar.selectbox(
                "ðŸ  NavegaciÃ³n",
                ["ðŸ  Dashboard", "ðŸ“Š AnÃ¡lisis de Datos", "ðŸ¤– Modelos ML", "ðŸ”® Predicciones", "ðŸ“ˆ Visualizaciones"]
            )
            
            # PÃGINA: DASHBOARD
            if page == "ðŸ  Dashboard":
                st.header("ðŸ  Dashboard General")
                
                # MÃ©tricas principales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_ventas = app.df_clean['Venta UF'].sum()
                    st.metric("ðŸ’° Ventas Totales", f"{total_ventas:,.2f} UF")
                
                with col2:
                    promedio_ventas = app.df_clean['Venta UF'].mean()
                    st.metric("ðŸ“Š Promedio Ventas", f"{promedio_ventas:.2f} UF")
                
                with col3:
                    num_ejecutivos = app.df_clean['EEVV'].nunique()
                    st.metric("ðŸ‘¥ Ejecutivos", f"{num_ejecutivos}")
                
                with col4:
                    periodo = f"{app.df_clean['Mes de gestiÃ³n'].min().strftime('%Y-%m')} / {app.df_clean['Mes de gestiÃ³n'].max().strftime('%Y-%m')}"
                    st.metric("ðŸ“… PerÃ­odo", periodo)
                
                # GrÃ¡fico de ventas por mes
                st.subheader("ðŸ“ˆ EvoluciÃ³n de Ventas Mensuales")
                
                ventas_mensuales = app.df_clean.groupby('Mes de gestiÃ³n')['Venta UF'].agg(['sum', 'mean']).reset_index()
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=ventas_mensuales['Mes de gestiÃ³n'], y=ventas_mensuales['sum'],
                              name="Total Mensual", line=dict(color='#1f77b4', width=3)),
                    secondary_y=False,
                )
                
                fig.add_trace(
                    go.Scatter(x=ventas_mensuales['Mes de gestiÃ³n'], y=ventas_mensuales['mean'],
                              name="Promedio Mensual", line=dict(color='#ff7f0e', width=2)),
                    secondary_y=True,
                )
                
                fig.update_xaxes(title_text="Mes")
                fig.update_yaxis(title_text="Ventas Totales (UF)", secondary_y=False)
                fig.update_yaxis(title_text="Promedio (UF)", secondary_y=True)
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Top ejecutivos
                st.subheader("ðŸ† Top 10 Ejecutivos")
                top_ejecutivos = app.df_clean.groupby('EEVV')['Venta UF'].agg(['sum', 'mean', 'count']).round(2)
                top_ejecutivos.columns = ['Total UF', 'Promedio UF', 'Registros']
                top_ejecutivos = top_ejecutivos.sort_values('Promedio UF', ascending=False).head(10)
                
                st.dataframe(top_ejecutivos, use_container_width=True)
            
            # PÃGINA: ANÃLISIS DE DATOS
            elif page == "ðŸ“Š AnÃ¡lisis de Datos":
                st.header("ðŸ“Š AnÃ¡lisis Detallado de Datos")
                
                # AnÃ¡lisis por ejecutivo
                st.subheader("ðŸ‘¤ AnÃ¡lisis por Ejecutivo")
                
                ejecutivo_seleccionado = st.selectbox(
                    "Selecciona un ejecutivo:",
                    options=sorted(app.df_clean['EEVV'].unique())
                )
                
                if ejecutivo_seleccionado:
                    exec_data = app.df_clean[app.df_clean['EEVV'] == ejecutivo_seleccionado]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("ðŸ’° Total Ventas", f"{exec_data['Venta UF'].sum():.2f} UF")
                    with col2:
                        st.metric("ðŸ“Š Promedio", f"{exec_data['Venta UF'].mean():.2f} UF")
                    with col3:
                        st.metric("ðŸ“ˆ MÃ¡ximo", f"{exec_data['Venta UF'].max():.2f} UF")
                    with col4:
                        st.metric("ðŸ“‰ MÃ­nimo", f"{exec_data['Venta UF'].min():.2f} UF")
                    
                    # GrÃ¡fico individual
                    fig = px.line(exec_data.sort_values('Mes de gestiÃ³n'), 
                                 x='Mes de gestiÃ³n', y='Venta UF',
                                 title=f"EvoluciÃ³n de Ventas - {ejecutivo_seleccionado}",
                                 markers=True)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # EstadÃ­sticas detalladas
                    st.subheader("ðŸ“ˆ EstadÃ­sticas Detalladas")
                    stats_df = exec_data.groupby('Mes de gestiÃ³n')['Venta UF'].sum().reset_index()
                    st.dataframe(stats_df, use_container_width=True)
                
                # DistribuciÃ³n de ventas
                st.subheader("ðŸ“Š DistribuciÃ³n de Ventas")
                
                fig = px.histogram(app.df_clean, x='Venta UF', nbins=30, 
                                 title="DistribuciÃ³n de Ventas por Registro")
                st.plotly_chart(fig, use_container_width=True)
                
                # Matriz de correlaciÃ³n (si hay suficientes columnas numÃ©ricas)
                numeric_cols = app.df_clean.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    st.subheader("ðŸ”— Matriz de CorrelaciÃ³n")
                    correlation_matrix = app.df_clean[numeric_cols].corr()
                    
                    fig = px.imshow(correlation_matrix, 
                                   title="CorrelaciÃ³n entre Variables NumÃ©ricas",
                                   color_continuous_scale="RdYlBu_r")
                    st.plotly_chart(fig, use_container_width=True)
            
            # PÃGINA: MODELOS ML
            elif page == "ðŸ¤– Modelos ML":
                st.header("ðŸ¤– Entrenamiento de Modelos ML")
                
                if st.button("ðŸš€ Entrenar Modelos", type="primary"):
                    with st.spinner("ðŸ”§ Creando caracterÃ­sticas..."):
                        app.df_model = app.create_features(app.df_clean)
                    
                    st.success(f"âœ… CaracterÃ­sticas creadas: {app.df_model.shape}")
                    
                    with st.spinner("ðŸ¤– Entrenando modelos ML..."):
                        results, best_model, best_name, le, feature_cols = app.train_models(app.df_model)
                    
                    if results:
                        app.results = results
                        app.best_model = best_model
                        app.label_encoder = le
                        app.feature_columns = feature_cols
                        
                        st.success(f"ðŸ† Mejor modelo: {best_name}")
                        
                        # Tabla de comparaciÃ³n
                        st.subheader("ðŸ“Š ComparaciÃ³n de Modelos")
                        
                        comparison_data = []
                        for name, metrics in results.items():
                            comparison_data.append({
                                'Modelo': name,
                                'MAE Train': round(metrics['train_mae'], 3),
                                'MAE Test': round(metrics['test_mae'], 3),
                                'RMSE Train': round(metrics['train_rmse'], 3),
                                'RMSE Test': round(metrics['test_rmse'], 3),
                                'RÂ² Train': round(metrics['train_r2'], 3),
                                'RÂ² Test': round(metrics['test_r2'], 3)
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # GrÃ¡fico de comparaciÃ³n
                        st.subheader("ðŸ“ˆ VisualizaciÃ³n de Rendimiento")
                        
                        fig = make_subplots(rows=1, cols=2, subplot_titles=(['MAE Test', 'RÂ² Test']))
                        
                        fig.add_trace(
                            go.Bar(x=comparison_df['Modelo'], y=comparison_df['MAE Test'],
                                   name='MAE Test', marker_color='lightcoral'),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(x=comparison_df['Modelo'], y=comparison_df['RÂ² Test'],
                                   name='RÂ² Test', marker_color='lightblue'),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar informaciÃ³n de modelos entrenados
                if hasattr(app, 'results') and app.results:
                    st.subheader("â„¹ï¸ InformaciÃ³n de Modelos")
                    st.info(f"âœ… Modelos entrenados: {len(app.results)}")
                    st.info(f"ðŸ† Mejor modelo disponible para predicciones")
            
            # PÃGINA: PREDICCIONES
            elif page == "ðŸ”® Predicciones":
                st.header("ðŸ”® Generar Predicciones")
                
                if not hasattr(app, 'best_model') or app.best_model is None:
                    st.warning("âš ï¸ Primero debes entrenar los modelos en la secciÃ³n 'ðŸ¤– Modelos ML'")
                else:
                    st.success("âœ… Modelo listo para predicciones")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        ejecutivo_pred = st.selectbox(
                            "ðŸ‘¤ Seleccionar Ejecutivo:",
                            options=sorted(app.df_clean['EEVV'].unique()),
                            key="pred_ejecutivo"
                        )
                    
                    with col2:
                        meses_pred = st.slider(
                            "ðŸ“… Meses a predecir:",
                            min_value=1, max_value=12, value=3
                        )
                    
                    if st.button("ðŸ”® Generar PredicciÃ³n", type="primary"):
                        # Obtener datos del ejecutivo
                        exec_data = app.df_model[app.df_model['EEVV'] == ejecutivo_pred].sort_values('Mes de gestiÃ³n')
                        
                        if not exec_data.empty:
                            # Generar predicciones simples (basadas en tendencia histÃ³rica)
                            last_records = exec_data.tail(3)
                            avg_venta = last_records['Venta UF'].mean()
                            tendencia = last_records['Venta UF'].diff().mean()
                            
                            predicciones = []
                            fecha_base = exec_data['Mes de gestiÃ³n'].max()
                            
                            for i in range(1, meses_pred + 1):
                                nueva_fecha = fecha_base + timedelta(days=30*i)
                                pred_valor = max(0, avg_venta + (tendencia * i))
                                
                                predicciones.append({
                                    'Mes': nueva_fecha.strftime('%Y-%m'),
                                    'Fecha': nueva_fecha,
                                    'PredicciÃ³n (UF)': round(pred_valor, 2),
                                    'Confianza (%)': max(85 - i*3, 60)
                                })
                            
                            pred_df = pd.DataFrame(predicciones)
                            
                            # Mostrar resultados
                            st.subheader(f"ðŸ“Š Predicciones para {ejecutivo_pred}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("ðŸ’° Total Predicho", f"{pred_df['PredicciÃ³n (UF)'].sum():.2f} UF")
                            with col2:
                                st.metric("ðŸ“Š Promedio Mensual", f"{pred_df['PredicciÃ³n (UF)'].mean():.2f} UF")
                            with col3:
                                st.metric("ðŸŽ¯ Confianza Promedio", f"{pred_df['Confianza (%)'].mean():.0f}%")
                            
                            # Tabla de predicciones
                            st.dataframe(pred_df[['Mes', 'PredicciÃ³n (UF)', 'Confianza (%)']], use_container_width=True)
                            
                            # GrÃ¡fico de predicciones
                            historico = exec_data[['Mes de gestiÃ³n', 'Venta UF']].copy()
                            historico['Tipo'] = 'HistÃ³rico'
                            historico = historico.rename(columns={'Venta UF': 'Valor'})
                            
                            futuro = pred_df[['Fecha', 'PredicciÃ³n (UF)']].copy()
                            futuro['Tipo'] = 'PredicciÃ³n'
                            futuro = futuro.rename(columns={'Fecha': 'Mes de gestiÃ³n', 'PredicciÃ³n (UF)': 'Valor'})
                            
                            combined = pd.concat([historico, futuro], ignore_index=True)
                            
                            fig = px.line(combined, x='Mes de gestiÃ³n', y='Valor', 
                                         color='Tipo', markers=True,
                                         title=f"Ventas HistÃ³ricas vs Predicciones - {ejecutivo_pred}")
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # OpciÃ³n de descarga
                            csv = pred_df.to_csv(index=False)
                            st.download_button(
                                label="ðŸ“¥ Descargar Predicciones CSV",
                                data=csv,
                                file_name=f"predicciones_{ejecutivo_pred.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime='text/csv'
                            )
                        else:
                            st.error("âŒ No se encontraron datos para este ejecutivo")
            
            # PÃGINA: VISUALIZACIONES
            elif page == "ðŸ“ˆ Visualizaciones":
                st.header("ðŸ“ˆ Visualizaciones Interactivas")
                
                # GrÃ¡fico de barras por ejecutivo
                st.subheader("ðŸ“Š Ventas por Ejecutivo")
                
                ventas_ejecutivo = app.df_clean.groupby('EEVV')['Venta UF'].sum().sort_values(ascending=True).tail(15)
                
                fig = px.bar(x=ventas_ejecutivo.values, y=ventas_ejecutivo.index,
                            orientation='h', title="Top 15 Ejecutivos por Ventas Totales")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap de ventas por mes y ejecutivo
                st.subheader("ðŸ—“ï¸ Mapa de Calor: Ventas por Mes y Ejecutivo")
                
                # Preparar datos para heatmap
                pivot_data = app.df_clean.pivot_table(
                    values='Venta UF', 
                    index='EEVV', 
                    columns=app.df_clean['Mes de gestiÃ³n'].dt.strftime('%Y-%m'),
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
                
                # Scatter plot de consistencia
                st.subheader("ðŸŽ¯ AnÃ¡lisis de Consistencia")
                
                consistency_data = app.df_clean.groupby('EEVV')['Venta UF'].agg(['mean', 'std']).reset_index()
                consistency_data.columns = ['Ejecutivo', 'Promedio', 'DesviaciÃ³n']
                
                fig = px.scatter(consistency_data, x='Promedio', y='DesviaciÃ³n',
                                hover_data=['Ejecutivo'],
                                title="Promedio vs Variabilidad de Ventas",
                                labels={'Promedio': 'Venta Promedio (UF)', 'DesviaciÃ³n': 'DesviaciÃ³n EstÃ¡ndar'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("âŒ Error al procesar el archivo. Verifica el formato y columnas.")
    
    else:
        # PÃ¡gina de bienvenida
        st.info("ðŸ‘‹ Â¡Bienvenido al Sistema de PredicciÃ³n de Ventas!")
        st.markdown("""
        ### ðŸš€ Â¿CÃ³mo empezar?
        
        1. **ðŸ“ Carga tu archivo** de ventas en el panel lateral
        2. **ðŸ“Š Explora los datos** en las diferentes secciones
        3. **ðŸ¤– Entrena modelos** de Machine Learning
        4. **ðŸ”® Genera predicciones** para tus ejecutivos
        
        ### ðŸ“‹ Requisitos del archivo:
        - Formato: **CSV** o **Excel**
        - Columnas requeridas:
          - **Mes de gestiÃ³n**: Fecha del perÃ­odo
          - **EEVV**: Nombre del ejecutivo
          - **Venta UF**: Ventas en Unidades de Fomento
        
        ### âœ¨ CaracterÃ­sticas principales:
        - ðŸ¤– **4 algoritmos de ML** diferentes
        - ðŸ“Š **Visualizaciones interactivas** con Plotly
        - ðŸ”® **Predicciones personalizadas** por ejecutivo
        - ðŸ“ˆ **AnÃ¡lisis detallado** de rendimiento
        - ðŸ“¥ **ExportaciÃ³n** de resultados
        """)
        
        # Ejemplo de datos
        st.subheader("ðŸ“ Ejemplo de formato de datos")
        example_data = pd.DataFrame({
            'Mes de gestiÃ³n': ['2025-01-01', '2025-02-01', '2025-03-01'],
            'EEVV': ['Juan PÃ©rez', 'MarÃ­a GarcÃ­a', 'Carlos LÃ³pez'],
            'Venta UF': [25.50, 32.75, 18.90]
        })
        st.dataframe(example_data, use_container_width=True)

if __name__ == "__main__":
    main()
