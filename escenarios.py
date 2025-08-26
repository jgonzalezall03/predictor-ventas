import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show_escenarios_page(app):
    """Página de Escenarios de Predicción Optimista"""
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