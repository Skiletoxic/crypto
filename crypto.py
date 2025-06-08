import streamlit as st
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from binance.client import Client
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
import time
import warnings
warnings.filterwarnings('ignore')

# Конфигурация страницы
st.set_page_config(
    page_title="🔮 Прогноз криптовалют",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Константы
START_DEFAULT = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# CSS стили
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.success-text {
    color: #00ff00;
    font-weight: bold;
}
.error-text {
    color: #ff0000;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title('🔮 Прогноз криптовалют с расширенной аналитикой')

# Сайдбар с настройками
st.sidebar.header("⚙️ Настройки")

# Расширенный список криптовалют
stocks = {
    'Bitcoin (BTC)': 'BTCUSDT',
    'Ethereum (ETH)': 'ETHUSDT',
    'Binance Coin (BNB)': 'BNBUSDT',
    'Ripple (XRP)': 'XRPUSDT',
    'Cardano (ADA)': 'ADAUSDT',
    'Solana (SOL)': 'SOLUSDT',
    'Polygon (MATIC)': 'MATICUSDT',
    'Avalanche (AVAX)': 'AVAXUSDT',
    'Chainlink (LINK)': 'LINKUSDT',
    'Uniswap (UNI)': 'UNIUSDT',
    'Litecoin (LTC)': 'LTCUSDT',
    'Polkadot (DOT)': 'DOTUSDT'
}

selected_name = st.sidebar.selectbox('🪙 Выберите криптовалюту', list(stocks.keys()))
selected_stock = stocks[selected_name]

# Настройки временного периода
st.sidebar.subheader("📅 Временной период")
start_date = st.sidebar.date_input(
    "Начальная дата",
    value=datetime.strptime(START_DEFAULT, "%Y-%m-%d").date(),
    min_value=datetime(2017, 1, 1).date(),
    max_value=datetime.now().date()
)

# Настройки прогноза
st.sidebar.subheader("🔮 Параметры прогноза")
n_years = st.sidebar.slider('Период прогноза (лет):', 0.1, 4.0, 1.0, 0.1)
period = int(n_years * 365)

# Продвинутые настройки Prophet
st.sidebar.subheader("🧠 Настройки модели")
yearly_seasonality = st.sidebar.checkbox("Годовая сезонность", value=True)
weekly_seasonality = st.sidebar.checkbox("Недельная сезонность", value=True)
daily_seasonality = st.sidebar.checkbox("Дневная сезонность", value=False)
changepoint_prior_scale = st.sidebar.slider("Чувствительность к трендам", 0.001, 0.5, 0.05, 0.001)

# Инициализация Binance API
@st.cache_resource
def init_binance_client():
    try:
        return Client()
    except Exception as e:
        st.error(f"Ошибка инициализации Binance API: {e}")
        return None

client = init_binance_client()

# Улучшенная функция загрузки данных
@st.cache_data(ttl=300)  # Кэш на 5 минут
def load_data(symbol, start_date_str):
    if not client:
        return None
        
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_klines = []
        start_time = int(pd.Timestamp(start_date_str).timestamp() * 1000)
        end_time = int(pd.Timestamp(TODAY).timestamp() * 1000)
        
        total_days = (pd.Timestamp(TODAY) - pd.Timestamp(start_date_str)).days
        processed_days = 0
        
        while start_time < end_time:
            status_text.text(f'📥 Загружено {processed_days}/{total_days} дней...')
            
            klines = client.get_klines(
                symbol=symbol, 
                interval=Client.KLINE_INTERVAL_1DAY,
                startTime=start_time, 
                endTime=end_time,
                limit=1000
            )
            
            if not klines:
                break
            
            all_klines.extend(klines)
            start_time = klines[-1][0] + 1
            processed_days += len(klines)
            
            # Обновляем прогресс бар
            progress = min(processed_days / total_days, 1.0)
            progress_bar.progress(progress)
            
            time.sleep(0.1)  # Ограничение скорости запросов
        
        progress_bar.progress(1.0)
        status_text.text('✅ Данные успешно загружены!')
        
        if not all_klines:
            return None
        
        # Создание DataFrame с улучшенной обработкой
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_asset_volume', 'number_of_trades', 
                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        
        df = pd.DataFrame(all_klines, columns=columns)
        
        # Конвертация типов данных
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Удаление дубликатов и сортировка
        df = df.drop_duplicates(subset=['timestamp']).sort_values('date')
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки данных: {str(e)}")
        return None

# Функция для расчета технических индикаторов
def calculate_technical_indicators(df):
    """Расчет технических индикаторов"""
    df = df.copy()
    
    # Скользящие средние
    df['MA_7'] = df['close'].rolling(window=7).mean()
    df['MA_30'] = df['close'].rolling(window=30).mean()
    df['MA_90'] = df['close'].rolling(window=90).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # Волатильность
    df['volatility'] = df['close'].pct_change().rolling(window=30).std() * np.sqrt(365)
    
    return df

# Функция для анализа качества модели
def model_diagnostics(model, forecast, actual_data):
    """Диагностика качества модели"""
    # Подготовка данных для сравнения
    forecast_comparison = forecast.set_index('ds')
    actual_comparison = actual_data.set_index('date')
    
    # Объединение данных
    comparison = actual_comparison.join(forecast_comparison, how='inner')
    comparison = comparison.dropna()
    
    if len(comparison) == 0:
        return None
    
    # Расчет метрик
    mae = np.mean(np.abs(comparison['close'] - comparison['yhat']))
    mape = np.mean(np.abs((comparison['close'] - comparison['yhat']) / comparison['close'])) * 100
    rmse = np.sqrt(np.mean((comparison['close'] - comparison['yhat']) ** 2))
    
    return {
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse,
        'R²': np.corrcoef(comparison['close'], comparison['yhat'])[0, 1] ** 2
    }

# Основная логика приложения
if client:
    # Загрузка данных
    start_date_str = start_date.strftime("%Y-%m-%d")
    data = load_data(selected_stock, start_date_str)
    
    if data is not None and not data.empty:
        # Расчет технических индикаторов
        data_with_indicators = calculate_technical_indicators(data)
        
        # Создание колонок для макета
        col1, col2, col3, col4 = st.columns(4)
        
        # Метрики
        with col1:
            current_price = data['close'].iloc[-1]
            st.metric("💰 Текущая цена", f"${current_price:,.2f}")
        
        with col2:
            price_change = ((data['close'].iloc[-1] / data['close'].iloc[-2]) - 1) * 100
            st.metric("📈 Изменение за день", f"{price_change:+.2f}%")
        
        with col3:
            max_price = data['close'].max()
            st.metric("🔝 Максимум", f"${max_price:,.2f}")
        
        with col4:
            min_price = data['close'].min()
            st.metric("🔻 Минимум", f"${min_price:,.2f}")
        
        # Табы для различных видов анализа
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Данные", "📈 Технический анализ", "🔮 Прогноз", "🧠 Диагностика", "📋 Отчет"])
        
        with tab1:
            st.subheader(f'📊 Исторические данные ({selected_name})')
            
            # Основной график цены
            fig_main = go.Figure()
            fig_main.add_trace(go.Candlestick(
                x=data['date'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Цена'
            ))
            
            fig_main.update_layout(
                title=f'График цены {selected_name}',
                xaxis_title='Дата',
                yaxis_title='Цена (USDT)',
                height=500
            )
            st.plotly_chart(fig_main, use_container_width=True)
            
            # Таблица данных с возможностью скачивания
            st.subheader("📋 Последние данные")
            st.dataframe(data.tail(10))
            
            # Кнопка для скачивания всех исторических данных
            col1, col2 = st.columns(2)
            with col1:
                csv_historical = data.to_csv(index=False)
                st.download_button(
                    label="📥 Скачать все исторические данные (CSV)",
                    data=csv_historical,
                    file_name=f"{selected_stock}_historical_data_{TODAY}.csv",
                    mime="text/csv",
                    key="download_historical"
                )
            
            with col2:
                csv_technical = data_with_indicators.to_csv(index=False)
                st.download_button(
                    label="📊 Скачать данные с индикаторами (CSV)",
                    data=csv_technical,
                    file_name=f"{selected_stock}_technical_analysis_{TODAY}.csv",
                    mime="text/csv",
                    key="download_technical"
                )
        
        with tab2:
            st.subheader("📈 Технический анализ")
            
            # График с техническими индикаторами
            fig_tech = go.Figure()
            fig_tech.add_trace(go.Scatter(x=data_with_indicators['date'], y=data_with_indicators['close'], 
                                        name='Цена закрытия', line=dict(color='black', width=2)))
            fig_tech.add_trace(go.Scatter(x=data_with_indicators['date'], y=data_with_indicators['MA_7'], 
                                        name='MA 7', line=dict(color='blue')))
            fig_tech.add_trace(go.Scatter(x=data_with_indicators['date'], y=data_with_indicators['MA_30'], 
                                        name='MA 30', line=dict(color='red')))
            fig_tech.add_trace(go.Scatter(x=data_with_indicators['date'], y=data_with_indicators['MA_90'], 
                                        name='MA 90', line=dict(color='green')))
            
            fig_tech.update_layout(title='Цена и скользящие средние', height=400)
            st.plotly_chart(fig_tech, use_container_width=True)
            
            # RSI
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=data_with_indicators['date'], y=data_with_indicators['RSI'], 
                                       name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Перекуплено")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Перепродано")
            fig_rsi.update_layout(title='RSI (Relative Strength Index)', height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with tab3:
            st.subheader(f"🔮 Прогноз на {n_years} лет")
            
            # Подготовка данных для Prophet
            df_train = data[['date', 'close']].dropna().rename(columns={"date": "ds", "close": "y"})
            
            # Создание и обучение модели
            with st.spinner('🧠 Обучение модели...'):
                m = Prophet(
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    changepoint_prior_scale=changepoint_prior_scale,
                    interval_width=0.95
                )
                
                # Добавление дополнительных сезонностей
                if st.sidebar.checkbox("Месячная сезонность", value=True):
                    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                
                m.fit(df_train)
            
            # Создание будущих дат ТОЛЬКО для прогноза (без исторических данных)
            last_date = df_train['ds'].max()
            future_only = pd.DataFrame({
                'ds': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=period, freq='D')
            })
            
            # Прогноз только для будущих дат
            forecast_future = m.predict(future_only)
            
            # Создание улучшенного графика прогноза
            fig_forecast = go.Figure()
            
            # Исторические данные (красная линия)
            fig_forecast.add_trace(go.Scatter(
                x=df_train['ds'],
                y=df_train['y'],
                mode='lines',
                name='Исторические данные',
                line=dict(color='red', width=2)
            ))
            
            # Прогноз (синяя линия)
            fig_forecast.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat'],
                mode='lines',
                name='Прогноз',
                line=dict(color='blue', width=2)
            ))
            
            # Доверительный интервал
            fig_forecast.add_trace(go.Scatter(
                x=pd.concat([forecast_future['ds'], forecast_future['ds'][::-1]]),
                y=pd.concat([forecast_future['yhat_upper'], forecast_future['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,200,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Доверительный интервал',
                showlegend=True
            ))
            
            fig_forecast.update_layout(
                title=f'Прогноз цены {selected_name} (разделение исторических данных и прогноза)',
                xaxis_title='Дата',
                yaxis_title='Цена (USDT)',
                height=600,
                hovermode='x unified'
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Информация о разделении данных
            st.info(f"🔴 **Красная линия**: Исторические данные до {last_date.strftime('%d.%m.%Y')}")
            st.info(f"🔵 **Синяя линия**: Прогноз с {(last_date + pd.Timedelta(days=1)).strftime('%d.%m.%Y')}")
            
            # Компоненты прогноза (используем полный прогноз для анализа компонентов)
            future_full = m.make_future_dataframe(periods=period, freq='D')
            forecast_full = m.predict(future_full)
            
            st.subheader("📊 Компоненты прогноза")
            fig_components = m.plot_components(forecast_full)
            st.pyplot(fig_components)
            
            # Прогноз на конкретные даты
            st.subheader("📅 Прогноз на ключевые даты")
            future_dates = [
                datetime.now() + timedelta(days=30),
                datetime.now() + timedelta(days=90),
                datetime.now() + timedelta(days=365)
            ]
            
            for future_date in future_dates:
                if future_date.strftime('%Y-%m-%d') in forecast_future['ds'].dt.strftime('%Y-%m-%d').values:
                    forecast_row = forecast_future[forecast_future['ds'].dt.strftime('%Y-%m-%d') == future_date.strftime('%Y-%m-%d')]
                    if not forecast_row.empty:
                        predicted_price = forecast_row['yhat'].iloc[0]
                        lower_bound = forecast_row['yhat_lower'].iloc[0]
                        upper_bound = forecast_row['yhat_upper'].iloc[0]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"📅 {future_date.strftime('%d.%m.%Y')}", f"${predicted_price:.2f}")
                        with col2:
                            st.metric("📉 Нижняя граница", f"${lower_bound:.2f}")
                        with col3:
                            st.metric("📈 Верхняя граница", f"${upper_bound:.2f}")
            
            # Возможность скачать прогноз
            st.subheader("💾 Скачать прогноз")
            col1, col2 = st.columns(2)
            with col1:
                csv_forecast = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                    'ds': 'Дата',
                    'yhat': 'Прогноз',
                    'yhat_lower': 'Нижняя_граница',
                    'yhat_upper': 'Верхняя_граница'
                }).to_csv(index=False)
                st.download_button(
                    label="📥 Скачать прогноз (CSV)",
                    data=csv_forecast,
                    file_name=f"{selected_stock}_forecast_{n_years}y_{TODAY}.csv",
                    mime="text/csv",
                    key="download_forecast"
                )
            
            with col2:
                # Объединенные данные (исторические + прогноз)
                combined_data = pd.concat([
                    df_train.rename(columns={'ds': 'Дата', 'y': 'Цена'}).assign(Тип='Исторические'),
                    forecast_future[['ds', 'yhat']].rename(columns={'ds': 'Дата', 'yhat': 'Цена'}).assign(Тип='Прогноз')
                ])
                csv_combined = combined_data.to_csv(index=False)
                st.download_button(
                    label="📊 Скачать объединенные данные (CSV)",
                    data=csv_combined,
                    file_name=f"{selected_stock}_combined_{TODAY}.csv",
                    mime="text/csv",
                    key="download_combined"
                )
        
        with tab4:
            st.subheader("🧠 Диагностика модели")
            
            # Анализ качества модели (используем полный прогноз для диагностики)
            future_full = m.make_future_dataframe(periods=period, freq='D')
            forecast_full = m.predict(future_full)
            diagnostics = model_diagnostics(m, forecast_full, data)
            
            if diagnostics:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{diagnostics['MAE']:.2f}")
                with col2:
                    st.metric("MAPE", f"{diagnostics['MAPE']:.2f}%")
                with col3:
                    st.metric("RMSE", f"{diagnostics['RMSE']:.2f}")
                with col4:
                    st.metric("R²", f"{diagnostics['R²']:.3f}")
                
                # Интерпретация результатов
                if diagnostics['MAPE'] < 10:
                    st.success("✅ Отличная точность прогноза (MAPE < 10%)")
                elif diagnostics['MAPE'] < 20:
                    st.warning("⚠️ Хорошая точность прогноза (MAPE < 20%)")
                else:
                    st.error("❌ Низкая точность прогноза (MAPE > 20%)")
            
            # График остатков
            if 'yhat' in forecast_full.columns and len(data) > 0:
                residuals_data = forecast_full[forecast_full['ds'].isin(data['date'])].copy()
                residuals_data = residuals_data.merge(
                    data[['date', 'close']], 
                    left_on='ds', 
                    right_on='date', 
                    how='inner'
                )
                
                if not residuals_data.empty:
                    residuals = residuals_data['close'] - residuals_data['yhat']
                    
                    fig_residuals = go.Figure()
                    fig_residuals.add_trace(go.Scatter(
                        x=residuals_data['ds'], 
                        y=residuals,
                        mode='markers',
                        name='Остатки',
                        marker=dict(color='red', size=4)
                    ))
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="blue")
                    fig_residuals.update_layout(
                        title='График остатков (разность между фактом и прогнозом)',
                        xaxis_title='Дата',
                        yaxis_title='Остатки'
                    )
                    st.plotly_chart(fig_residuals, use_container_width=True)
        
        with tab5:
            st.subheader("📋 Сводный отчет")
            
            # Общая информация
            st.write(f"**Криптовалюта:** {selected_name}")
            st.write(f"**Период данных:** {start_date} - {TODAY}")
            st.write(f"**Всего дней:** {len(data)}")
            st.write(f"**Период прогноза:** {n_years} лет ({period} дней)")
            
            # Статистика
            stats = {
                'Средняя цена': data['close'].mean(),
                'Медианная цена': data['close'].median(),
                'Стандартное отклонение': data['close'].std(),
                'Максимальная цена': data['close'].max(),
                'Минимальная цена': data['close'].min(),
                'Средний объем торгов': data['volume'].mean() if 'volume' in data.columns else 'N/A'
            }
            
            stats_df = pd.DataFrame(list(stats.items()), columns=['Метрика', 'Значение'])
            st.table(stats_df)
            
            # Рекомендации
            st.subheader("💡 Рекомендации")
            
            current_rsi = data_with_indicators['RSI'].iloc[-1] if not pd.isna(data_with_indicators['RSI'].iloc[-1]) else None
            
            if current_rsi:
                if current_rsi > 70:
                    st.warning("⚠️ RSI показывает перекупленность. Возможна коррекция цены.")
                elif current_rsi < 30:
                    st.success("✅ RSI показывает перепроданность. Возможен рост цены.")
                else:
                    st.info("ℹ️ RSI в нейтральной зоне.")
            
            # Экспорт данных - перенесено в соответствующие табы
            st.subheader("💡 Дополнительная информация")
            st.info("📊 **Все данные можно скачать в соответствующих разделах:**")
            st.info("• Исторические данные - вкладка 'Данные'")
            st.info("• Прогнозы - вкладка 'Прогноз'")
            st.info("• Технические индикаторы - вкладка 'Данные'")
    
    else:
        st.error("❌ Не удалось загрузить данные. Проверьте соединение с интернетом.")
else:
    st.error("❌ Не удалось подключиться к Binance API.")

# Футер
st.markdown("---")
st.markdown("*Данный прогноз предназначен только для образовательных целей и не является инвестиционной рекомендацией.*  Создатель: магистрант ИСМ23-2.*")