import streamlit as st
from datetime import date
import pandas as pd
from binance.client import Client
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import time

# Даты загрузки данных
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('🔮 Прогноз криптовалют')

# Доступные криптовалюты на Binance (торгуются в USDT)
stocks = {
    'Bitcoin (BTC)': 'BTCUSDT',
    'Ethereum (ETH)': 'ETHUSDT',
    'Binance Coin (BNB)': 'BNBUSDT',
    'Ripple (XRP)': 'XRPUSDT',
    'Cardano (ADA)': 'ADAUSDT',
    'Solana (SOL)':'SOLUSDT'
}
selected_name = st.selectbox('Выберите криптовалюту', list(stocks.keys()))
selected_stock = stocks[selected_name]

# Выбор периода предсказания
n_years = st.slider('Выберите период прогноза (в годах):', 1, 4)
period = n_years * 365

# Инициализация Binance API (публичные данные, без API-ключей)
client = Client()

# Функция загрузки всех данных с Binance
@st.cache_data
def load_data(symbol):
    try:
        all_klines = []
        start_time = int(pd.Timestamp(START).timestamp() * 1000)  # Начало в миллисекундах
        
        while True:
            klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, 
                                       startTime=start_time, limit=1000)  # Максимум 1000 свечей за раз
            
            if not klines:
                break  # Если данных больше нет, выходим из цикла
            
            all_klines.extend(klines)
            start_time = klines[-1][0] + 1  # Устанавливаем новое начало (следующая свеча)
            time.sleep(0.1)  # Небольшая задержка, чтобы не упереться в лимиты Binance
        
        if not all_klines:
            return None
        
        # Создаем DataFrame
        data = pd.DataFrame(all_klines, columns=['Время', 'Открытие', 'Макс.', 'Мин.', 'Закрытие', 'Объём',
                                                 'Закрытие_время', 'Объём_котировки', 'Сделки', 
                                                 'Объём_покупок', 'Объём_покупок_котировки', 'Игнор'])
        
        data['Дата'] = pd.to_datetime(data['Время'], unit='ms')
        data['Открытие'] = data['Открытие'].astype(float)
        data['Закрытие'] = data['Закрытие'].astype(float)
        
        return data[['Дата', 'Открытие', 'Закрытие']]
    except Exception as e:
        st.error(f"Ошибка загрузки данных с Binance: {e}")
        return None

# Загружаем данные
data_load_state = st.text('📥 Загрузка данных...')
data = load_data(selected_stock)

# Проверяем, удалось ли загрузить данные
if data is None or data.empty:
    st.error("❌ Не удалось загрузить данные. Проверьте соединение или тикер.")
    st.stop()

data_load_state.text('✅ Данные успешно загружены!')

st.subheader(f'📊 Исходные данные ({selected_name})')
st.write(data.tail())

# Функция для визуализации данных
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Дата'], y=data['Открытие'], name="Цена открытия", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Дата'], y=data['Закрытие'], name="Цена закрытия", line=dict(color='red')))
    fig.layout.update(
        title='Динамика цен',
        xaxis_title='Дата',
        yaxis_title='Цена (USDT)',
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig)

plot_raw_data()

# Преобразуем данные в формат Prophet
df_train = data[['Дата', 'Закрытие']].dropna().rename(columns={"Дата": "ds", "Закрытие": "y"})

# Обучаем модель Prophet
m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # Добавляем месячную сезонность
m.fit(df_train)

# Генерация будущих дат
future = m.make_future_dataframe(periods=period, freq='D')

# Прогноз
forecast = m.predict(future)

# Отображение прогноза
forecast_ru = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
    'ds': 'Дата',
    'yhat': 'Прогноз',
    'yhat_lower': 'Нижняя граница',
    'yhat_upper': 'Верхняя граница'
})

st.write(forecast_ru.tail())

st.write(f'📅 Прогноз на {n_years} лет ({selected_stock})')
fig1 = plot_plotly(m, forecast)
fig1.update_layout(title='Прогноз цен', xaxis_title='Дата', yaxis_title='Цена (USDT)')
st.plotly_chart(fig1)

st.write("📊 Компоненты прогноза")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)
