
###############################################
import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

# 전체 페이지 설정
st.set_page_config(page_title="통합 퀀트 전략 백테스터", layout="wide")

# 사이드바 메뉴
st.sidebar.title("메뉴")
selected_strategy = st.sidebar.radio(
    "원하는 전략을 선택하세요:",
    ["PER/PBR 전략", "기술적 분석 전략", "미래 주가 예측"]
)

st.title("📊 통합 퀀트 전략 백테스터")
st.markdown("PER/PBR 기반 펀더멘털 전략과 기술적 분석 전략을 함께 활용해 백테스팅하고, 미래 주가를 예측할 수 있는 앱입니다.")

# --------------------------------------------
# 함수 정의 (기존 코드와 동일)
# 볼린저 밴드, RSI 계산 등 통합 함수
def calculate_bollinger_bands(df, window=20, num_std=2):
    df['MA'] = df['Close'].rolling(window=window).mean()
    df['Std'] = df['Close'].rolling(window=window).std()
    df['Upper'] = df['MA'] + num_std * df['Std']
    df['Lower'] = df['MA'] - num_std * df['Std']
    return df

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_moving_average(df, short_window, long_window):
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
    return df

def generate_golden_cross_signals(df):
    df['Signal_GC'] = 0
    df['Buy_GC'] = (df['Short_MA'] > df['Long_MA']) & (df['Short_MA'].shift(1) <= df['Long_MA'].shift(1))
    df['Sell_GC'] = (df['Short_MA'] < df['Long_MA']) & (df['Short_MA'].shift(1) >= df['Long_MA'].shift(1))
    df.loc[df['Buy_GC'], 'Signal_GC'] = 1
    df.loc[df['Sell_GC'], 'Signal_GC'] = -1
    return df

def generate_rsi_signals(df, buy_threshold=30, sell_threshold=70):
    df['Signal_RSI'] = 0
    df['Buy_RSI'] = (df['RSI'] < buy_threshold) & (df['RSI'].shift(1) >= buy_threshold)
    df['Sell_RSI'] = (df['RSI'] > sell_threshold) & (df['RSI'].shift(1) <= sell_threshold)
    df.loc[df['Buy_RSI'], 'Signal_RSI'] = 1
    df.loc[df['Sell_RSI'], 'Signal_RSI'] = -1
    return df

def generate_bollinger_signals(df):
    df['Signal_BB'] = 0
    df['Buy_BB'] = (df['Close'] < df['Lower']) & (df['Close'].shift(1) >= df['Lower'].shift(1))
    df['Sell_BB'] = (df['Close'] > df['Upper']) & (df['Close'].shift(1) <= df['Upper'].shift(1))
    df.loc[df['Buy_BB'], 'Signal_BB'] = 1
    df.loc[df['Sell_BB'], 'Signal_BB'] = -1
    return df

def backtest(df, signal_column):
    initial_balance = 1000000
    balance = initial_balance
    holdings = 0
    transactions = []

    for i in range(1, len(df)):
        if df[signal_column].iloc[i] == 1 and df[signal_column].iloc[i-1] == 0 and balance > 0:
            price = df['Close'].iloc[i]
            qty = balance // price
            if qty > 0:
                holdings += qty
                balance -= qty * price
                transactions.append({'Date': df.index[i], 'Action': 'Buy', 'Price': price, 'Qty': qty})
        elif df[signal_column].iloc[i] == -1 and df[signal_column].iloc[i-1] == 0 and holdings > 0:
            price = df['Close'].iloc[i]
            balance += holdings * price
            transactions.append({'Date': df.index[i], 'Action': 'Sell', 'Price': price, 'Qty': holdings})
            holdings = 0

    final_value = balance + holdings * df['Close'].iloc[-1]
    return_rate = (final_value - initial_balance) / initial_balance * 100
    return return_rate, pd.DataFrame(transactions)

def predict_future_price(df, selected_code, selected_name, n_future_days=30):
    import matplotlib.pyplot as plt
    
    def calculate_bollinger_bands_pred(prices, window=20, num_std=2):
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return rolling_mean, upper_band, lower_band

    def calculate_rsi_pred(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def build_model(input_shape):
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def recursive_forecast(model, last_sequence, n_days, scaler, n_features):
        forecasts = []
        current_seq = last_sequence.copy()

        for _ in range(n_days):
            pred = model.predict(current_seq.reshape(1, -1, n_features), verbose=0)[0][0]
            forecasts.append(pred)

            new_feature = np.hstack([pred] * n_features)[:n_features]
            current_seq = np.vstack([current_seq[1:], new_feature])

        forecasts_scaled = scaler.inverse_transform(np.hstack([np.array(forecasts).reshape(-1, 1)] * n_features))[:, 0]
        return forecasts_scaled

    df_stock = df[df['Code'] == selected_code].copy()
    df_stock.sort_values('Date', inplace=True)

    # PER/PBR 데이터가 없는 경우를 대비하여 조건부 추가
    if 'PER' not in df_stock.columns or 'PBR' not in df_stock.columns:
        st.warning("PER/PBR 데이터가 없어 예측에 사용되지 않습니다. 'merged_data_with_per_pbrs.csv' 파일에 PER/PBR 컬럼이 있는지 확인해주세요.")
        df_stock['PER'] = 0.0 # 임시 값 할당 또는 다른 처리
        df_stock['PBR'] = 0.0 # 임시 값 할당 또는 다른 처리

    df_stock['RSI'] = calculate_rsi_pred(df_stock['Close'])
    df_stock['BB_Mid'], df_stock['BB_Upper'], df_stock['BB_Lower'] = calculate_bollinger_bands_pred(df_stock['Close'])
    df_stock.dropna(inplace=True)

    features = ['Close', 'RSI', 'BB_Upper', 'BB_Lower', 'PER', 'PBR']
    target = 'Close'

    # 데이터가 충분한지 확인
    if len(df_stock) < 2 * 20 + 1: # 최소 시퀀스 길이의 두 배 이상
        st.warning(f"데이터가 부족하여 {selected_name}의 미래 주가를 예측할 수 없습니다. 최소 {2 * 20 + 1}일 이상의 데이터가 필요합니다.")
        return


    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_stock[features])
    scaled_target = scaler.fit_transform(df_stock[[target]])

    seq_len = 20
    X, y = [], []
    for i in range(len(scaled_features) - seq_len):
        X.append(scaled_features[i:i+seq_len])
        y.append(scaled_target[i+seq_len])
    
    if not X: # X가 비어있는 경우
        st.warning(f"데이터 전처리 후 남은 데이터가 부족하여 {selected_name}의 미래 주가를 예측할 수 없습니다. 시퀀스 길이를 조절하거나 더 많은 데이터를 확보해주세요.")
        return

    X, y = np.array(X), np.array(y)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model_path = f"model_{selected_code}.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
        st.success("✅ 저장된 모델 로드 완료")
    else:
        model = build_model(input_shape=(X.shape[1], X.shape[2]))
        with st.spinner("🔄 모델 학습 중..."):
            model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test),
                      callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
        model.save(model_path)
        st.success("✅ 모델 학습 및 저장 완료")

    last_sequence = X[-1]
    future_preds = recursive_forecast(model, last_sequence, n_future_days, scaler, X.shape[2])

    last_date = df_stock['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(n_future_days)]

    st.subheader("📊 실제 주가 및 미래 예측 주가")
    fig, ax = plt.subplots()
    ax.plot(df_stock['Date'].iloc[-1500:], df_stock['Close'].iloc[-1500:], label='Actual Price')
    ax.plot(future_dates, future_preds, label='Future Predicted Price')
    ax.axvline(last_date, color='gray', linestyle='--', label='Forecast base date')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📈 예측 수익률")
    returns = (future_preds[-1] - future_preds[0]) / future_preds[0] * 100
    st.metric(label=f"예측 기간 수익률 ({future_dates[0].strftime('%Y-%m-%d')} ~ {future_dates[-1].strftime('%Y-%m-%d')})",
              value=f"{returns:.2f}%")


# --------------------------------------------
# 각 전략에 따른 화면 분기
# --------------------------------------------

if selected_strategy == "PER/PBR 전략":
    st.markdown("---")
    st.header("📊 PER / PBR 기반 수익률 분석")

    per_pbr_file = 'merged_data_with_per_pbrs.csv'

    try:
        df_fundamental = pd.read_csv(per_pbr_file)
        df_fundamental['Date'] = pd.to_datetime(df_fundamental['Date'])
        df_fundamental = df_fundamental.dropna(subset=['PER', 'PBR', 'Close'])
        st.success("PER/PBR 데이터를 성공적으로 불러왔습니다.")

        # 날짜 선택
        per_pbr_start = st.date_input("PER/PBR 시작일", min_value=df_fundamental['Date'].min().date(), max_value=df_fundamental['Date'].max().date(), value=df_fundamental['Date'].min().date())
        per_pbr_end = st.date_input("PER/PBR 종료일", min_value=per_pbr_start, max_value=df_fundamental['Date'].max().date(), value=df_fundamental['Date'].max().date())

        # PER 입력
        st.write("PER 범위 선택")
        col1, col2 = st.columns(2)
        with col1:
            per_min = st.number_input("PER 최소값", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        with col2:
            per_max = st.number_input("PER 최대값", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        st.slider("PER 범위 슬라이더", 0.0, 100.0, (per_min, per_max), disabled=True)
        
        # PBR 입력
        st.write("PBR 범위 선택")
        col3, col4 = st.columns(2)
        with col3:
            pbr_min = st.number_input("PBR 최소값", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        with col4:
            pbr_max = st.number_input("PBR 최대값", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        st.slider("PBR 범위 슬라이더", 0.0, 10.0, (pbr_min, pbr_max), disabled=True)
        
        # 필터링
        df_filtered = df_fundamental[
            (df_fundamental['PER'] >= per_min) & (df_fundamental['PER'] <= per_max) &
            (df_fundamental['PBR'] >= pbr_min) & (df_fundamental['PBR'] <= pbr_max) &
            (df_fundamental['Date'] >= pd.to_datetime(per_pbr_start)) &
            (df_fundamental['Date'] <= pd.to_datetime(per_pbr_end))
        ]

        if df_filtered.empty:
            st.warning("선택한 조건에 해당하는 종목이 없습니다.")
        else:
            df_pivot = df_filtered.pivot_table(index='Date', columns='Code', values='Close')
            df_return = df_pivot.pct_change().fillna(0)
            cumulative_return = (1 + df_return).cumprod() - 1
            final_return = cumulative_return.iloc[-1]
            top_codes = final_return.sort_values(ascending=False).head(10).index
            code_name_map = df_filtered.drop_duplicates('Code').set_index('Code')['Name'].to_dict()
            top_names = [code_name_map.get(code, code) for code in top_codes]

            st.subheader("🏆 수익률 상위 10개 종목")
            st.dataframe(pd.DataFrame({
                '종목코드': top_codes,
                '종목명': top_names,
                '수익률(%)': (final_return[top_codes] * 100).round(2).values
            }).reset_index(drop=True))

            st.line_chart(cumulative_return[top_codes])
    except FileNotFoundError:
        st.error("PER/PBR 데이터 파일이 존재하지 않습니다. 'merged_data_with_per_pbrs.csv' 파일이 현재 디렉토리에 있는지 확인해주세요.")

elif selected_strategy == "기술적 분석 전략":
    st.markdown("---")
    st.header("📌 기술적 분석 기반 전략 백테스팅")

    # 기업 리스트 불러오기
    @st.cache_data
    def get_company_list():
        return pd.read_csv("company_list.csv", dtype={"Code": str})
    
    company_df = get_company_list()
    company_df["label"] = company_df["Name"] + " (" + company_df["Code"] + ")"
    selected_label = st.selectbox("종목 선택", company_df["label"].tolist())
    selected_code = company_df[company_df["label"] == selected_label]["Code"].values[0]

    min_date = datetime.today().replace(year=datetime.today().year - 10)
    start_date = st.date_input("시작일", min_value=min_date, max_value=datetime.today(), value=min_date)
    end_date = st.date_input("종료일", min_value=start_date, max_value=datetime.today(), value=datetime.today())

    run_gc_backtest = st.checkbox("골든크로스/데드크로스 전략")
    run_rsi_backtest = st.checkbox("RSI 전략")
    run_bb_backtest = st.checkbox("볼린저 밴드 전략")

    # 수익률 계산 버튼
    if st.button("📊 수익률 계산"):
        df = fdr.DataReader(selected_code, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if df.empty or len(df) < 30:
            st.warning("데이터가 부족합니다. 최소 30일 이상 필요합니다.")
        else:
            st.metric("📈 단순 수익률", f"{((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100:.2f}%")
            st.line_chart(df['Close'])

            # 골든크로스
            if run_gc_backtest and len(df) >= 60:
                st.subheader("💰 골든크로스 전략")
                df_gc = calculate_moving_average(df.copy(), 20, 60)
                df_gc = generate_golden_cross_signals(df_gc)
                r, log = backtest(df_gc, 'Signal_GC')
                st.metric("수익률", f"{r:.2f}%")
                st.line_chart(df_gc[['Close', 'Short_MA', 'Long_MA']])
                if not log.empty:
                    with st.expander("매매 기록"):
                        st.dataframe(log)

            # RSI
            if run_rsi_backtest:
                st.subheader("💰 RSI 전략")
                df_rsi = calculate_rsi(df.copy())
                df_rsi = generate_rsi_signals(df_rsi)
                r, log = backtest(df_rsi, 'Signal_RSI')
                st.metric("수익률", f"{r:.2f}%")
                st.line_chart(df_rsi[['Close', 'RSI']])
                if not log.empty:
                    with st.expander("매매 기록"):
                        st.dataframe(log)

            # 볼린저밴드
            if run_bb_backtest:
                st.subheader("💰 볼린저 밴드 전략")
                df_bb = calculate_bollinger_bands(df.copy())
                df_bb = generate_bollinger_signals(df_bb)
                r, log = backtest(df_bb, 'Signal_BB')
                st.metric("수익률", f"{r:.2f}%")
                st.line_chart(df_bb[['Close', 'Upper', 'MA', 'Lower']])
                if not log.empty:
                    with st.expander("매매 기록"):
                        st.dataframe(log)

elif selected_strategy == "미래 주가 예측":
    st.markdown("---")
    st.header("🔮 미래 주가 예측 (TensorFlow)")

    # 데이터 로드
    @st.cache_data
    def load_merged_data():
        df = pd.read_csv('merged_data_with_per_pbrs.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df['Code'] = df['Code'].astype(str).str.zfill(6)
        return df

    df_all_data = load_merged_data()

    name_code_dict = df_all_data.drop_duplicates(subset=['Code']).set_index('Name')['Code'].to_dict()
    selected_name = st.selectbox("종목 선택", sorted(name_code_dict.keys()))
    selected_code = name_code_dict[selected_name]

    # 미래 예측 함수 호출
    n_days = st.slider("예측할 미래 일 수", 5, 60, 30)

    if st.button("🚀 주가 예측 시작"):
        predict_future_price(df_all_data, selected_code, selected_name, n_future_days=n_days)