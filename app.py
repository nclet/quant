import streamlit as st
import FinanceDataReader as fdr
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="국내 주식 수익률 분석기", layout="centered")

# 타이틀
st.title("📈 국내 주식 수익률 분석기")
st.markdown("한국 증시에 상장된 종목 중 선택하여 특정 기간의 수익률을 계산하고, 다양한 전략으로 백테스팅합니다.")

# 종목 리스트 가져오기 (KOSPI + KOSDAQ)
@st.cache_data
def get_stock_list():
    kospi = fdr.StockListing('KOSPI')
    kosdaq = fdr.StockListing('KOSDAQ')
    return pd.concat([kospi, kosdaq], ignore_index=True)

stock_list = get_stock_list()

# 종목 선택
selected_name = st.selectbox("종목 선택", stock_list['Name'])
selected_row = stock_list[stock_list['Name'] == selected_name].iloc[0]
code = selected_row['Code']

# 날짜 선택
min_date = datetime.today().replace(year=datetime.today().year - 5)
start_date = st.date_input("시작일", min_value=min_date, max_value=datetime.today(), value=min_date)
end_date = st.date_input("종료일", min_value=start_date, max_value=datetime.today(), value=datetime.today())

# 이동평균선 계산 함수
def calculate_moving_average(df, short_window, long_window):
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
    return df

# 골든크로스/데드크로스 매매 신호 생성 함수
def generate_golden_cross_signals(df):
    df['Signal_GC'] = 0
    df['Buy_GC'] = (df['Short_MA'] > df['Long_MA']) & (df['Short_MA'].shift(1) <= df['Long_MA'].shift(1))
    df['Sell_GC'] = (df['Short_MA'] < df['Long_MA']) & (df['Short_MA'].shift(1) >= df['Long_MA'].shift(1))
    df.loc[df['Buy_GC'], 'Signal_GC'] = 1
    df.loc[df['Sell_GC'], 'Signal_GC'] = -1
    return df

# RSI 계산 함수
def calculate_rsi(df, period=14):
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# RSI 매매 신호 생성 함수
def generate_rsi_signals(df, buy_threshold=30, sell_threshold=70):
    df['Signal_RSI'] = 0
    df['Buy_RSI'] = (df['RSI'] < buy_threshold) & (df['RSI'].shift(1) >= buy_threshold)
    df['Sell_RSI'] = (df['RSI'] > sell_threshold) & (df['RSI'].shift(1) <= sell_threshold)
    df.loc[df['Buy_RSI'], 'Signal_RSI'] = 1
    df.loc[df['Sell_RSI'], 'Signal_RSI'] = -1
    return df

# 볼린저 밴드 계산 함수
def calculate_bollinger_bands(df, window=20, num_std=2):
    df['MA'] = df['Close'].rolling(window=window).mean()
    df['Std'] = df['Close'].rolling(window=window).std()
    df['Upper'] = df['MA'] + (df['Std'] * num_std)
    df['Lower'] = df['MA'] - (df['Std'] * num_std)
    return df

# 볼린저 밴드 매매 신호 생성 함수
def generate_bollinger_signals(df):
    df['Signal_BB'] = 0
    df['Buy_BB'] = (df['Close'] < df['Lower']) & (df['Close'].shift(1) >= df['Lower'].shift(1))
    df['Sell_BB'] = (df['Close'] > df['Upper']) & (df['Close'].shift(1) <= df['Upper'].shift(1))
    df.loc[df['Buy_BB'], 'Signal_BB'] = 1
    df.loc[df['Sell_BB'], 'Signal_BB'] = -1
    return df

# 백테스팅 함수 (전략별 시그널 컬럼을 받아 매매)
def backtest(df, signal_column):
    initial_balance = 1000000  # 초기 투자 금액
    balance = initial_balance
    holdings = 0
    transactions = []

    for i in range(1, len(df)):
        if df[signal_column].iloc[i] == 1 and df[signal_column].iloc[i-1] == 0 and balance > 0:
            buy_price = df['Close'].iloc[i]
            amount_to_buy = balance // buy_price
            if amount_to_buy > 0:
                holdings += amount_to_buy
                balance -= amount_to_buy * buy_price
                transactions.append({'Date': df.index[i], 'Action': 'Buy', 'Price': buy_price, 'Quantity': amount_to_buy})
        elif df[signal_column].iloc[i] == -1 and df[signal_column].iloc[i-1] == 0 and holdings > 0:
            sell_price = df['Close'].iloc[i]
            balance += holdings * sell_price
            transactions.append({'Date': df.index[i], 'Action': 'Sell', 'Price': sell_price, 'Quantity': holdings})
            holdings = 0

    final_value = balance + holdings * df['Close'].iloc[-1]
    return_rate = ((final_value - initial_balance) / initial_balance) * 100
    return return_rate, pd.DataFrame(transactions)

# 백테스팅 활성화 체크박스
run_gc_backtest = st.checkbox("골든크로스/데드크로스 전략 백테스팅 실행")
run_rsi_backtest = st.checkbox("RSI 전략 백테스팅 실행")
run_bb_backtest = st.checkbox("볼린저 밴드 전략 백테스팅 실행")

# 데이터 수집 및 수익률 계산
if st.button("📊 수익률 계산"):
    df = fdr.DataReader(code, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    if df.empty or len(df) < 30: # RSI, 볼린저 밴드 계산 위한 최소 기간 고려
        st.warning("해당 기간에 데이터가 부족하여 백테스팅을 수행할 수 없습니다. 최소 30일 이상의 데이터가 필요합니다.")
    else:
        # 단순 수익률 계산
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        simple_return_rate = ((end_price - start_price) / start_price) * 100
        st.metric(label="📈 단순 기간 수익률", value=f"{simple_return_rate:.2f}%")
        st.line_chart(df['Close'])
        with st.expander("📋 원본 데이터 보기"):
            st.dataframe(df.tail(10))

        # 골든크로스/데드크로스 백테스팅
        if run_gc_backtest:
            if len(df) < 60:
                st.warning("골든크로스/데드크로스 백테스팅을 위해서는 최소 60일 이상의 데이터가 필요합니다.")
            else:
                st.subheader("💰 골든크로스/데드크로스 전략 백테스팅 결과")
                df_ma = calculate_moving_average(df.copy(), short_window=20, long_window=60)
                df_signal_gc = generate_golden_cross_signals(df_ma.copy())
                gc_return_rate, gc_transactions_df = backtest(df_signal_gc.copy(), 'Signal_GC')

                st.metric(label="📈 백테스팅 수익률 (골든크로스/데드크로스)", value=f"{gc_return_rate:.2f}%")
                st.line_chart(df_ma[['Close', 'Short_MA', 'Long_MA']])
                st.caption("종가, 단기 이동평균선(Short_MA), 장기 이동평균선(Long_MA)")
                if not gc_transactions_df.empty:
                    with st.expander("📝 매매 기록 보기 (골든크로스/데드크로스)"):
                        st.dataframe(gc_transactions_df)
                else:
                    st.info("골든크로스/데드크로스 전략 매매 기록이 없습니다.")

        # RSI 백테스팅
        if run_rsi_backtest:
            if len(df) < 30:
                st.warning("RSI 백테스팅을 위해서는 최소 30일 이상의 데이터가 필요합니다.")
            else:
                st.subheader("💰 RSI 전략 백테스팅 결과")
                df_rsi = calculate_rsi(df.copy())
                df_signal_rsi = generate_rsi_signals(df_rsi.copy())
                rsi_return_rate, rsi_transactions_df = backtest(df_signal_rsi.copy(), 'Signal_RSI')

                st.metric(label="📈 백테스팅 수익률 (RSI)", value=f"{rsi_return_rate:.2f}%")
                st.line_chart(df_rsi[['Close', 'RSI']])
                st.caption("종가 및 RSI")
                if not rsi_transactions_df.empty:
                    with st.expander("📝 매매 기록 보기 (RSI)"):
                        st.dataframe(rsi_transactions_df)
                else:
                    st.info("RSI 전략 매매 기록이 없습니다.")

        # 볼린저 밴드 백테스팅
        if run_bb_backtest:
            if len(df) < 30:
                st.warning("볼린저 밴드 백테스팅을 위해서는 최소 30일 이상의 데이터가 필요합니다.")
            else:
                st.subheader("💰 볼린저 밴드 전략 백테스팅 결과")
                df_bb = calculate_bollinger_bands(df.copy())
                df_signal_bb = generate_bollinger_signals(df_bb.copy())
                bb_return_rate, bb_transactions_df = backtest(df_signal_bb.copy(), 'Signal_BB')

                st.metric(label="📈 백테스팅 수익률 (볼린저 밴드)", value=f"{bb_return_rate:.2f}%")
                st.line_chart(df_bb[['Close', 'Upper', 'MA', 'Lower']])
                st.caption("종가 및 볼린저 밴드 (상단, 중심, 하단)")
                if not bb_transactions_df.empty:
                    with st.expander("📝 매매 기록 보기 (볼린저 밴드)"):
                        st.dataframe(bb_transactions_df)
                else:
                    st.info("볼린저 밴드 전략 매매 기록이 없습니다.")
