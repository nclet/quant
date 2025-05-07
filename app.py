###################2025-05-07######################
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

#######################################################
# import streamlit as st
# import FinanceDataReader as fdr
# from datetime import datetime
# import pandas as pd

# st.set_page_config(page_title="국내 주식 수익률 분석기", layout="centered")

# # 타이틀
# st.title("📈 국내 주식 수익률 분석기")
# st.markdown("한국 증시에 상장된 종목 중 선택하여 특정 기간의 수익률을 계산하고, 골든크로스/데드크로스 전략으로 백테스팅합니다.")

# # 종목 리스트 가져오기 (KOSPI + KOSDAQ)
# @st.cache_data
# def get_stock_list():
#     kospi = fdr.StockListing('KOSPI')
#     kosdaq = fdr.StockListing('KOSDAQ')
#     return pd.concat([kospi, kosdaq], ignore_index=True)

# stock_list = get_stock_list()

# # 종목 선택
# selected_name = st.selectbox("종목 선택", stock_list['Name'])
# selected_row = stock_list[stock_list['Name'] == selected_name].iloc[0]
# code = selected_row['Code']

# # 날짜 선택
# min_date = datetime.today().replace(year=datetime.today().year - 5)
# start_date = st.date_input("시작일", min_value=min_date, max_value=datetime.today(), value=min_date)
# end_date = st.date_input("종료일", min_value=start_date, max_value=datetime.today(), value=datetime.today())

# # 이동평균선 계산 함수
# def calculate_moving_average(df, short_window, long_window):
#     df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
#     df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
#     return df

# # 매매 신호 생성 함수
# def generate_trading_signals(df):
#     df['Signal'] = 0
#     df['Buy'] = (df['Short_MA'] > df['Long_MA']) & (df['Short_MA'].shift(1) <= df['Long_MA'].shift(1))
#     df['Sell'] = (df['Short_MA'] < df['Long_MA']) & (df['Short_MA'].shift(1) >= df['Long_MA'].shift(1))
#     df.loc[df['Buy'], 'Signal'] = 1
#     df.loc[df['Sell'], 'Signal'] = -1
#     return df

# # 백테스팅 함수
# def backtest(df):
#     initial_balance = 1000000  # 초기 투자 금액
#     balance = initial_balance
#     holdings = 0
#     transactions = []

#     for i in range(1, len(df)):
#         if df['Signal'].iloc[i] == 1 and df['Signal'].iloc[i-1] == 0 and balance > 0:
#             buy_price = df['Close'].iloc[i]
#             amount_to_buy = balance // buy_price
#             if amount_to_buy > 0:
#                 holdings += amount_to_buy
#                 balance -= amount_to_buy * buy_price
#                 transactions.append({'Date': df.index[i], 'Action': 'Buy', 'Price': buy_price, 'Quantity': amount_to_buy})
#         elif df['Signal'].iloc[i] == -1 and df['Signal'].iloc[i-1] == 0 and holdings > 0:
#             sell_price = df['Close'].iloc[i]
#             balance += holdings * sell_price
#             transactions.append({'Date': df.index[i], 'Action': 'Sell', 'Price': sell_price, 'Quantity': holdings})
#             holdings = 0

#     final_value = balance + holdings * df['Close'].iloc[-1]
#     return_rate = ((final_value - initial_balance) / initial_balance) * 100
#     return return_rate, pd.DataFrame(transactions)

# # 백테스팅 활성화 체크박스
# run_backtest = st.checkbox("골든크로스/데드크로스 전략 백테스팅 실행")

# # 데이터 수집 및 수익률 계산
# if st.button("📊 수익률 계산"):
#     df = fdr.DataReader(code, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

#     if df.empty or len(df) < 2:
#         st.warning("해당 기간에 데이터가 부족하거나 존재하지 않습니다.")
#     else:
#         # 단순 수익률 계산
#         start_price = df['Close'].iloc[0]
#         end_price = df['Close'].iloc[-1]
#         simple_return_rate = ((end_price - start_price) / start_price) * 100
#         st.metric(label="📈 단순 기간 수익률", value=f"{simple_return_rate:.2f}%")
#         st.line_chart(df['Close'])
#         with st.expander("📋 원본 데이터 보기"):
#             st.dataframe(df.tail(10))

#         # 골든크로스/데드크로스 백테스팅 실행 조건
#         if run_backtest:
#             if len(df) < 60:  # 최소 60일 이상의 데이터 필요 (단기 20일, 장기 60일 기준)
#                 st.warning("골든크로스/데드크로스 백테스팅을 위해서는 최소 60일 이상의 데이터가 필요합니다.")
#             else:
#                 st.subheader("💰 골든크로스/데드크로스 전략 백테스팅 결과")
#                 df_ma = calculate_moving_average(df.copy(), short_window=20, long_window=60) # 20일, 60일 이동평균선 사용
#                 df_signal = generate_trading_signals(df_ma.copy())
#                 backtest_return_rate, transactions_df = backtest(df_signal.copy())

#                 st.metric(label="📈 백테스팅 수익률", value=f"{backtest_return_rate:.2f}%")

#                 st.line_chart(df_ma[['Close', 'Short_MA', 'Long_MA']])
#                 st.caption("종가, 단기 이동평균선(Short_MA), 장기 이동평균선(Long_MA)")

#                 if not transactions_df.empty:
#                     with st.expander("📝 매매 기록 보기"):
#                         st.dataframe(transactions_df)
#                 else:
#                     st.info("매매 기록이 없습니다.")




#####################################################
# import streamlit as st
# import FinanceDataReader as fdr
# from datetime import datetime
# import pandas as pd

# st.set_page_config(page_title="국내 주식 수익률 분석기", layout="centered")

# # 타이틀
# st.title("📈 국내 주식 수익률 분석기")
# st.markdown("한국 증시에 상장된 종목 중 선택하여 특정 기간의 수익률을 계산하고, 골든크로스/데드크로스 전략으로 백테스팅합니다.")

# # 종목 리스트 가져오기 (KOSPI + KOSDAQ)
# @st.cache_data
# def get_stock_list():
#     kospi = fdr.StockListing('KOSPI')
#     kosdaq = fdr.StockListing('KOSDAQ')
#     return pd.concat([kospi, kosdaq], ignore_index=True)

# stock_list = get_stock_list()

# # 종목 선택
# selected_name = st.selectbox("종목 선택", stock_list['Name'])
# selected_row = stock_list[stock_list['Name'] == selected_name].iloc[0]
# code = selected_row['Code']

# # 날짜 선택
# min_date = datetime.today().replace(year=datetime.today().year - 5)
# start_date = st.date_input("시작일", min_value=min_date, max_value=datetime.today(), value=min_date)
# end_date = st.date_input("종료일", min_value=start_date, max_value=datetime.today(), value=datetime.today())

# # 이동평균선 계산 함수
# def calculate_moving_average(df, short_window, long_window):
#     df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
#     df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
#     return df

# # 매매 신호 생성 함수
# def generate_trading_signals(df):
#     df['Signal'] = 0
#     df['Buy'] = (df['Short_MA'] > df['Long_MA']) & (df['Short_MA'].shift(1) <= df['Long_MA'].shift(1))
#     df['Sell'] = (df['Short_MA'] < df['Long_MA']) & (df['Short_MA'].shift(1) >= df['Long_MA'].shift(1))
#     df.loc[df['Buy'], 'Signal'] = 1
#     df.loc[df['Sell'], 'Signal'] = -1
#     return df

# # 백테스팅 함수
# # 백테스팅 함수 수정
# def backtest(df):
#     initial_balance = 1000000  # 초기 투자 금액
#     balance = initial_balance
#     holdings = 0
#     transactions = []

#     for i in range(1, len(df)):
#         if df['Signal'].iloc[i] == 1 and df['Signal'].iloc[i-1] == 0 and balance > 0:
#             buy_price = df['Close'].iloc[i]
#             amount_to_buy = balance // buy_price
#             if amount_to_buy > 0:
#                 holdings += amount_to_buy
#                 balance -= amount_to_buy * buy_price
#                 transactions.append({'Date': df.index[i], 'Action': 'Buy', 'Price': buy_price, 'Quantity': amount_to_buy})
#         elif df['Signal'].iloc[i] == -1 and df['Signal'].iloc[i-1] == 0 and holdings > 0:
#             sell_price = df['Close'].iloc[i]
#             balance += holdings * sell_price
#             transactions.append({'Date': df.index[i], 'Action': 'Sell', 'Price': sell_price, 'Quantity': holdings})
#             holdings = 0

#     final_value = balance + holdings * df['Close'].iloc[-1]
#     return_rate = ((final_value - initial_balance) / initial_balance) * 100
#     return return_rate, pd.DataFrame(transactions) # 명시적으로 두 값을 반환
# # def backtest(df):
# #     initial_balance = 1000000  # 초기 투자 금액
# #     balance = initial_balance
# #     holdings = 0
# #     transactions = []

# #     for i in range(1, len(df)):
# #         if df['Signal'].iloc[i] == 1 and df['Signal'].iloc[i-1] == 0 and balance > 0:
# #             buy_price = df['Close'].iloc[i]
# #             amount_to_buy = balance // buy_price
# #             if amount_to_buy > 0:
# #                 holdings += amount_to_buy
# #                 balance -= amount_to_buy * buy_price
# #                 transactions.append({'Date': df.index[i], 'Action': 'Buy', 'Price': buy_price, 'Quantity': amount_to_buy})
# #         elif df['Signal'].iloc[i] == -1 and df['Signal'].iloc[i-1] == 0 and holdings > 0:
# #             sell_price = df['Close'].iloc[i]
# #             balance += holdings * sell_price
# #             transactions.append({'Date': df.index[i], 'Action': 'Sell', 'Price': sell_price, 'Quantity': holdings})
# #             holdings = 0

# #     final_value = balance + holdings * df['Close'].iloc[-1]
# #     return_rate = ((final_value - initial_balance) / initial_balance) * 100, pd.DataFrame(transactions)

# # 데이터 수집 및 수익률 계산
# if st.button("📊 수익률 계산 및 백테스팅"):
#     df = fdr.DataReader(code, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

#     if df.empty or len(df) < 60:  # 최소 60일 이상의 데이터 필요 (단기 20일, 장기 60일 기준)
#         st.warning("해당 기간에 데이터가 부족하여 백테스팅을 수행할 수 없습니다. 최소 60일 이상의 데이터가 필요합니다.")
#     else:
#         # 단순 수익률 계산
#         start_price = df['Close'].iloc[0]
#         end_price = df['Close'].iloc[-1]
#         simple_return_rate = ((end_price - start_price) / start_price) * 100
#         st.metric(label="📈 단순 기간 수익률", value=f"{simple_return_rate:.2f}%")

#         # 골든크로스/데드크로스 백테스팅
#         df_ma = calculate_moving_average(df.copy(), short_window=20, long_window=60) # 20일, 60일 이동평균선 사용
#         df_signal = generate_trading_signals(df_ma.copy())
#         backtest_return_rate, transactions_df = backtest(df_signal.copy())

#         st.subheader("💰 골든크로스/데드크로스 전략 백테스팅 결과")
#         st.metric(label="📈 백테스팅 수익률", value=f"{backtest_return_rate:.2f}%")

#         st.line_chart(df_ma[['Close', 'Short_MA', 'Long_MA']])
#         st.caption("종가, 단기 이동평균선(Short_MA), 장기 이동평균선(Long_MA)")

#         if not transactions_df.empty:
#             with st.expander("📝 매매 기록 보기"):
#                 st.dataframe(transactions_df)
#         else:
#             st.info("매매 기록이 없습니다.")

#         with st.expander("📋 원본 데이터 보기"):
#             st.dataframe(df.tail(10))



# #########################2025-05-05####################
# import streamlit as st
# import pandas as pd
# import pandas_datareader as pdr
# import datetime
# import plotly.express as px

# st.title('매그니피센트 7 주가 수익률 (지난 1년) - Google Finance 시도')

# # 매그니피센트 7 종목 티커 리스트 (Google Finance 형식에 맞게 조정 필요할 수 있음)
# magnificent_7_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META'] # GOOG로 변경

# # 기간 설정 (지난 1년)
# end_date = datetime.date.today()
# start_date = end_date - datetime.timedelta(days=365)

# @st.cache_data
# def get_stock_data(ticker, start, end, source='google'):
#     try:
#         data = pdr.get_data_yahoo(ticker, start=start, end=end) # 기본은 Yahoo
#         if source == 'google':
#             try:
#                 data = pdr.get_data_google(ticker, start=start, end=end)
#             except Exception as e:
#                 st.error(f'Google Finance에서 {ticker} 데이터 로딩 중 오류 발생: {e}')
#                 return pd.DataFrame()
#         return data
#     except Exception as e:
#         st.error(f'{source}에서 {ticker} 데이터 로딩 중 오류 발생: {e}')
#         return pd.DataFrame()

# # 각 종목별 데이터 다운로드 및 수익률 계산
# stock_data = {}
# for ticker in magnificent_7_tickers:
#     data = get_stock_data(ticker, start_date, end_date, source='google') # Google Finance 사용
#     if not data.empty:
#         initial_price = data['Close'].iloc[0]
#         data['Return'] = (data['Close'] / initial_price - 1) * 100
#         stock_data[ticker] = data
#     else:
#         st.warning(f'{ticker}에 대한 데이터를 찾을 수 없습니다.')

# ... (이하 시각화 코드는 동일) ...

######################################################
# import streamlit as st
# import pandas as pd
# import pandas_datareader as pdr
# import datetime
# import plotly.express as px

# st.title('매그니피센트 7 주가 수익률 (지난 1년)')

# # 매그니피센트 7 종목 티커 리스트
# magnificent_7_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']

# # 기간 설정 (지난 1년)
# end_date = datetime.date.today()
# start_date = end_date - datetime.timedelta(days=365)

# @st.cache_data
# def get_stock_data(ticker, start, end):
#     try:
#         data = pdr.get_data_yahoo(ticker, start=start, end=end)
#         return data
#     except Exception as e:
#         st.error(f'{ticker} 데이터 로딩 중 오류 발생: {e}')
#         return pd.DataFrame()

# # 각 종목별 데이터 다운로드 및 수익률 계산
# stock_data = {}
# for ticker in magnificent_7_tickers:
#     data = get_stock_data(ticker, start_date, end_date)
#     if not data.empty:
#         initial_price = data['Close'].iloc[0]
#         data['Return'] = (data['Close'] / initial_price - 1) * 100
#         stock_data[ticker] = data
#     else:
#         st.warning(f'{ticker}에 대한 데이터를 찾을 수 없습니다.')

# # 수익률 시각화
# st.subheader('지난 1년 간 주가 수익률 (%)')
# all_returns = pd.DataFrame()
# for ticker, data in stock_data.items():
#     if 'Return' in data.columns:
#         all_returns[ticker] = data['Return']

# if not all_returns.empty:
#     fig = px.line(all_returns, x=all_returns.index, y=magnificent_7_tickers,
#                   title='매그니피센트 7 주가 수익률 변화 (지난 1년)')
#     fig.update_layout(yaxis_title='수익률 (%)', xaxis_title='날짜')
#     st.plotly_chart(fig)
# else:
#     st.warning('매그니피센트 7 종목의 주가 수익률 데이터를 불러올 수 없습니다.')

# # 개별 종목 데이터 표시 (선택 사항)
# if st.checkbox('개별 종목 데이터 보기'):
#     selected_ticker = st.selectbox('종목 선택:', magnificent_7_tickers)
#     if selected_ticker in stock_data and not stock_data[selected_ticker].empty:
#         st.subheader(f'{selected_ticker} 주가 데이터 ({start_date.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")})')
#         st.dataframe(stock_data[selected_ticker].head())

#         st.subheader(f'{selected_ticker} 수익률 변화 (지난 1년)')
#         fig_individual = px.line(stock_data[selected_ticker], x=stock_data[selected_ticker].index, y='Return',
#                                  title=f'{selected_ticker} 수익률 (%) (지난 1년)')
#         fig_individual.update_layout(yaxis_title='수익률 (%)', xaxis_title='날짜')
#         st.plotly_chart(fig_individual)
#     elif selected_ticker:
#         st.warning(f'{selected_ticker}에 대한 수익률 데이터가 없습니다.')

################################################
# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.express as px

# st.title('매그니피센트 7 주가 수익률 (지난 1년)')

# # 매그니피센트 7 종목 티커 리스트
# magnificent_7_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']

# @st.cache_data
# def get_stock_data(ticker, period='1y'):
#     data = yf.download(ticker, period=period)
#     return data

# # 각 종목별 데이터 다운로드 및 수익률 계산
# stock_data = {}
# for ticker in magnificent_7_tickers:
#     data = get_stock_data(ticker)
#     if not data.empty:
#         initial_price = data['Close'].iloc[0]
#         data['Return'] = (data['Close'] / initial_price - 1) * 100
#         stock_data[ticker] = data
#     else:
#         st.warning(f'{ticker}에 대한 데이터를 찾을 수 없습니다.')

# # 수익률 시각화
# st.subheader('지난 1년 간 주가 수익률 (%)')
# all_returns = pd.DataFrame()
# for ticker, data in stock_data.items():
#     if 'Return' in data.columns:
#         all_returns[ticker] = data['Return']

# if not all_returns.empty:
#     fig = px.line(all_returns, x=all_returns.index, y=magnificent_7_tickers,
#                   title='매그니피센트 7 주가 수익률 변화 (지난 1년)')
#     fig.update_layout(yaxis_title='수익률 (%)', xaxis_title='날짜')
#     st.plotly_chart(fig)
# else:
#     st.warning('매그니피센트 7 종목의 주가 수익률 데이터를 불러올 수 없습니다.')

# # 개별 종목 데이터 표시 (선택 사항)
# if st.checkbox('개별 종목 데이터 보기'):
#     selected_ticker = st.selectbox('종목 선택:', magnificent_7_tickers)
#     if selected_ticker in stock_data and not stock_data[selected_ticker].empty:
#         st.subheader(f'{selected_ticker} 주가 데이터 (지난 1년)')
#         st.dataframe(stock_data[selected_ticker].head())

#         st.subheader(f'{selected_ticker} 수익률 변화 (지난 1년)')
#         fig_individual = px.line(stock_data[selected_ticker], x=stock_data[selected_ticker].index, y='Return',
#                                  title=f'{selected_ticker} 수익률 (%) (지난 1년)')
#         fig_individual.update_layout(yaxis_title='수익률 (%)', xaxis_title='날짜')
#         st.plotly_chart(fig_individual)
#     elif selected_ticker:
#         st.warning(f'{selected_ticker}에 대한 수익률 데이터가 없습니다.')
        
        

# ####################2025-05-04#######################
# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.express as px

# # 페이지 설정
# st.set_page_config(page_title="AAPL 수익률 백테스트", layout="centered")
# st.title("🍏 Apple(AAPL) 주식 수익률 백테스트")

# # 기본 종목 (AAPL 고정)
# ticker = "AAPL"

# # 사용자에게 기간을 선택하도록
# period = st.select_slider(
#     '기간을 선택하세요:',
#     options=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'],
#     value='1y'
# )

# # 캐시 데이터 함수 (10분 TTL)
# @st.cache_data(ttl=600)
# def get_stock_data(ticker, period):
#     try:
#         data = yf.download(ticker, period=period)
#         return data
#     except Exception as e:
#         return None

# # 데이터 요청
# stock_data = get_stock_data(ticker, period)

# # 데이터 확인 및 시각화
# if stock_data is None or stock_data.empty:
#     st.error("❌ 데이터 불러오기 실패. 잠시 후 다시 시도해주세요.")
# else:
#     st.subheader(f'📊 {ticker} 주가 데이터 ({period})')
#     st.dataframe(stock_data.head())

#     # 수익률 계산
#     initial_price = stock_data['Close'].iloc[0]
#     stock_data['Return (%)'] = (stock_data['Close'] / initial_price - 1) * 100

#     # 수익률 그래프
#     st.subheader("📈 수익률 변화 (%)")
#     fig = px.line(stock_data, x=stock_data.index, y='Return (%)', title=f"{ticker} 수익률 추이")
#     st.plotly_chart(fig)

#     # 최종 수익률 출력
#     final_return = stock_data['Return (%)'].iloc[-1]
#     st.success(f"📌 총 수익률: {final_return:.2f}%")




#####################################
# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.express as px

# st.title("📈 주식 수익률 시각화")

# # 사용자 입력
# ticker = st.text_input('티커 심볼을 입력하세요 (예: AAPL, TSLA, 005930.KS)', 'AAPL').strip().upper()
# period = st.select_slider('기간을 선택하세요:', options=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'], value='1y')

# @st.cache_data(ttl=600)
# def get_stock_data(ticker, period):
#     try:
#         stock = yf.Ticker(ticker)
#         data = stock.history(period=period)
#         return data
#     except Exception as e:
#         st.error(f"❌ 데이터 불러오기 실패: {e}")
#         return pd.DataFrame()

# if ticker:
#     stock_data = get_stock_data(ticker, period)

#     if not stock_data.empty:
#         st.subheader(f'📊 {ticker} 주가 데이터 ({period})')
#         st.dataframe(stock_data.head())

#         # 수익률 계산
#         initial_price = stock_data['Close'].iloc[0]
#         stock_data['Return (%)'] = (stock_data['Close'] / initial_price - 1) * 100

#         # 시각화
#         st.subheader(f'📈 {ticker} 수익률 변화 ({period})')
#         fig = px.line(stock_data, x=stock_data.index, y='Return (%)', title=f'{ticker} 수익률 변화 (%)')
#         st.plotly_chart(fig)
#     else:
#         st.warning(f"📭 '{ticker}'에 대한 데이터를 불러오지 못했습니다. 종목 코드 또는 기간을 확인해주세요.")
# else:
#     st.info("📝 티커 심볼을 입력해주세요.")

########################################
# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.express as px

# st.title("주식 수익률 시각화")

# #사용자로부터 티커 심볼 입력받기
# ticker = st.text_input('티커 심볼을 입력하세요(예: AAPL, TSLA, 005930.KS):', 'AAPL').upper()

# #사용자로부터 기간 선택 받기
# period = st.select_slider('기간을 선택하세요:',  options=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'], value='1y')

# @st.cache_data
# def get_stock_data(ticker, period):
#     data = yf.download(ticker, period=period)
#     return data

# if ticker:
#     stock_data = get_stock_data(ticker, period)
    
#     if not stock_data.empty:
#         st.subheader(f'{ticker} 주가데이터 ({period})')
#         st.dataframe(stock_data.head())
        
#         #수익률 계산
#         initial_price = stock_data['Close'].iloc[0]
#         stock_data['Return'] = (stock_data['Close'] / initial_price - 1) *100
        
#         #수익률 시각화
#         st.subheader(f'{ticker}수익률 변화 ({period})')
#         fig = px.line(stock_data, x=stock_data.index, y='Return', title=f'{ticker} 수익률(%)')
#         st.plotly_chart(fig)
#     else:
#         st.error(f'{ticker}에 대한 데이터를 찾을 수 없습니다.') 
        
# else:
#     st.info('티커 심볼을 입력해주세요')   
    
####################2025-05-04#######################
# import streamlit as st
# import pandas as pd
# import numpy as np

# #간단한 주가 데이터 생성 함수
# def generate_stock_data(start_price=100, num_days=100, volatilty=0.01):
#     dates = pd.date_range(start='2024-01-01', periods=num_days)
#     prices = np.zeros(num_days)
#     prices[0] = start_price
#     for i in range(1, num_days):
#         change = np.random.normal(0, volatilty)
#         prices[i] = prices[i-1] * (1+change)
#     return pd.DataFrame({'Date':dates, 'Price':prices}).set_index('Date')

# #간단한 buy and hold 백테스팅
# def simple_backtest(data, initial_cash=1000000):
#     initial_price = data['Price'].iloc[0]
#     num_shares = initial_cash // initial_price
#     final_value = num_shares * data['Price'].iloc[-1]
#     profit = final_value - initial_cash
#     profit_rate = (profit / initial_cash) * 100
#     return final_value, profit, profit_rate

# # Streamlit 앱 구성
# st.title('간단한 퀀트 백테스팅 웹페이지')

# # 데이터 생성
# stock_data = generate_stock_data()

# # 백테스팅 실행
# final_value, profit, profit_rate = simple_backtest(stock_data)

# # 결과 표시
# st.subheader('백테스팅 결과')
# st.write(f"초기 투자 금액: {1000000:,.0f} 원")
# st.write(f"최종 자산 가치: {final_value:,.0f} 원")
# st.write(f"총 수익: {profit:,.0f} 원")
# st.write(f"수익률: {profit_rate:.2f}%")

# # 주가 데이터 표시
# st.subheader('주가 데이터')
# st.line_chart(stock_data['Price'])