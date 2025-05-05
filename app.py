####################2025-05-04#######################
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# 페이지 설정
st.set_page_config(page_title="AAPL 수익률 백테스트", layout="centered")
st.title("🍏 Apple(AAPL) 주식 수익률 백테스트")

# 기본 종목 (AAPL 고정)
ticker = "AAPL"

# 사용자에게 기간을 선택하도록
period = st.select_slider(
    '기간을 선택하세요:',
    options=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'],
    value='1y'
)

# 캐시 데이터 함수 (10분 TTL)
@st.cache_data(ttl=600)
def get_stock_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
        return data
    except Exception as e:
        return None

# 데이터 요청
stock_data = get_stock_data(ticker, period)

# 데이터 확인 및 시각화
if stock_data is None or stock_data.empty:
    st.error("❌ 데이터 불러오기 실패. 잠시 후 다시 시도해주세요.")
else:
    st.subheader(f'📊 {ticker} 주가 데이터 ({period})')
    st.dataframe(stock_data.head())

    # 수익률 계산
    initial_price = stock_data['Close'].iloc[0]
    stock_data['Return (%)'] = (stock_data['Close'] / initial_price - 1) * 100

    # 수익률 그래프
    st.subheader("📈 수익률 변화 (%)")
    fig = px.line(stock_data, x=stock_data.index, y='Return (%)', title=f"{ticker} 수익률 추이")
    st.plotly_chart(fig)

    # 최종 수익률 출력
    final_return = stock_data['Return (%)'].iloc[-1]
    st.success(f"📌 총 수익률: {final_return:.2f}%")




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