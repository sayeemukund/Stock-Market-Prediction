import streamlit as st
from streamlit_option_menu import option_menu
import numpy

st.write('hello')

st.set_page_config(page_title="Stock Prediction", page_icon=":chart_with_upwards_trend:")

selected=option_menu(
        menu_title='Stock Market Prediction',
        options= ['Introduction','Procedure'],
        icons=['briefcase-fill','hourglass-split'],
        default_index=0,
        orientation='horizontal',
        styles={
        "container": {"padding": "0!important", "background-color": "grey"},
        "icon": {"color": "black", "font-size": "30px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "white"},
        "nav-link-selected": {"background-color": "#13A88A"},
    }
    )

if selected=='Introduction':
  st.title('Stock Makert Prediction.')
  st.header('Please Note: This app is created just for educational purpose.')
  st.write('-    Stock market prediction using stacked LSTM is a popular application of deep learning in finance..Several LSTM layers make up a stacked LSTM model, a kind of recurrent neural network (RNN). A series of historical stock market data, including daily stock prices, trading volume serves as the input of the model.The output is a forecast of future stock price movement.')
  st.write('-    The stacked LSTM model employs a number of levels of LSTM cells, each layer made up of a number of LSTM units. The first layer of the model receives the input data, while the subsequent layers receive the output of the previous layer as input. When each layer can capture more abstract and high-level representations of the input, the model is able to learn increasingly complicated patterns in the input data.')
  st.write('-    Once trained, the model can be used to make predictions based on fresh, unused data. A series of historical data are used as the input, and its output is a forecast of how the stock price will move in the future. To assess the accuracy, the projected values and the actual values can be compared.')
  st.write('-    In conclusion, a stacked LSTM model is an effective tool for stock market forecasting because it can recognise intricate patterns in historical data and predict future stock price movements with precision. We can create a trustworthy stock market prediction system that can be utilised for investment and trading decisions by training the model on historical data and assessing its performance on a test set.')
  st.write('-    It is crucial to remember that there is no assurance that stock market predictions will be accurate, even if they might be valuable for making wise financial decisions. The stock market is complicated and affected by a variety of unpredictable circumstances, such as world events, political unrest, and natural disasters.')

if selected=='Procedure':
  st.header('Steps Involved.')
  st.write('-    A text input is created for entering your desired stock symbol(PS: CASE SENSITIVE).')
  st.write('-    Data retreived from yfinance library ranging from 1st January 2021 till today.')
  st.write('-    Plotting of graphs which includes:')
  st.write('--    **Candlestick charts**')
  st.write('--    **Line chart to plot the trend in close price.**')
  st.write('--    **Box and Whisker Chart**')
  st.write('--    **Plot to check the trend in differences between close and open price**')
  st.write('-    As the motive of the project is to predict close price for the next 30 days, the close price is split in to training and testing data.')
  st.write('-    The training and testing data is now used to for performing a STACKED LSTM with a timestep ==100.')
  st.write('-    A while loop is created for prediction of the stock price for the next 30 days.')
