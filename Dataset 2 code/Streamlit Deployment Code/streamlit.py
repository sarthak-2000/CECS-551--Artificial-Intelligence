import statsmodels.api as sm
import streamlit as st
import pandas as pd
import matplotlib
from matplotlib import pyplot
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)


#Loading up the Regression model we created
#model = sm.tsa.arima.ARIMA()
#model.load('model_arima.json')
food_series_df = pd.read_csv("food_series_df.csv")
food_series_df = food_series_df['FOODS']
movingAverage = food_series_df.rolling(window=30).mean()
movingAverage.fillna(0)

food_derived = food_series_df-movingAverage
food_derived.fillna(food_derived.mean(),inplace=True)

model = sm.tsa.arima.ARIMA(food_derived,order=(2,2,0))
model_ARIMA = model.fit()

shfited = pd.DataFrame({'predicShfited2':pd.Series(model_ARIMA.fittedvalues,copy=True),'day':food_series_df.index[:1969]})
shfited = shfited.set_index('day')

predictVsActual = pd.DataFrame({'actual':food_series_df,
                                'predictDiff':shfited['predicShfited2'],
                                'base':movingAverage})

predictVsActual['predict'] = predictVsActual.loc[:,['predictDiff','base']].sum(axis=1)

#Caching the model for faster loading
@st.cache(suppress_st_warning=True)



def display_graph(s,e):
    if e <=s:
        e = s+1
    pyplot.plot(predictVsActual['predict'].iloc[s:e],label='Predicted Foods Sale')
    pyplot.legend()
    pyplot.title("Prediction using ARIMA for selected {} days".format(e-s))
    st.pyplot()



st.title('Foods Sales Forecasting')
st.image("""https://www.eatthis.com/wp-content/uploads/sites/4/2021/06/walmart-shopping.jpg?quality=82&strip=1""")
st.header('Enter forecasting days:')


s = st.number_input('Enter start day for prediction ', min_value=0, max_value=1960, value=1)
e = st.number_input('Enter end day for prediction less than 1969', min_value=0, max_value=1969, value=1)

if st.button('Predict'):
    display_graph(s,e)
