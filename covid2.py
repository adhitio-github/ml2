
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')
import matplotlib as mpl
import matplotlib.dates as mdate

#import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential   #, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU   #, Bidirectional
#from tensorflow.keras.optimizers import SGD

import math
from datetime import datetime
import datetime
#from datetime import date
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
#import numpy.random as rnd
import pandas_datareader.data as web

#import keras.backend.tensorflow_backend as tb
from tensorflow.keras import backend 
#from tensorflow.python.keras import backend as tb
#tb._SYMBOLIC_SCOPE.value = True

countries=['Indonesia','Afghanistan','Albania','Andorra','Angola','Anguilla','Algeria','Argentina','Armenia','Australia','Barbados',
'Antigua and Barbuda','Aruba','Austria','Bangladesh','Azerbaijan','Bahamas','Bahrain','Belgium','Bolivia','Bosnia and Herzegovina',
'Belize','Benin','Bhutan','Belarus','Bermuda','Brazil','Brunei','Bulgaria','Cambodia','British Virgin Islands','Burkina Faso',
'British Virgin Islands','Canada','Cayman Islands','Chile','Central African Republic','Chad','Cameroon','Cape Verde','China','Colombia',
'Croatia','Cuba','Denmark','Romania','Russia','Rwanda','Czech Republic','Saudi Arabia','Senegal','Serbia','Singapore','Slovakia',
'Slovenia','Somalia','South Africa','South Korea','Spain','Sri Lanka','South Sudan','Switzerland','Taiwan','Tanzania','Thailand','Sudan',
'Suriname','Syria','Sweden','Uganda','Ukraine','United Kingdom','United Arab Emirates','United States','Uruguay','Uzbekistan','Venezuela','Vietnam',
'World']

#def main():
# ----------------------------------------------------------------
#                          D A T A S E T
# ----------------------------------------------------------------

# TExt/Title
st.title("--COVID 19 WEB APPLICATION--") 

#Create sidebar header
st.sidebar.header('User Input')

#Create a function to get user Input
def get_input():

    st.sidebar.markdown('### -----------DATASET-----------')
    option_1 = st.sidebar.selectbox("Country",(countries))
    option_2 = st.sidebar.selectbox("Feature",('new_cases','new_deaths','total_cases','total_deaths'))
    start_date=st.sidebar.text_input("Start Date","2020-03-01")
    split_date=st.sidebar.text_input("Split Date","2020-09-01")
    end_date=st.sidebar.text_input("End Date","2020-09-27") 
    button_1=st.sidebar.button('RUN DATASET')

    st.sidebar.markdown('### -------DEEP LEARNING-------')

    mod=pd.DataFrame(['LSTM-LSTM-LSTM-LSTM',   # baris=0 (model 0)
    'GRU-GRU-GRU-GRU',        # baris=1 (model 1)
    'LSTM-LSTM-GRU-GRU',      # baris=2 (model 2)
    'GRU-GRU-LSTM-LSTM',      # baris=3 (model 3)
    'LSTM-GRU-LSTM-GRU',      # baris=4 (model 4)
    'GRU-LSTM-GRU-LSTM',      # baris=5 (model 5)
    'LSTM-GRU-GRU-LSTM',      # baris=6 (model 6)
    'GRU-LSTM-LSTM-GRU'      # baris=7 (model 7)
    ])

    layer = st.sidebar.selectbox("Arsitetur layer",mod)
    optimasi = st.sidebar.selectbox("Optimizer",['rmsprop' , 'ADAM' , 'Adagrad' , 'Adadelta' , 'Nadam' , 'Adamax' , 'SGD'])
    epoch = st.sidebar.selectbox("Epoch",[5,10,15,20,25,30,35,40,45,50,55])
    button_2=st.sidebar.button('RUN DEEP LEARNING')
    return option_1, option_2, start_date, split_date, end_date,button_1,button_2,layer,optimasi,epoch

#-------------------------------------------------------

#Get the user Input
option_1, option_2, start, split, end, button_1,button_2,layer,optimasi,epoch = get_input()

@st.cache
def load_data(option_1,option_2):

    folder = pd.read_csv("full_data.csv")
    #folder = pd.read_csv("owid-covid-data")
    folder = folder.set_index('date')
    folder.index = pd.to_datetime(folder.index)
    folder = folder.fillna(0)
    #folder_loc1=folder.loc[folder['location']==option_1]  #[['location','date','new_cases','new_deaths']]
    folder_loc2=folder.loc[folder['location']==option_1][[option_2]]

    #data = folder_loc1[option_2]
    data = folder_loc2
    #data = web.DataReader([option_1],'yahoo')[option_2]
    return data

#Create funtion to the proper company data and the proper timeframe from the user
def get_data (option_1,option_2,start_date, split_date, end_date):
    # Get the data range
    start = pd.to_datetime(start_date)
    split = pd.to_datetime(split_date)
    end = pd.to_datetime(end_date)

    #load the data    
    df=load_data(option_1,option_2)        
    #df=df[start:end]
    return df   

#Get the data
df = get_data(option_1, option_2,start, split, end)

def grafik(df,option_1,option_2,start,split,end):

    plt.rc('figure', figsize=(15, 6))
    plt.plot(df[start:split],color='green',label='TrainSet')
    plt.plot(df[split:end],color='blue',label='TestSet')

    plt.legend()
    plt.title(option_1+" "+option_2+" Price")
    plt.xlabel('Date')
    #plt.xticks(rotation=90)
    locator = mdate.MonthLocator()
    plt.gca().xaxis.set_major_locator(locator)

    plt.ylabel(option_2+" Price")
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot() 
    

def tampil_1(option_1,option_2,df,start,split,end):
    #Display the course price
    st.subheader(option_1+" "+option_2+" Price\n")
    grafik(df,option_1,option_2,start,split,end)
    
    #Get statistics on the data
    st.subheader('Data statistics') 
    st.write(df[start:end].describe())

# ----------------------------------------------------------------
#                  D E E P    L E A R N I N G
# ----------------------------------------------------------------

def get_baris_layer(layer):
    if layer == 'LSTM-LSTM-LSTM-LSTM':
        return 0
    elif layer == 'GRU-GRU-GRU-GRU':
        return 1
    elif layer == 'LSTM-LSTM-GRU-GRU':
        return 2
    elif layer == 'GRU-GRU-LSTM-LSTM':
        return 3
    elif layer == 'LSTM-GRU-LSTM-GRU':
        return 4
    elif layer == 'GRU-LSTM-GRU-LSTM':
        return 5
    elif layer == 'LSTM-GRU-GRU-LSTM':
        return 6
    elif layer == 'GRU-LSTM-LSTM-GRU':
        return 7

baris_layer=get_baris_layer(layer)

# ----------------------------------------------------------------
#                     T R A I N N I N G
# ----------------------------------------------------------------

#TRAINNING

def train1(df,start,split,end,option_1,option_2,optimasi,epoch,baris_layer):
    dataset__= df
    tgl2 = split
    tgl3 = pd.date_range(tgl2, periods=2, freq='1D')[-1]
    optimizer=optimasi
    baris=baris_layer

    mod=pd.DataFrame([['LSTM','LSTM','LSTM','LSTM'],   # baris=0 (model 0)
    ['GRU','GRU','GRU','GRU'],        # baris=1 (model 1)
    ['LSTM','LSTM','GRU','GRU'],      # baris=2 (model 2)
    ['GRU','GRU','LSTM','LSTM'],      # baris=3 (model 3)
    ['LSTM','GRU','LSTM','GRU'],      # baris=4 (model 4)
    ['GRU','LSTM','GRU','LSTM'],      # baris=5 (model 5)
    ['LSTM','GRU','GRU','LSTM'],      # baris=6 (model 6)
    ['GRU','LSTM','LSTM','GRU'],      # baris=7 (model 7)
    ])

    j=epoch    
    node=30
    time_step = 3

    training_set = dataset__[start:tgl2].iloc[:,0:2].values
    test_set = dataset__[tgl3:end].iloc[:,0:2].values

    def return_rmse(test,predicted):
        rmse = math.sqrt(mean_squared_error(test, predicted))
        print("The root mean squared error is {}.".format(rmse))

    # Scaling the training set
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []

    for i in range(0,training_set.shape[0]-time_step):
        X_train.append(training_set_scaled[i:i+time_step,0])
        y_train.append(training_set_scaled[i+time_step,0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    # The LSTM architecture
    regressor = Sequential()

    # First LSTM layer with Dropout regularisation
    if mod[0][baris]== 'LSTM':
        regressor.add(LSTM(units=node, return_sequences=True, input_shape=(X_train.shape[1],1)))
    else:
        regressor.add(GRU(units=node, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressor.add(Dropout(0.2))

    # Second LSTM layer
    if mod[1][baris]=='LSTM':
        regressor.add(LSTM(units=node, return_sequences=True))
    else:
        regressor.add(GRU(units=node, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Third LSTM layer
    if mod[2][baris]=='LSTM':
        regressor.add(LSTM(units=node, return_sequences=True))
    else:
        regressor.add(GRU(units=node, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressor.add(Dropout(0.2))

    # Fourth LSTM layer
    if mod[3][baris]=='LSTM':
        regressor.add(LSTM(units=node))
    else:
        regressor.add(GRU(units=node, activation='tanh'))
    regressor.add(Dropout(0.2))
    
    # The output layer
    regressor.add(Dense(units=1))
    # Compiling the RNN
    regressor.compile(optimizer=optimizer,loss='mean_squared_error')

    # Fitting to the training set
    #regressor.fit(X_train,y_train,epochs=50,batch_size=32)    

# ----------------------------------------------------------------
#                         T E S T I N G
# ----------------------------------------------------------------
#def test1(df,start,split,end,option_1,option_2,optimasi,epoch,baris_layer):

    st.write("RUNNING, sabar ya...")
    r=regressor.fit(X_train,y_train,epochs=j,batch_size=64)
    st.write(r)

    dataset_total = pd.concat((dataset__[start:tgl2],dataset__[tgl3:end]),axis=0)
    inputs = dataset_total[len(dataset_total)-len(test_set)-time_step:].values
    inputs = inputs.reshape(-1,1)
    inputs  = sc.transform(inputs)

    # Preparing X_test and predicting the prices
    X_test = []
    for i in range(0,test_set.shape[0]):
        X_test.append(inputs[i:i+time_step,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    rmse = math.sqrt(mean_squared_error(test_set[0:], predicted_stock_price))
    RMSE=format(rmse)
    #st.write('RMSE = ',RMSE)
    test_set2 =[]
    
    predicted_stock_price2=pd.DataFrame(predicted_stock_price)    
    pred_test=dataset__[tgl3:end]
    pred_test.loc[:, ('predict')] = pd.DataFrame(predicted_stock_price)[0]
    pred_test[:]["predict"] = pd.DataFrame(predicted_stock_price)[:][0].values
    #st.write(len(pred_test))
    #st.write(pred_test.head(5))    
    #st.write(pred_test.tail(5))
# ----------------------------------------------------------------
#                          F U T U R E
# ----------------------------------------------------------------
    pred_test_new=[]
    pred_test_new=pred_test
    for i in range (0,50,1):  # prediksi jumlah hari kedepan     
        inputs = pred_test_new[len(pred_test_new)-time_step:][option_2].values
        inputs = inputs.reshape(-1,1)
        inputs  = sc.transform(inputs)
        #display(inputs)
        X_test_Future = []  
        #X_test_Future.append(inputs[pred_test[-time_step:][stock].shape[0]:pred_test[-time_step:][Pilih_].shape[0]+time_step,0])
        X_test_Future.append(inputs)
        X_test_Future = np.array(X_test_Future)
        
        X_test_Future = np.reshape(X_test_Future, (X_test_Future.shape[0],X_test_Future.shape[1],1))
        #X_test_Future = np.reshape(X_test_Future, (time_step,1))
        predicted_stock_price_Future = regressor.predict(X_test_Future)
        predicted_stock_price_Future = sc.inverse_transform(predicted_stock_price_Future)  #+rmse 
        #display(predicted_stock_price_Future)

        end_add = pred_test_new.index[-1]
        add_tgl = pd.date_range(end_add, periods=2, freq='1D')[-1]       
        pred_future=predicted_stock_price_Future[0]
        new_row=[]        
        new_row = pd.DataFrame([[0, pd.DataFrame(pred_future).iloc[0,0]]], columns=[option_2,'predict'], index=[add_tgl])

        #==============TEST========================
        bottom_predict=pred_test_new['predict'][pred_test_new.index[-1]]
        bottom_Pilih =pred_test_new[option_2][pred_test_new.index[-1]]

        #==============TEST========================

        #display(bottom_predict,bottom_Pilih)
        pred_future_=pd.DataFrame(pred_future).iloc[0,0]
        if bottom_predict < bottom_Pilih:
            new_row = pd.DataFrame([[pred_future_+rmse*((len(test_set))**(-0.5)), pred_future_]], columns=[option_2,'predict'], index=[add_tgl])
        else:
            new_row = pd.DataFrame([[pred_future_-rmse*((len(test_set))**(-0.5)), pred_future_]], columns=[option_2,'predict'], index=[add_tgl])
            #pred_future_-rmse*((len(test_set))**(-0.5))
        #display(new_row)
        #==============TEST======================== 
        pred_test_new = pd.concat([pred_test_new[:], pd.DataFrame(new_row)], ignore_index=False)
        pred_test_new.index.name = 'Date'

    #st.write("Selesai trainning.")
    st.write("Plotting....")
    plt.rc('figure', figsize=(10, 6))  
    plt.plot(df[start:tgl2][option_2],color='green',label='TrainSet')
    plt.plot(pred_test[tgl2:end][option_2],color='blue',label='TestSet')
    plt.plot(pred_test_new[tgl2:end]['predict'],color='Red',label='Predict')
    plt.plot(pred_test_new[end:]['predict'],color='m',label='Future Predict')
    cob_lstm = option_2+ " , EPOCH="+str(epoch)+", RMSE ="+str(RMSE)+", LAYER="+mod[0][baris],mod[1][baris],mod[2][baris],mod[3][baris]
    plt.legend()
    plt.title(cob_lstm)
    plt.xlabel('Date')
    plt.ylabel(option_2+" Price")
    #plt.show
    #plt.close
    st.pyplot()

if button_1:
    tampil_1(option_1,option_2,df,start,split,end)
    #st.markdown("## Press RUN DEEP LEARNING button")

    if button_2:
        tampil_1(option_1,option_2,df,start,split,end)
        st.title("Deep Learning")
        train1(df,start,split,end,option_1,option_2,optimasi,epoch,baris_layer)        
    else:
        st.markdown("## Press RUN DEEP LEARNING button")    
else:
    if button_2:
        tampil_1(option_1,option_2,df,start,split,end)
        st.title("Deep Learning")
        train1(df,start,split,end,option_1,option_2,optimasi,epoch,baris_layer)
        #test1(df,start,split,end,option_1,option_2,optimasi,epoch,baris_layer)
    else:
        st.markdown("## Press DATASET button")

#if __name__ == '__main__':
#    main()
