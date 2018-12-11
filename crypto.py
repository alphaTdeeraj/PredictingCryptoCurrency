#importing necessary libraries 
import pandas as pd 
from keras.layers import Dense ,CuDNNLSTM, Dropout , BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint , TensorBoard
import seaborn as sns 
import os
import numpy as np
import random
from sklearn import preprocessing
from collections import deque


RATIO_TO_PREDICT = 'LTC-USD'
SEQ_LEN = 60
FUTURE_TO_PREDICT = 3
NAMES =['time', 'low', 'high', 'open', 'close', 'volume']
RATIOS = ['BCH-USD', 'BTC-USD','ETH-USD','LTC-USD']
EPOCHS = 10
BATCH_SIZE = 64
MODEL_NAME = f'{SEQ_LEN}_LEN-{FUTURE_TO_PREDICT}_PREDICT_FUTURE-{BATCH_SIZE}_SIZE'
 
main_df = pd.DataFrame()





for ratio in RATIOS:
  df = pd.read_csv('{}.csv'.format(ratio), names=NAMES)
  df.rename(columns={'close':'{}_close'.format(ratio),'volume':'{}_volume'.format(ratio)},inplace=True)
  df = df[['time','{}_close'.format(ratio),'{}_volume'.format(ratio)]]
  if len(main_df)==0:
    main_df = df
  else:
    main_df = pd.merge(main_df , df)
main_df.set_index('time',inplace=True)





#THIS CELL CONTAINS THE FUCNTION WHICH ARE AVALAIBLE 
#function to check whether we can but the crypto or not
def classify(current , future):
  if float(future)>float(current):
    return 1
  else:
    return 0

  
def preprocess(df):
  df = df.drop("future", 1) 
  #next step is normalizing for becoming impartality within the features which is used by the model
  for col in df.columns:
    if not(col=='target'):
      df[col].pct_change()
      df.dropna(inplace=True)
      df[col] = preprocessing.scale(df[col].values)
  df.dropna(inplace=True)
  sequential_data = []
  pre_days = deque(maxlen=SEQ_LEN)
  for sample in df.values:
    pre_days.append([np.array(x) for x in sample[:-1]])
    if len(pre_days)==SEQ_LEN:
      sequential_data.append([np.array(pre_days) ,sample[-1]])
  random.shuffle(sequential_data)
  
  #balancing the data 
  sells = []
  buys  = []
  for data , target in sequential_data:
    if target==1:
      sells.append([data,target])
    else:
      buys.append([data, target])
  random.shuffle(buys)
  random.shuffle(sells)
   
  lower = min(len(sells), len(buys))
  sells = sells[:lower]
  buys = buys[:lower]
  sequential_data = sells + buys 
  random.shuffle(sequential_data)
  X = []
  Y = []
  for seq , target in sequential_data:
    X.append(np.array(seq))
    Y.append(target)
  
  return np.array(X) , Y  
 
 
 
 main_df.fillna(method='ffill', inplace=True)
main_df.dropna(inplace=True)
main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_TO_PREDICT)
main_df['target'] = list(map(classify , main_df['future'] ,main_df[f'{RATIO_TO_PREDICT}_close']))





time = sorted(main_df.index.values)
last_5pct = time[-int(0.05*len(time))]
validation_set = main_df[(main_df.index>=last_5pct)]
training_set = main_df[(main_df.index<last_5pct)]



train_x , train_y = preprocess(training_set)
test_x , test_y = preprocess(validation_set)
train_x = np.array(train_x)
train_y = np.array(test_x)





model = Sequential()


model.add(CuDNNLSTM(128,input_shape=(train_x.shape[1:]),return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())



model.add(CuDNNLSTM(128,input_shape=(train_x.shape[1:]),return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())




model.add(CuDNNLSTM(128,input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))



model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


history = model.fit(test_x ,test_y , epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_x,test_y),callbacks=[tensorboard,checkpoint])
