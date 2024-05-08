import numpy as np
import keras 
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from sklearn.preprocessing import StandardScaler,MinMaxScaler

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' #実行したときに”ファイルが重複しています”とエラーならないようにしてる

file_name = input("読み込むファイル名(csv):")+".csv"
nrows = int(input("読み込む時系列のデータ数:"))
encoding = int(input("文字コード(UTF-8:0,shift_jis:1):"))
if encoding == 0:
   encoding = None
else:
   encoding = "932"
df = pd.read_csv(file_name,usecols=[1],nrows=nrows,encoding=encoding).values#１列目の１１１行まで取得
print('type of sequence:',type(df))
print('shape of sequence:',df.shape)

#以下はpandasを使わないでcsvを読み込む方法
#with open('data74227.csv', 'r') as f:
#    reader = csv.reader(f)
#    csv_data = [row for row in reader]
#print(csv_data)
#print(csv_data.shape)

scaler = int(input("スケーリング方法(無し:2正規化:1,標準化:0):"))

if scaler ==0:
   scaler = StandardScaler()
   scaled_df = scaler.fit_transform(df)
elif scaler ==1:
   scaler = MinMaxScaler(feature_range=(0,1))
   scaled_df = scaler.fit_transform(df)
else:
   scaled_df = df

window_size = int(input("ウィンドウサイズを指定:"))
batch_size = len(df)-(window_size+1)


if batch_size%10==0:
   train_size = int(batch_size*0.7)
   valid_size = int(batch_size*0.2 + train_size)
   test_size  = int(batch_size)
else:
   extra = batch_size%10
   re_batch_size = batch_size-extra

   train_size = int(re_batch_size*0.7 + extra)
   valid_size = int(re_batch_size*0.2 + train_size)
   test_size  = int(batch_size)



X_batch = np.zeros((batch_size,window_size,1))
y_batch = np.zeros((batch_size,1,1))
for i in range(0,len(df)-(window_size+1)):
   X_batch[i],y_batch[i] = scaled_df[i:window_size+i],scaled_df[window_size+i]
  
X_train,y_train = X_batch[:train_size,:(window_size+1)],y_batch[:train_size,0]
X_valid,y_valid = X_batch[train_size:valid_size,:(window_size+1)],y_batch[train_size:valid_size,0]
X_test,y_test = X_batch[valid_size:,:(window_size+1)],y_batch[valid_size:,0]

print("window_size:",window_size)
print("batch_size:",batch_size)
print("X_batch.shape:",X_batch.shape)
print("X_train.shape:",X_train.shape)
print("X_valid.shape:",X_valid.shape)
print("X_test.shape:",X_test.shape)

epochs = int(input("epoch数:"))
#LSTM3層
model = keras.models.Sequential([
    keras.layers.LSTM(32,input_shape=[None,1],return_sequences=True),
    keras.layers.LSTM(32,input_shape=[None,1],return_sequences=True),
    keras.layers.LSTM(1,input_shape=[None,1])
])
model.compile(loss="mse",
              optimizer="Adam",
              metrics=["mse"])
history = model.fit(X_train,y_train,epochs=epochs,
                    validation_data=(X_valid,y_valid))
mse_test = model.evaluate(X_test,y_test)

pred_lengh = int(input("予測する長さ:"))
test_number = int(input("予測に使うデータ(数字0~):"))

X_new = X_batch[np.newaxis,0+test_number]
predict = X_new
predict.reshape(window_size)
print(X_new.shape)
print(predict.shape)
for i in range(pred_lengh):
 X_new[0][0:(window_size+1)] = predict[0][i:window_size+i]
 y_pred = model.predict(X_new)
 y_pred = y_pred[...,np.newaxis]
 predict = np.append(predict,y_pred,axis=1)



print("window_size:",window_size)
print("batch_size:",batch_size)
print("epochs:",epochs)
print("pred_lengh:",pred_lengh)
print("X_batch.shape:",X_batch.shape)
print("X_train.shape:",X_train.shape)
print("X_valid.shape:",X_valid.shape)
print("X_test.shape:",X_test.shape)
print("predict_shape",predict.shape)


predict = np.squeeze(predict)
predict = predict[...,np.newaxis]


print("predict.shape_dim2",predict.shape)


ax = pd.DataFrame(scaled_df[(test_number):(test_number)+window_size+pred_lengh]).plot(figsize=(8,5))
pd.DataFrame(predict).plot(figsize=(8,5),ax=ax)
plt.grid(True)
plt.gca().set_ylim(-1,1)
plt.gca()
plt.show()