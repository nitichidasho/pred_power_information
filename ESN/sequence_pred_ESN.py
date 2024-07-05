import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import model
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



X_batch = np.zeros((batch_size,window_size,1),dtype=np.float32)
y_batch = np.zeros((batch_size,1,1),dtype=np.float32)
for i in range(0,len(df)-(window_size+1)):
   X_batch[i],y_batch[i] = scaled_df[i:window_size+i],scaled_df[window_size+i]

#シャッフル
rng = np.random.default_rng()
rng.shuffle(X_batch, axis=0)
rng.shuffle(y_batch, axis=0)

X_train,y_train = X_batch[:train_size,:(window_size+1)],y_batch[:train_size,0]
X_valid,y_valid = X_batch[train_size:valid_size,:(window_size+1)],y_batch[train_size:valid_size,0]
X_test,y_test = X_batch[valid_size:,:(window_size+1)],y_batch[valid_size:,0]

print("window_size:",window_size)
print("batch_size:",batch_size)
print("X_batch.shape:",X_batch.shape)
print("X_train.shape:",X_train.shape)
print("X_valid.shape:",X_valid.shape)
print("X_test.shape:",X_test.shape)

#Run ESN
ESN = model.EchoStateNetwork(units=300,density=0.01,leaking_rate=0.9)
ESN.fit(X_train, y_train,beta=0.07)
print("Train MSE:",ESN.MSE_Score(X_train, y_train))
print("Test MSE:",ESN.MSE_Score(X_test, y_test))
y_test_hat = ESN.predict(X_test)

print(y_test_hat.shape)

pred_lengh = int(input("予測する長さ:"))
test_number = int(input("予測に使うデータ(数字0~):"))

X_new = X_batch[np.newaxis,0+test_number]
predict = X_new
predict.reshape(window_size)
print(X_new.shape)
print(predict.shape)
for i in range(pred_lengh):
 X_new[0][0:(window_size+1)] = predict[0][i:window_size+i]
 y_pred = ESN.predict(X_new)
 y_pred = y_pred[...,np.newaxis]
 predict = np.append(predict,y_pred,axis=1)



print("window_size:",window_size)
print("batch_size:",batch_size)
#print("epochs:",epochs)
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
plt.show()
