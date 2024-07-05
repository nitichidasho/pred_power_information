import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam 
import tensorflow.keras.layers as layers 
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from sklearn.preprocessing import StandardScaler,MinMaxScaler

dtype='f2'#扱うデータ型を、'float16'に指定。
f_model = './model'#モデルの保存先を相対パス指定
f_weights = './weights'#重みの保存先を相対パス指定

# 利用可能な物理デバイスのリストを取得
physical_devices = tf.config.list_physical_devices('GPU')
print("利用可能なGPUデバイス:", physical_devices)

# 利用可能なGPUがあれば、メモリ成長を許可する
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' #実行したときに”ファイルが重複しています”とエラーならないようにしてる

file_name = input("読み込むファイル名(csv):")+".csv"#読み込みたいファイル名
nrows = int(input("読み込む時系列のデータ数:"))#ファイルから読み込みたいデータ数
encoding = int(input("文字コード(UTF-8:0,shift_jis:1):"))#読み込み文字コード指定
if encoding == 0:
   encoding = None
else:
   encoding = "932"
df = pd.read_csv(file_name,usecols=[1],nrows=nrows,encoding=encoding).values#１列目のnrows行まで取得。
print('type of sequence:',type(df))#dfの型を調べる。
print('shape of sequence:',df.shape)#dfのshapeを調べる。
#以下はpandasを使わないでcsvを読み込む方法。
#with open('filename', 'r') as f:
#    reader = csv.reader(f)
#    df = [row for row in reader]
#print(df)
#print(df.shape)

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
batch_size = len(df)-(window_size+1)#(バッチサイズ)＝(全データ)ー(窓数＋1）

if batch_size%10==0:#バッチサイズが10で割り切れる場合。データを次の割合で分割する。訓練:検証:テスト＝7:2:1
   train_size = int(batch_size*0.7)
   valid_size = int(batch_size*0.2 + train_size)
   test_size  = int(batch_size)
else:#バッチサイズが10で割り切れない場合。全データから10の剰余個をひいたデータ群を次の割合で分割する。訓練:検証:テスト＝7:2:1。余った10の余剰個のデータは、訓練データに加える。
   extra = batch_size%10
   re_batch_size = batch_size-extra

   train_size = int(re_batch_size*0.7 + extra)
   valid_size = int(re_batch_size*0.2 + train_size)
   test_size  = int(batch_size)


#入力[バッチサイズ,窓数,1]、教師[バッチサイズ,1,1]。3次元配列。0で初期化。
X_batch = np.zeros((batch_size,window_size,1),dtype=dtype)
y_batch = np.zeros((batch_size,1,1),dtype=dtype)
#窓数分のデータを訓練へ、その次のタイムステップのデータを教師へ格納。1ステップずつスライドし続けて、全データを走査する。
for i in range(0,len(df)-(window_size+1)):
   X_batch[i] = scaled_df[i:window_size+i]
   y_batch[i] = scaled_df[window_size+i]

#データをシャッフルする前に、時系列の開始時間部分を取得しておく。→予測の一回目に使う。
X_first = np.zeros((batch_size,window_size,1),dtype=dtype)
X_first = X_batch.copy()

#データをシャッフル
seed = int(input("シャッフルシード値:"))
np.random.seed(seed=seed)
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
print("X_train_type:",type(X_train))
print("X_train_dtype:",X_train.dtype)
print("X_valid.shape:",X_valid.shape)
print("X_test.shape:",X_test.shape)



load_model = str(input("モデルをロードしますか？[y/n]:"))
if load_model == "y":
 model_filename = str(input("model_filename:"))
 json_string = open(os.path.join(f_model, model_filename)).read()
 model = model_from_json(json_string)
else: 
 #LSTM3層
 model = keras.models.Sequential([
    keras.layers.LSTM(100,input_shape=[None,1],return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
    keras.layers.LSTM(80,input_shape=[None,1],return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
    keras.layers.LSTM(60,input_shape=[None,1],return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
    keras.layers.LSTM(1,input_shape=[None,1]),
    keras.layers.BatchNormalization(),
 ])

load_weights = str(input("重みをロードしますか？[y/n]:"))
if load_weights== "y": 
 weights_filename = str(input("model_filename:"))
 model.load_weights(os.path.join(f_weights,weights_filename))


epochs = int(input("epoch数:"))
model.compile(loss="mse",
              optimizer=Adam(learning_rate=0.01),
              metrics=["mse"])
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
history = model.fit(X_train,y_train,epochs=epochs,
                    validation_data=(X_valid,y_valid),callbacks=[early_stopping_cb])
mse_test = model.evaluate(X_test,y_test)

pred_lengh = int(input("予測する長さ:"))
test_number = int(input("予測に使うデータ(数字0~):"))

X_new = X_first[np.newaxis,test_number]
predict = X_new
print(X_new.shape)
print(predict.shape)
for i in range(pred_lengh):
 X_new[0][0:(window_size+1)] = predict[0][i:window_size+i]
 y_pred = model.predict(X_new)
 y_pred = y_pred[...,np.newaxis]
 predict = np.append(predict,y_pred,axis=1)
 print(i)


model.summary()
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

if epochs!=0:
 plt.plot(history.history['loss'], label='train loss')
 plt.plot(history.history['val_loss'], label='valoss')
 plt.title('Model Loss')
 plt.xlabel('Epoch')
 plt.ylabel('Loss')
 plt.legend()
 plt.grid(True)


ax = pd.DataFrame(scaled_df[(test_number):(test_number)+window_size+pred_lengh]).plot(figsize=(8,5))
pd.DataFrame(predict).plot(figsize=(8,5),ax=ax)
plt.gca().set_ylim(0,1)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Scaled Value')
plt.title('Real & Predicted Sequence')
plt.grid(True)
plt.show()

#モデルのアーキテクチャを保存
save_model = str(input("モデルを保存しますか？[y/n]:"))
if save_model == "y":
 print("<save the architecture of a model>")
 model_filename_save =str(input("モデルの保存名？:"))
 json_string = model.to_json()
 open(os.path.join(f_model,model_filename_save), 'w').write(json_string)
 #モデルをyaml形式で保存する場合
 #　yaml_string = model.to_yaml()
 #　open(os.path.join(f_model,model_filename_save), 'w').write(yaml_string)

#モデルの重みを保存
save_weighs = str(input("重みを保存しますか？[y/n]:"))
if save_weighs == "y":
 print("<save weights>")
 weights_filename_save =str(input("重みの保存名？:"))
 model.save_weights(os.path.join(f_weights,weights_filename_save),save_format='h5')
 #重みをtensorflow形式で保存する方法
 #model.save_weights(os.path.join(f_weights,weights_filename_save),save_format='th')