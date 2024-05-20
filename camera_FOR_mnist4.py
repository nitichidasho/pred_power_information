import cv2                  #CV2(OpenCV)：画像や動画を扱うpythonライブラリ
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle               #学習後のモデルパラメータを保存・読み込みするために必要なpythonライブラリ
from mylib.function import sigmoid, softmax   #自前ライブラリ,今回はシグモイド関数・ソフトマックス関数を使う
from PIL import ImageFont     #PIL(Pillow):画像を扱うpythonライブラリ。今回は判定結果を数字表示するために使う。
from PIL import ImageDraw
from PIL import Image
from collections import OrderedDict
from common.layers import *      #自前ライブラリ,モデルの構築に必要なレイヤをインポートしている
from deep_convnet import *       #自前ライブラリ,学習・推論時に使用するモデルを指定する。-
                                 #-今回はCNNを使った何層かのニューラルネットワークモデルを使う

HEIGHT =500
WIDTH  =500
FONT_SIZE  = 400


#判定した数字をGUIウィンドウに表示する
def number_show(num:str):
		## 描画準備
		canvas = np.full((HEIGHT,WIDTH,3),255,np.uint8)
		
		## 日本語を描画するのは少し手間がかかる
		### 自身の環境に合わせてフォントへのpathを指定する
		font = ImageFont.truetype(
			'C:\Windows\Fonts\meiryo.ttc',
			FONT_SIZE)
		canvas = Image.fromarray(canvas)
		draw = ImageDraw.Draw(canvas)
		draw.text((120, 0),
			num,
			font=font,
			fill=(0, 0, 0, 0)) 
		canvas = np.array(canvas)
		## 判定結果を描画
		cv2.imshow('number',canvas)
   


#カメラからの入力映像を処理するための下準備 
cap = cv2.VideoCapture(0)                      #使うカメラをポート番号で指定
cv2.namedWindow('camera',cv2.WINDOW_NORMAL)    #画像表示ウィンドウを拡大可能にする
cv2.namedWindow('camera2',cv2.WINDOW_NORMAL)   #画像表示ウィンドウを拡大可能にする
cv2.namedWindow('number',cv2.WINDOW_NORMAL)   #画像表示ウィンドウを拡大可能にする
cv2.namedWindow('camera3',cv2.WINDOW_NORMAL)   #画像表示ウィンドウを拡大可能にする

#データの準備
count = 0                     #while処理の中のif文に使う
network = DeepConvNet()       #モデルの指定
network.load_params('deep_convnet_params.pkl')    #すでに学習済みの重みパラメータを行列で取得



#メイン処理
while True:
    ret, frame = cap.read()                                            #カメラから静止画を取得。取得出来たら'ret=True'となる。'frame'には静止画データが３次元配列で入っている。
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)               #frameに格納された画像情報をグレイスケール化
    frame_gray_resize = cv2.resize(frame_gray,(28,28),cv2.INTER_CUBIC) #さらに画像情報を”28×28”化。カーネルは指定できる。
    frame_gray_resize_flat = frame_gray_resize.flatten()               #さらに画像情報を多次元→1次元配列化
    frame_gray_resize_flat_normalize_inversion= 1-(frame_gray_resize_flat/255)      #さらに画像情報を正規化,白黒反転

    for i in range(0,783): #正規化、白黒反転した画像のピクセル毎に、１か０に振り分けて入力を単純化する。（mnistの学習の際は、白黒の極端な二値じゃないみたいだが、mnistがどんな処理を受けてるか知らないのでここは改善の余地あり）
       if frame_gray_resize_flat_normalize_inversion[i] > 190/255:
         frame_gray_resize_flat_normalize_inversion[i] = 1
       elif frame_gray_resize_flat_normalize_inversion[i] > 120/255:
         frame_gray_resize_flat_normalize_inversion[i] = 0
       else :
        frame_gray_resize_flat_normalize_inversion[i] = 0

       i += 1
       
    frame_gray_resize_flat_normalize_inversion_demension4 = np.array(frame_gray_resize_flat_normalize_inversion).reshape(1,1,28,28)#バッチ処理対応
    frame_gray_resize_flat_normalize_inversion_demension2 = np.array(frame_gray_resize_flat_normalize_inversion).reshape(28,28)#imshow()対応

    cv2.imshow('camera3',frame_gray_resize_flat_normalize_inversion_demension2)#画像表示(反転,正規化,二値化,28×28化)
    cv2.imshow('camera2',frame_gray_resize)#画像表示(反転,28×28化)
    cv2.imshow('camera',frame_gray)#画像表示(反転)
    cv2.waitKey(10)#動画フレーム指定。(静止画像出力間時間の指定)
    
    y = network.predict(frame_gray_resize_flat_normalize_inversion_demension4)#推論
    p = np.argmax(y)         #1次元配列"y"の最も確率の高いインデックスを取得する
    number_show(str(p))  #予測された数字を出力
    cv2.waitKey(10)#待機時間
    
    
    
    if count == 0:
       print(frame_gray.shape)
       print(frame_gray_resize.shape)
       print(frame.shape)
       print(frame_gray_resize_flat.shape)
       print(frame_gray_resize_flat_normalize_inversion.shape)
     #GUIウィンドウの位置指定
       cv2.moveWindow('camera',200,100)
       cv2.moveWindow('camera2',600,100)
       cv2.moveWindow('camera3',1000,100)
       cv2.moveWindow('number',300,500)

       count = 1

cap.release()#カメラを停止
cv2.destroyAllWindows()#表示ウィンドウをデストロイ