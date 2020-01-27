# 途中で使用するため、あらかじめ読み込んでおいてください。
# データ加工・処理・分析モジュール
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd
pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 500)

# 可視化モジュール
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# 機械学習モジュール
import sklearn


import requests, zipfile
from io import StringIO
import io

# データの分割（学習データとテストデータに分ける）
# sklearnのバージョンによっては train_test_splitはsklearn.cross_validationにしか入ってない場合があります
from sklearn.model_selection import train_test_split

# モデル
from sklearn import linear_model

# url
auto_data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
s = requests.get(auto_data_url).content
auto_data = pd.read_csv(io.StringIO(s.decode('utf-8')),header=None)
auto_data.columns =["symboling","normalized-losses","make","fuel-type"
                     ,"aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length"
                   ,"width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system"
                    ,"bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

print(auto_data)
auto_data.to_csv('auto_data.csv', index=False)
sub_auto_data = auto_data[["price","horsepower","width","height"]]
sub_auto_data = sub_auto_data.replace('?', np.nan).dropna()
sub_auto_data = sub_auto_data.assign(price=pd.to_numeric(sub_auto_data.price))
sub_auto_data = sub_auto_data.assign(horsepower=pd.to_numeric(sub_auto_data.horsepower))
sub_auto_data.head()

# モデルのインスタンス
l_model = linear_model.LinearRegression()
r_model = linear_model.Ridge()

# 説明変数に "price" 以外を利用
X = sub_auto_data.drop("price", axis=1)
# 目的変数
Y = sub_auto_data["price"]

# 学習データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
print("X_train:\n", X_train, "\n")
print("X_test:\n", X_test, "\n")

# モデルのあてはめ
#linear
clf = l_model.fit(X_train, y_train)
print("clf:", clf, "\n")
print("train:", clf.__class__.__name__, clf.score(X_train, y_train))
print("test:", clf.__class__.__name__, clf.score(X_test, y_test), "\n")

#ridge
clf2 = r_model.fit(X_train, y_train)
print("clf:", clf2, "\n")
print("train:", clf2.__class__.__name__, clf2.score(X_train, y_train))
print("test:", clf2.__class__.__name__, clf2.score(X_test, y_test), "\n")


# 偏回帰係数
print(pd.DataFrame({"Name": X.columns,
                    "Coefficients": clf.coef_}).sort_values(by='Coefficients'))

# 切片
print(clf.intercept_)