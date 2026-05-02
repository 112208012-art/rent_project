from urllib.parse import urlencode
import pandas as pd
from time import sleep
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.special import expit  # sigmoid

df = pd.read_csv("591完整資料(1).csv")
outliers = [210,15]
df = df.drop(index=outliers)


feature_names = [' 拎包入住', 'AI影音',   '可短租', '可開伙',
       '可養寵物', '押一付一',  '有陽台', '有電梯',  '租金補貼',
       '高齡友善', 'ping',
        'cluster','type','可租補']
#'南北通透' ,'有車位', '免管理費', '可入籍','可報稅','社會住宅','免服務費', '新上架','隨時可遷入',
df_new=df[feature_names]
df_rent = df['Rent']

# 建立模型
model = XGBRegressor(
    n_estimators=300,     # 樹的數量
    max_depth=6,          # 樹的最大深度
    learning_rate=0.05,   # 每棵樹對最終預測的影響
    subsample=0.8,        # 隨機取樣資料防 overfit
    colsample_bytree=0.8, # 隨機取樣 feature
    random_state=42,
    enable_categorical = True,
    objective="reg:absoluteerror"
)

X = df_new
y = df_rent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
steepness = 0.5
X_train["ping"].mean()
w = 1 + 2 * expit((X_train["ping"] - X_train["ping"].mean()) * steepness)
# 訓練
model.fit(X_train, y_train, sample_weight=w)

# 預測
y_pred = model.predict(X_test)

# 評估
print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

#預測租金