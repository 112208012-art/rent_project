from openai import OpenAI
import streamlit as st
import pandas as pd
import joblib
from sklearn.cluster import KMeans
import shap
import matplotlib.pyplot as plt

model = joblib.load("rent_model.pkl")
kmeans = joblib.load("kmeans.pkl")
#shap
explainer = shap.TreeExplainer(model)

feature_names = [
    ' 拎包入住', 'AI影音', '可短租', '可開伙',
    '可養寵物', '押一付一', '有陽台', '有電梯',
    '租金補貼', '高齡友善', 'ping',
    'cluster', 'type', '可租補'
]

st.title("租金合理性預測系統")

st.write("輸入房屋條件，系統會預測合理租金，並判斷這間房子值不值得。")

actual_rent = st.number_input("實際租金", min_value=0, value=15000)

ping = st.number_input("坪數", min_value=1.0, value=10.0)

lat = st.number_input("緯度 lat", value=24.9876, format="%.6f")
lng = st.number_input("經度 lng", value=121.5754, format="%.6f")

room_type = st.selectbox("房型", ["雅房", "分租套房", "獨立套房", "整層住家"])

type_mapping = {
    "雅房": 1,
    "分租套房": 2,
    "獨立套房": 3,
    "整層住家": 4
}

st.subheader("設備條件")

col1, col2 = st.columns(2)

with col1:
    move_in = st.checkbox(" 拎包入住")
    ai_video = st.checkbox("AI影音")
    short_rent = st.checkbox("可短租")
    cook = st.checkbox("可開伙")
    pet = st.checkbox("可養寵物")
    one_pay_one = st.checkbox("押一付一")
    balcony = st.checkbox("有陽台")

with col2:
    elevator = st.checkbox("有電梯")
    rent_subsidy = st.checkbox("租金補貼")
    elderly = st.checkbox("高齡友善")
    subsidy = st.checkbox("可租補")

if st.button("開始預測"):

    cluster = kmeans.predict([[lat, lng]])[0]

    user_input = {
        ' 拎包入住': int(move_in),
        'AI影音': int(ai_video),
        '可短租': int(short_rent),
        '可開伙': int(cook),
        '可養寵物': int(pet),
        '押一付一': int(one_pay_one),
        '有陽台': int(balcony),
        '有電梯': int(elevator),
        '租金補貼': int(rent_subsidy),
        '高齡友善': int(elderly),
        'ping': ping,
        'cluster': cluster,
        'type': type_mapping[room_type],
        '可租補': int(subsidy)
    }

    user_df = pd.DataFrame([user_input])
    user_df = user_df[feature_names]

    pred_rent = model.predict(user_df)[0]

    #shap
    shap_values = explainer.shap_values(user_df)

    #open ai
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
    )

    MODEL = "qwen/qwen-2.5-7b-instruct" 

    feature_importance = dict(
        zip(user_df.columns.tolist(), shap_values[0].tolist())
    )

    print(feature_importance)

    diff = actual_rent - pred_rent

    if diff > 0:
        rent_judgement = "實際租金過高"
        worth_judgement = "不值得租"
        factor_rule = "請挑選 SHAP value 為正值的條件，說明哪些條件讓模型預測租金偏高。"
    elif diff < 0:
        rent_judgement = "實際租金過低"
        worth_judgement = "值得租"
        factor_rule = "請挑選 SHAP value 為負值的條件，說明哪些條件讓模型預測租金偏低。"
    else:
        rent_judgement = "實際租金合理"
        worth_judgement = "可以考慮租"
        factor_rule = "請簡短說明主要影響條件。"

    prompt = f"""
    你是一位租金合理性分析專家。

    請嚴格根據以下判斷結果回答，不可以自行更改結論。

    判斷規則：
    - 實際租金 > 預測租金：實際租金過高，不值得租
    - 實際租金 < 預測租金：實際租金過低，值得租
    - 實際租金 = 預測租金：租金合理，可以考慮租

    本次判斷：
    - 預測租金：{pred_rent:.0f} 元
    - 實際租金：{actual_rent:.0f} 元
    - 差額：{abs(diff):.0f} 元
    - 判斷：{rent_judgement}
    - 結論：{worth_judgement}

    房型代碼：
    1=雅房，2=分租套房，3=獨立套房，4=整層住家

    房屋條件與對租金的影響：
    {feature_importance}

    分析要求：
    {factor_rule}

    請用繁體中文 50 字以內回答。
    回答格式固定如下：
    預測租金為X元，實際租金為Y元，相差Z元，判斷為「...」。主要原因是...。結論：...。
    """

    with st.spinner("AI 正在分析租金合理性，請稍候..."):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = resp.choices[0].message.content

    st.subheader("AI 租金合理性分析")
    st.write(text)



