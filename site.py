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

    shap_df = pd.DataFrame({
    "條件": user_df.columns,
    "影響金額": shap_values[0]
    })

    shap_df["影響方向"] = shap_df["影響金額"].apply(
    lambda x: "提高租金" if x > 0 else "降低租金"
    )

    shap_df["絕對影響"] = shap_df["影響金額"].abs()

    shap_df = shap_df.sort_values("絕對影響", ascending=False)

    st.subheader("影響租金最多的條件")

    top = shap_df.iloc[0]

    st.write(f"影響最大的條件是：**{top['條件']}**")
    st.write(f"影響方向：{top['影響方向']}")
    st.write(f"影響金額：約 {top['影響金額']:.0f} 元")

    st.write("前 5 個影響租金最多的條件：")
    st.dataframe(shap_df[["條件", "影響方向", "影響金額"]].head(5))

    st.subheader("SHAP 解釋圖")
    fig, ax = plt.subplots()
    shap.plots.bar(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=user_df.iloc[0],
            feature_names=user_df.columns
        ),
        show=False
    )

    st.pyplot(fig)


    #shap結束

    premium_rate = (actual_rent - pred_rent) / pred_rent

    if cluster==9:
        place=''


    st.subheader("預測結果")
    st.write(f"系統判斷 cluster：{cluster}")
    st.write(f"模型預測合理租金：約 {pred_rent:,.0f} 元")
    st.write(f"實際租金：{actual_rent:,.0f} 元")
    st.write(f"價差比例：{premium_rate:.2%}")

    if premium_rate > 0.2:
        st.error("這間房子偏貴很多，不太值得。")
    elif premium_rate > 0.05:
        st.warning("這間房子有點偏貴。")
    elif premium_rate < -0.2:
        st.success("這間房子明顯便宜，可能很值得。")
    elif premium_rate < -0.05:
        st.success("這間房子略便宜，可以考慮。")
    else:
        st.info("這間房子的租金大致合理。")

