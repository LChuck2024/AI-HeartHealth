# 模型预测
import streamlit as st
import joblib
import pandas as pd
from ai_train import ai_train
import io
import matplotlib.pyplot as plt

# 设置页面配置
st.set_page_config(page_title="🗂批量预测",
                   layout="wide",
                   page_icon="🫀",
                   initial_sidebar_state="expanded"
                   )

# st.title("AI+健康 心脏病预测平台")
st.header("🗂 批量预测")
st.sidebar.subheader("🗂 批量预测")

st.markdown(
    """<div style="background-color:#f5f5f5;padding:10px;">
                <p style="color:#999999;">
                本页面是模型批量预测操作平台，能通过上传数据集批量预测出对应结果。
                <br/>
        </div>
    """, unsafe_allow_html=True)

selected_model = st.session_state.model
st.write(f'使用推荐模型：{selected_model}，如需更改请进入模型训练页面配置')

# 提示上传文件
st.markdown(
    """
    - 请按照提供的数据集样例上传待预测文件。
    """, unsafe_allow_html=True)

st.write("数据集样例")
sample_data = pd.read_csv("ai_train/heart_2020_cleaned.csv").drop(columns=['HeartDisease'])
st.write(sample_data.head())

# 设置上传文件按钮
uploaded_file = st.file_uploader("请上传数据集", type=["csv"])

if uploaded_file is None:
    # st.write("请上传数据集")
    exit()
else:
    st.write("文件上传成功！")

# 分割线
st.markdown("""---""")

# 设置按钮
button = st.button("开始预测")

if button:
    # 读取文件内容并存储在变量中
    file_content = uploaded_file.read()
    # 将文件内容转换为 DataFrame
    file_obj = io.BytesIO(file_content)

    # 重置文件指针到文件开头
    file_obj.seek(0)
    df = pd.read_csv(file_obj)

    # 读取数据展示
    # st.write(df)

    # 数据转换
    data = ai_train.data_change(df)

    # 转换数据展示
    # st.write(data)
    # 导入模型
    [model, mu, sigma, f1] = joblib.load(f'/mount/src/ai-hearthealth/ai_train/models/{selected_model}.pkl')
    # 数据标准化
    data = (data - mu) / sigma
    # 模型预测
    prediction = model.predict(data)
    # 加载映射
    HeartDisease_dict = joblib.load('/mount/src/ai-hearthealth/ai_train/dicts/HeartDisease_dict.dict')
    result_dict = {item: idex for idex, item in HeartDisease_dict.items()}
    # 数据转换
    result = pd.DataFrame(prediction, columns=['HeartDisease'])
    result['HeartDisease'] = result['HeartDisease'].apply(lambda x: result_dict.get(x))

    # # 预测数据展示
    st.write('预测结果如下：')

    # 创建列布局
    cols = st.columns(2)

    # 数据分析
    clas_count = result['HeartDisease'].value_counts()
    # st.write(clas_count)

    # 定义颜色,
    colors = ['#1f77b4', '#FF0000']  # 蓝色和红色

    # 创建条形图
    fig_bar, ax_bar = plt.subplots(figsize=(2, 2), dpi=200, constrained_layout=True)  # 设置图形大小为 3x3 英寸，增加 dpi 提高分辨率
    bars = ax_bar.bar(clas_count.index,
                      clas_count.values,
                      color=colors,
                      edgecolor='none'
                      )
    ax_bar.set_ylabel('Count')
    # ax_bar.set_title('Bar Chart for HeartDisease', fontsize=8, pad=20)  # 设置标题并在上方显示

    # 在每个条形上方添加数量标注
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, yval + 0.05 * yval, f'{yval}', ha='center', va='bottom')

    # 创建饼图
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4), dpi=200, constrained_layout=True)  # 设置图形大小为 3x3 英寸，增加 dpi 提高分辨率
    wedges, texts, autotexts = ax_pie.pie(
        clas_count,
        labels=clas_count.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10, 'color': 'white'}  # 设置文本字体大小和颜色
    )
    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # ax_pie.set_title(f"Pie Chart for HeartDisease", fontsize=8, pad=20)
    # 添加图例
    ax_pie.legend(wedges, clas_count.index, title="HeartDisease", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # 在第1列中显示条形图
    with cols[0]:
        st.pyplot(fig_bar)
    # 在第2列中显示饼图
    with cols[1]:
        st.pyplot(fig_pie)

    file_obj.seek(0)
    df = pd.read_csv(file_obj)

    # df拼接result
    df = pd.concat([result, df], axis=1)

    # 前100行
    st.write('预测结果样例（100行）如下:')
    df = df.head(100)
    # 展示数据集
    st.write(df)

    # 下载df为csv
    st.download_button(
        label="下载完整预测结果",
        data=file_obj,
        file_name='predict_result.csv',
        mime='text/csv',
    )

