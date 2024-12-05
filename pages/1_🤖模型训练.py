import streamlit as st
import pandas as pd
from ai_train import ai_train
import io
import matplotlib.pyplot as plt
import datetime

# 设置页面配置
st.set_page_config(page_title="🤖模型训练",
                   layout="wide",
                   page_icon="🫀",
                   initial_sidebar_state="expanded"
                   )

# st.title("AI+健康 心脏病预测平台")
st.header("🤖 模型训练")
st.sidebar.subheader("🤖 模型训练")

# 实例化模型
client = ai_train.mlClient()

rowcount = None

st.markdown(
    """<div style="background-color:#f5f5f5;padding:10px;">
                <p style="color:#999999;">
                本页面是模型训练操作平台，能通过上传数据集并选择训练模型获得评估出最优预测模型。
                <br/>
        </div>
    """, unsafe_allow_html=True)

st.markdown(
    """
    - 数据集上传
        - 页面有数据集样例供您参考格式。点击 “上传数据集” 按钮，从本地选择符合样例格式的数据集上传，上传完成后系统会自动进行数据均衡性分析和训练集调整操作，这样能够提高模型训练的效果。
    - 训练模型选择
        - 系统列出多种可用训练模型，如KNN、支持向量机、决策树、随机森林等。您需依据数据集特点和问题需求来选择一个或者多个模型。
    - 开始训练
        - 选好数据集和模型后，点击 “开始训练”。训练时，页面会显示相关状态信息，您需耐心等待，不要中断操作。
    - 获取最优模型
        - 训练结束后，系统自动评估并展示结果，包括精准率、准确率、回归率、F1等指标。通过对比，您可确定最优模型并保存，用于后续应用。若遇问题，可查看错误提示或联系技术支持。
    """
)
# 分割线
st.markdown("""---""")
st.write("数据集样例")
sample_data = pd.read_csv("ai_train/heart_2020_cleaned.csv")
st.write(sample_data.head())

# 设置上传文件按钮
uploaded_file = st.file_uploader("请上传数据集", type=["csv"])

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

if uploaded_file is None:
    # st.write("请上传数据集")
    exit()
else:
    st.write("文件上传成功！")

progress_bar.progress(10)
status_text.text("完成10%")

# 读取文件内容并存储在变量中
file_content = uploaded_file.read()
# 将文件内容转换为 DataFrame
file_obj = io.BytesIO(file_content)

# 重置文件指针到文件开头
file_obj.seek(0)
df = pd.read_csv(file_obj)

# 数据分析
st.write('文件数据分析展示如下图：')

# 创建列布局
cols = st.columns(2)

clas_count = df['HeartDisease'].value_counts()

# st.write(clas_count.index,clas_count.values)

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
    ax_bar.text(bar.get_x() + bar.get_width() / 2, yval + 0.05 * yval, f'{yval // 1000}k', ha='center', va='bottom')

# 创建饼图
fig_pie, ax_pie = plt.subplots(figsize=(4, 4), dpi=200, constrained_layout=True)  # 设置图形大小为 4x4 英寸，增加 dpi 提高分辨率
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

# 如果clas_count中数据量差异很大，则进行数据处理
if clas_count[0] / clas_count[1] > 1.5 or clas_count[1] / clas_count[0] > 1.5:
    # rowcount = 2
    rowcount = min(clas_count[0], clas_count[1]) * 2
    st.write(f'数据不均衡，将自动均衡取其中{rowcount}行进行训练')

progress_bar.progress(25)
status_text.text("完成25%")
# 分割线
st.write("---")

# 请选择您想预测的模型,多选框
selections = st.multiselect("请选择您想训练的模型", client.models.keys())
if not selections:
    st.write("请至少选择一个模型！")
    exit()
else:
    st.write("您选择的模型是：", ', '.join(selections))

progress_bar.progress(50)
status_text.text("完成50%")
# 分割线
st.write("---")

# 设置按钮
button = st.button("开始训练")
if button:
    st.write(">> 开始训练")
    start_time = datetime.datetime.now()
    # 重置文件指针到文件开头
    file_obj.seek(0)
    st.write(">> 训练中，请稍等...")
    client.main(filename=file_obj, rowCount=rowcount, selections=selections)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    st.write(f">> 训练结束，总耗时:{duration}秒")

    progress_bar.progress(75)
    status_text.text("完成75%")
    # 分割线
    st.write("---")

    st.write(f">> 各模型训练结果对比：")
    model_compare = client.df
    # 绘制表格
    st.dataframe(model_compare)
  
    max_index = model_compare["f1"].idxmax()
    best_model = model_compare.loc[max_index][0]
    print(model_compare.loc[max_index])

    st.write(f">> 推荐使用模型：{best_model}")

    importances = model_compare.loc[max_index][7]
    if importances is not None:
        file_obj.seek(0)
        feature_names = pd.read_csv(file_obj).drop(columns=['HeartDisease']).columns.tolist()
        df_importances = pd.DataFrame(importances, index=feature_names, columns=['Importance'])
    
      
        # 分割线
        st.write("---")
        st.write(f'{best_model}模型特征重要性如下：')
        # 绘制横向条形图
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        df_importances.plot(kind='barh', legend=False, ax=ax)
    
        # 添加标题和轴标签
        ax.set_title('Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
    
        # 显示图形
        st.pyplot(fig)

    progress_bar.progress(100)
    status_text.text("完成100%")

    st.session_state.model = best_model
