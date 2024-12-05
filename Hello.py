# 首页
import streamlit as st
import os
import sys

# 设置页面配置
st.set_page_config(page_title="🏠项目主页",
                   layout="wide",
                   page_icon="🫀",
                   initial_sidebar_state="expanded"
                   )
st.title("AI+健康 心脏病预测平台")
st.header("🏠 项目主页")

st.sidebar.success("在上方选择一个演示。")
st.sidebar.subheader("🏠 项目主页")

# 设置默认模型算法
session_state = st.session_state
st.session_state.model = 'LGBM'
st.session_state.path = os.path.dirname(os.path.abspath(sys.argv[0]))

st.markdown(
    """<div style="background-color:#f5f5f5;padding:10px;">
                <p style="color:#999999;">
                本平台基于AI+健康项目，使用机器学习算法对心脏病进行预测，用户可自行填写问卷，系统根据用户填写的数据进行预测，用户可自行选择模型进行预测，系统会返回预测结果。
                <br/>
        </div>
    """, unsafe_allow_html=True)

# 平台功能介绍
st.markdown(
    """
    - 项目背景：
        - 心脏病发病率不断攀升，危害极大，因其隐匿性强，发病危急，严重影响患者健康。
        - 传统诊断依赖医生经验与常规检查，像心电图易有误差，对早期病变检测不佳，且偏远地区医疗资源不均，患者难获精准诊断。
        - 如今机器学习发展迅速，在医疗数据处理和疾病预测等方面表现出色，所以我们打造此 AI + 健康心脏病预测平台，改善心脏病防治及资源配置情况。
        - 运用机器学习算法，基于大量心脏病病例数据训练模型，使其掌握发病特征及风险关联。
        - 患者输入数据后，平台实时生成风险预测报告，预估短期患心脏病概率，还能初步诊断具体心脏病类型，辅助医生决策。
    """
)

# 使用自定义 CSS 样式使图片自动缩放
st.markdown(
    """
    <style>
    .auto-resize-image img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 使用 st.markdown 嵌入图片并应用自定义样式
# st.markdown(
#     '<div class="auto-resize-image"><img src="images/AI 健康心脏病预测平台介绍.png" alt="AI 健康心脏病预测平台介绍"></div>',
#     unsafe_allow_html=True
# )

# 插入图片,设置图片自适应大小
st.image('images/AI 健康心脏病预测平台介绍.png', width=1087)
