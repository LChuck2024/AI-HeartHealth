# 模型预测
import streamlit as st
import joblib
import pandas as pd
from ai_train import ai_train

# 设置页面配置
st.set_page_config(page_title="🔍问卷预测",
                   layout="wide",
                   page_icon="🫀",
                   initial_sidebar_state="expanded"
                   )


def get_age_group(age):
    if 18 <= age <= 24:
        return '18-24'
    elif 25 <= age <= 29:
        return '25-29'
    elif 30 <= age <= 34:
        return '30-34'
    elif 35 <= age <= 39:
        return '35-39'
    elif 40 <= age <= 44:
        return '40-44'
    elif 45 <= age <= 49:
        return '45-49'
    elif 50 <= age <= 54:
        return '50-54'
    elif 55 <= age <= 59:
        return '55-59'
    elif 60 <= age <= 64:
        return '60-64'
    elif 65 <= age <= 69:
        return '65-69'
    elif 70 <= age <= 74:
        return '70-74'
    elif 75 <= age <= 79:
        return '75-79'
    elif age >= 80:
        return '80 or older'


# st.title("AI+健康 心脏病预测平台")
st.header("🔍 问卷预测")
st.sidebar.subheader("🔍 问卷预测")

# 设置问卷问题
st.markdown(
    """<div style="background-color:#f5f5f5;padding:10px;">
                <p style="color:#999999;">
                本页面是根据填写的问卷结果，使用最优预测模型进行心脏病预测。
                <br/>
        </div>
    """, unsafe_allow_html=True)

# 项目目录
Home_path = st.session_state.path

selected_model = st.session_state.model
st.write(f'使用推荐模型：{selected_model}，如需更改请进入模型训练页面配置')

st.write('* 请根据实际情况填写问卷：')

bmi = float(st.text_input('1.体重指数（BMI）', '20.3'))
smoking = st.radio('2.一生中是否至少抽过100支香烟？（注：5包 = 100支香烟）', ('否', '是'))
alcoholdrinking = st.radio('3.是否酗酒？（酗酒者成年男性每周饮酒超过14杯，成年女性每周饮酒超过7杯）', ('否', '是'))
stroke = st.radio('4.是否中风过？', ('否', '是'))
physicalhealth = st.number_input('5.在过去的30天里，有多少天身体健康状况不好（0 - 30天）？（身体健康状况，健康状况不好包括身体疾病和受伤等）', 0, 30)
mentalhealth = st.number_input('6.过去30天有多少天心理健康状况不好？（0 - 30天）', 0, 30)
diffwalking = st.radio('7.走路或爬楼梯是否严重困难？', ('否', '是'))
sex = st.radio('8.性别', ('男', '女'))
agecategory = st.number_input('9.年龄', 0, 140, value=30)
race = st.radio('10.种族人种', ('白人', '黑人', '亚洲人', '美洲原住民/阿拉斯加原住民', '西班牙裔', '其他'))
diabetic = st.radio('11.是否存在糖尿病史或正患有糖尿病？', ('否', '是', '无，但有糖尿病前期', '是（孕期）'))
physicalactivity = st.radio('12.是否是在过去30天内，在正常工作之外进行身体活动或锻炼的成年人？', ('否', '是'))
genhealth = st.radio('13.总体健康状况评价', ('极好', '非常好', '一般', '好', '差'))
sleeptime = st.number_input('14.平均每天的睡眠时间（小时）', 0, 24)
asthma = st.radio('15.是否曾经或正患有哮喘', ('否', '是'))
kidneydisease = st.radio('16.是否曾经或正患有肾脏疾病？（不包括肾结石、膀胱感染或尿失禁等）', ('否', '是'))
skincancer = st.radio('17.是否曾经或正患有皮肤癌', ('否', '是'))

data = [bmi, smoking, alcoholdrinking, stroke, physicalhealth, mentalhealth, diffwalking, sex, get_age_group(agecategory), race, diabetic, physicalactivity, genhealth,
        sleeptime, asthma,
        kidneydisease, skincancer]
translation_mapping = {'是': 'Yes',
                       '否': 'No',

                       '男': 'Male',
                       '女': 'Female',

                       '非常好': 'Very good',
                       '一般': 'Fair',
                       '好': 'Good',
                       '差': 'Poor',
                       '极好': 'Excellent',

                       '白人': 'White',
                       '黑人': 'Black',
                       '亚洲人': 'Asian',
                       '美洲原住民/阿拉斯加原住民': 'American Indian/Alaskan Native',
                       '其他': 'Other',
                       '西班牙裔': 'Hispanic',

                       '无，但有糖尿病前期': 'No, borderline diabetes',
                       '是（孕期）': 'Yes (during pregnancy)'
                       }
# 定义列名
columns = [
    'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking',
    'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime',
    'Asthma', 'KidneyDisease', 'SkinCancer'
]
data = pd.DataFrame([list(map(lambda x: translation_mapping.get(x) if x in translation_mapping else x, data))], columns=columns)

# 输出问卷结果
st.markdown('### 问卷数据：')
st.write(data)

# 数据转换
data = ai_train.data_change(data)
# 导入模型
[model, mu, sigma, f1] = joblib.load(f'{Home_path}/ai_train/models/{selected_model}.pkl')
# 数据标准化
data = (data - mu) / sigma

# 分割线
st.markdown("""---""")
# 设置按钮
button = st.button("开始预测")

if button:
    prediction = model.predict(data)[0]
    HeartDisease_dict = joblib.load(f'{Home_path}/ai_train/dicts/HeartDisease_dict.dict')
    result_dict = {item: idex for idex, item in HeartDisease_dict.items()}
    result = result_dict.get(prediction)
    st.write(f'模型预测结果：{result}，预测准确率：{round(f1 * 100, 2)}%')
    if result == 'No':
        # 绿色底色白色字体
        st.markdown('<div style="background-color:green;color:white;padding:10px;">恭喜您，您没有心脏病，继续保持！</div>', unsafe_allow_html=True)
        # st.markdown('恭喜您，您没有心脏病，继续保持！')
    else:
        # 红色底色白色字体
        st.markdown('<div style="background-color:red;color:white;padding:10px;">很遗憾，您可能患有心脏病，请及时就医检查！</div>', unsafe_allow_html=True)
        # st.markdown('很遗憾，您可能患有心脏病，请及时就医检查！')
