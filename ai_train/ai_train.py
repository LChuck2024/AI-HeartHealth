# 导入需要的包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import joblib
import os
import sys

warnings.filterwarnings('ignore', category=FutureWarning)  # 忽略 FutureWarning

def create_dict_path():
    return os.path.dirname(os.path.abspath(sys.argv[0]))

dicts_path = os.path.join(create_dict_path(), 'ai_train','dicts')
models_path = os.path.join(create_dict_path(), 'ai_train','models')

if not os.path.exists(dicts_path):
    os.makedirs(dicts_path)

if not os.path.exists(models_path):
    os.makedirs(models_path)


def data_read(filename, encoding='utf-8', rowCount=None):
    # 读取数据
    try:
        # 读取 CSV 文件
        src_data = pd.read_csv(filepath_or_buffer=filename, encoding=encoding)
    except Exception as e:
        print(f"读取文件 {filename} 时发生错误: {e}")
        return None
    # 样本均衡
    if rowCount is not None:
        rows = int(rowCount / 2)
        # 先从src_data中筛选出HeartDisease值为'No'的前25000行
        no_data = src_data[src_data['HeartDisease'] == 'No'].head(rows)
        # 同样的道理，筛选出HeartDisease值为'Yes'的前25000行
        yes_data = src_data[src_data['HeartDisease'] == 'Yes'].head(rows)
        # 将筛选并取好的两部分数据合并起来，axis=0表示按行方向合并（纵向合并）
        src_data = pd.concat([no_data, yes_data], axis=0)
    # 数据清洗
    result_data = src_data.drop_duplicates().dropna()
    return result_data


def data_change(data):
    # 数据转换
    for col in data.columns:
        if data[col].dtype != 'float64' or data[col].dtype != 'int64':
            # print(f'{col}:{tmp_data[col].dtype}')
            if os.path.exists(f'{dicts_path}/{col}_dict.dict'):
                tmp_dict = joblib.load(f'{dicts_path}/{col}_dict.dict')
                print(f'{col}:{tmp_dict}')
            else:
                tmp_dict = {item: index for index, item in enumerate(data[col].unique())}
            try:
                data[col] = data[col].apply(func=lambda x: tmp_dict[x])
            except:
                print(f'{col}转换失败: {data[col]}')
            joblib.dump(tmp_dict, f'{dicts_path}/{col}_dict.dict')
            print(tmp_dict)
    return data


class mlClient(object):
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.selected_models = None
        self.model_compare = []
        self.df = pd.DataFrame()
        self.models = {'KNN': KNeighborsClassifier,
                       'GaussianNB': GaussianNB,
                       'DecisionTree': DecisionTreeClassifier,
                       'Logistic': LogisticRegression,
                       'RandomForest': RandomForestClassifier,
                       'LinearSVC': LinearSVC,
                       'Voting': VotingClassifier,
                       'Bagging': BaggingClassifier,
                       'Boost': AdaBoostClassifier,
                       'Stacking': StackingClassifier,
                       'LGBM': LGBMClassifier
                       }
        self.estimators = [[('KNN', KNeighborsClassifier(n_neighbors=17)),
                            ('GaussianNB', GaussianNB()),
                            ('DecisionTree', DecisionTreeClassifier()),
                            ('Logistic', LogisticRegression(max_iter=100000)),
                            ('RandomForest', RandomForestClassifier()),
                            ('LinearSVC', LinearSVC(dual=True))
                            ]]
        self.params = {'KNN': {'n_neighbors': [17]},
                       'Logistic': {'max_iter': [100000]},
                       'LinearSVC': {'dual': [True]},
                       'Voting': {'estimators': self.estimators},
                       'Stacking': {'estimators': self.estimators, 'final_estimator': [RandomForestClassifier()]},
                       }
        self.column_names = ["model", "train_time", "pred_time", "accuracy", "precision", "recall", "f1", "importances"]

    def main(self, filename, rowCount, selections=['LGBM']):
        # 读取数据
        tmp_data = data_read(filename=filename, rowCount=rowCount)
        # 数据转换
        print('数据转换')
        tmp_data = data_change(tmp_data)
        # 构建数据样本
        X = tmp_data.drop(columns='HeartDisease')
        y = tmp_data['HeartDisease']

        # 数据拆分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # 数据标准化
        self.mu = X_train.mean()
        self.sigma = X_train.std()
        X_train = (X_train - self.mu) / self.sigma
        X_test = (X_test - self.mu) / self.sigma

        self.selected_models = {key: self.models[key] for key in selections if key in self.models}

        for name, model_class in self.selected_models.items():
            if name in self.params and name in selections:
                param = {k: v[0] for k, v in self.params[name].items()}
                # print(name,param)
                clf = model_class(**param)
            else:
                clf = model_class()

            train_start_time = datetime.datetime.now()
            # 模型训练
            clf.fit(X=X_train, y=y_train)
            train_end_time = datetime.datetime.now()
            train_time = round((train_end_time - train_start_time).total_seconds(), 2)
            # 模型预测
            y_pred = clf.predict(X=X_test)
            pred_end_time = datetime.datetime.now()
            pred_time = round((pred_end_time - train_end_time).total_seconds(), 2)
            # 模型评估
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            if name in ('DecisionTree','RandomForest','LGBM'):
                importances = clf.feature_importances_
            else:
                importances = None
            
            self.model_compare.append([name, train_time, pred_time, accuracy, precision, recall, f1, importances])
            self.df = pd.DataFrame(self.model_compare, columns=self.column_names)

            # 覆盖保存模型
            joblib.dump([clf, self.mu, self.sigma,f1], f'{models_path}/{name}.pkl')


if __name__ == '__main__':
    client = mlClient()
    client.main(filename='heart_2020_cleaned.csv', rowCount=50000)
