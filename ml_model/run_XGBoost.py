import xgboost as xgb
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_hastie_10_2
from sklearn import metrics
import pickle

def run_xgboost():
    data = pickle.load(open("./xgboost_sim.json", "rb"))
    X, y = data
    X = np.array(X)
    y = np.array(y)
    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.3,
                                                                      random_state=0)  ##test_size测试集合所占比例

    params = {
        'objective': 'binary:logistic',  # 多分类的问题
        'max_depth': 6,  # 构建树的深度，越大越容易过拟合
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.3,  # 如同学习率
        'lambda': 1.5,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 1,  # 随机采样训练样本
        'colsample_bytree': 0.7,  # 生成树时进行的列采样
        'min_child_weight': 3,
    }
    dtrain = xgb.DMatrix(train_data, label=train_label)

    # 训练数据
    num_round = 1000
    bst_model = xgb.train(params, dtrain, num_round)

    # 对测试集进行预测
    dtest = xgb.DMatrix(test_data)
    y_pred = bst_model.predict(dtest)
    y_pred = list(map(round, y_pred))
    print(metrics.accuracy_score(test_label, y_pred))

    bst_model.save_model("./xgboost_n.model")
    pass

def test_xgboost():
    feature = [[0.3, 50, 0.0]]
    feature = np.array(feature)
    dtest = xgb.DMatrix(feature)
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model("../XGBoost/xgboost.model")  # load data
    score = bst.predict(dtest)
    print(score)
if __name__ == '__main__':
    run_xgboost()
    # test_xgboost()
    pass
    x = dict()
    for key, value in x.i