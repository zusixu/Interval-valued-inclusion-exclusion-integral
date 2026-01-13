
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def generate_data():
    """
    从UCI获取数据集并通过加减标准差的方式获得区间值数据集
    :return: 训练集和测试集
    """
    # california房价预测数据集




    # fetch dataset
    auto_mpg = fetch_ucirepo(id=9)

    # data (as pandas dataframes)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets
  
    X.fillna(X.mean(), inplace=True)
    df = X
    y = (y - y.min()) / (y.max() - y.min())
    df = (df - df.min()) / (df.max() - df.min())  # 归一化

    # 初始化两个空的DataFrame，用于存储低于和高于两倍标准差的值
    data_low = pd.DataFrame(index=df.index, columns=df.columns)
    data_up = pd.DataFrame(index=df.index, columns=df.columns)
    # %%
    # 遍历每一列，计算标准差，并基于标准差创建新的列，获得了区间值集low,up表示上下界
    for feature in df.columns:
        std_deviation = df[feature].std()
        data_low[feature] = df[feature] - 2 * std_deviation
        data_up[feature] = df[feature] + 2 * std_deviation
    data_low = pd.concat((data_low, y), axis=1)
    data_up = pd.concat((data_up, y), axis=1)
    Df = pd.concat((data_low, data_up), axis=1)
    lenth = len(data_up.columns)  # 给出特征＋标签的数量

    data_train, data_test = train_test_split(Df, test_size=0.2, random_state=42)
    # 前一半是l后一半是u
    X_train = pd.concat((data_train.iloc[:, :lenth - 1], data_train.iloc[:, lenth:2 * lenth - 1]), axis=1)
    y_train = pd.concat((data_train.iloc[:, lenth - 1:lenth], data_train.iloc[:, 2 * lenth - 1:2 * lenth]), axis=1)
    X_test = pd.concat((data_test.iloc[:, :lenth - 1], data_test.iloc[:, lenth:2 * lenth - 1]), axis=1)
    y_test = pd.concat((data_test.iloc[:, lenth - 1:lenth], data_test.iloc[:, 2 * lenth - 1:2 * lenth]), axis=1)
    return X_train, X_test, y_train, y_test