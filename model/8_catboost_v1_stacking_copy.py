# -*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import os
dire = '../../data/'

train = pd.read_csv(dire + 'backup/train6.csv', encoding='utf-8')
test = pd.read_csv(dire + 'backup/test6.csv', encoding='utf-8')

data_train = pd.read_csv(dire + 'data_open/data_train.csv', encoding='gb2312')
data_test = pd.read_csv(dire + 'data_open/data_test.csv', encoding='gb2312')
data_train.drop(['futureOrderType'], axis=1, inplace=True)

train = pd.merge(train, data_train, on='userid', how='left')
test = pd.merge(test, data_test, on='userid', how='left')

train_feature = pd.read_csv(dire + 'backup/LOVECT/train_feature1.csv', encoding='utf-8', header=None)
test_feature = pd.read_csv(dire + 'backup/LOVECT/test_feature1.csv', encoding='utf-8', header=None)
train = pd.concat([train, train_feature], axis=1)
test = pd.concat([test, test_feature], axis=1)

print("训练集：", train.shape)
print("测试集：", test.shape)

# 重构的时间特征
# train8 = pd.read_csv(dire + 'backup/train8_56789_select.csv', encoding='utf-8')
# test8 = pd.read_csv(dire + 'backup/test8_56789_select.csv', encoding='utf-8')
# reextract_action_feature = [x for x in train8.columns if x not in ['userid', 'orderType']]
# # train.drop(reextract_action_feature, axis=1, inplace=True)
# # test.drop(reextract_action_feature, axis=1, inplace=True)
# train = pd.merge(train, train8, on=['userid', 'orderType'], how='left', suffixes=('_x', '_y'))
# test = pd.merge(test, test8, on='userid', how='left', suffixes=('_x', '_y'))

print("融合重构action特征后训练集：", train.shape)
print("融合重构action特征后测试集：", test.shape)

# 计算AUC
def caculateAUC(test_pre, test_label):
    pre = np.array(test_pre[:, 1])
    result = metrics.roc_auc_score(test_label, pre)
    return result

# initialize data
def one_hot(table, one_hot_feature):
    for f in one_hot_feature:
        dummies = pd.get_dummies(table[f], prefix= f)
        table = pd.concat([table, dummies], axis=1)
        table.drop([f], axis=1, inplace=True)
    return table

one_hot_feature = ['gender_e', 'age_e', 'province_e', 'continent_e', 'season']
dataset = pd.concat([train,test])
dataset = one_hot(dataset, one_hot_feature)
train = dataset[dataset.orderType.notnull()]
test = dataset[dataset.orderType.isnull()]

feature = [x for x in dataset.columns if x not in ['userid', 'gender', 'province', 'age',
         'orderType', 'actionTime_future', 'action_time_future', 'orderTime', 'order_time',
         'time_1', 'time_2', 'time_2_c', 'gender_e', 'age_e', 'province_e', 'continent_e', 'season', 'use_app_days_count'

         ,'莫斯科', '胡志明市', 'tag_举牌迎接', 'tag_额外收费', '芭堤雅', '花莲', '芽庄', '苏梅岛', 'tag_驾驶平稳', 'province_e_23.0', 'tag_建筑师', 'tag_驾驶技术一般', 'tag_留学生', 'province_e_21.0', 'province_e_26.0', '菲律宾', '萨尔茨堡', 'province_e_20.0', '蒂卡波湖', '蒙特利尔', '西藏', 'tag_驾驶鲁莽', 'tag_接待地点不合理', '谢菲尔德', '贝尔法斯特', '考文垂', 'tag_临时换司导', '老挝', '科茨沃尔德', 'tag_车况有点旧', '瑞士', 'rating_min', 'tag_车旧/脏', 'continent_e_5.0', '盐湖城', '神户', 'tag_不协助搬运行李', 'continent_e_2.0', 'tag_非订单车辆', '福森', '科隆', 'tag_提前就终止了服务', '箱根', 'tag_频繁催促', '约克', '约翰内斯堡', 'province_e_30.0', 'province_e_29.0', 'province_e_28.0', '维多利亚', 'province_e_27.0', '缅甸', 'tag_不爱讲话', 'tag_路线不熟', '费城', 'age_e_3.0', 'province_e_5.0', 'province_e_10.0', 'tag_行前联系不上', 'province_e_8.0', '雅典', '雅加达', '雷克雅未克', '霍巴特', 'province_e_7.0', '静冈县', 'province_e_6.0', '首尔', '香港', '赫尔辛基', '马拉加', 'tag_未提前联系', '马赛', 'tag_行前未主动联系', '黄金海岸', '黑龙江', 'tag_沉默寡言', 'province_e_2.0', 'province_e_1.0', 'province_e_0.0', 'tag_没水/空调不足', 'tag_景点不熟悉', '阿里山', '珀斯', '阿维尼翁', 'tag_言语粗鲁', '越南', 'province_e_15.0', '轻井泽', 'action_56789_count_c', '达拉斯', '迈阿密', 'tag_提前结束行程', '那不勒斯', '都柏林', '里斯本', '里昂', 'province_e_14.0', 'actiontimespancount_8_9', 'tag_行程有点紧', '金泽', '金边', '釜山', '长崎', '阿姆斯特丹', '阿尔勒', 'province_e_12.0', 'tag_行程安排不合理', '北海道--小樽', 'action_6789_count', '爱尔兰', '俄罗斯', '宿务', '宜兰', '伯明翰', '奥斯陆', '奥兰多', '奈良', '冰岛', '爱丁堡', '大西洋城', '大叻', '多哈', '夏威夷茂宜岛', '夏威夷大岛', '墨西哥城', '富国岛', '富士山', '富士河口湖', '富山市', '富良野', '尼斯', '尼泊尔', '休斯敦', '岐阜县', '岘港', '巴伦西亚', '巴厘岛', '伊尔库茨克', '巴尔的摩', '巴斯', '巴西', '布达佩斯', '墨西哥', '冲绳市', '塞维利亚', '台南', '北海道--登别', '千叶', '千叶市', '华欣', '华沙', '北海道--函馆', '南投', '南非', '卡塔尔', '卡萨布兰卡', '卢塞恩', '印度尼西亚', '台中', '台北', '匈牙利', '塞班岛', '加德满都', '名古屋', '哈尔施塔特', '哥德堡', '嘉义', '利瓦绿洲', '因特拉肯', '圣保罗', '圣地亚哥', '圣彼得堡', '垦丁', '埃及', '基督城', '堪培拉', '布里斯班', '布鲁塞尔', '京都', '波士顿', '楠迪', '横滨', '比利时', '毛里求斯', '毛里求斯.1', '汉堡', '汉密尔顿', 'continent_driver_count_avg', 'histord_sum_cont4', '河内', 'histord_sum_cont5', 'histord_time_last_2_year', '法兰克福', '波兰', '波尔图', '柬埔寨', 'action_5678_count', 'histord_time_last_3_year', '洞爷湖', 'latest_1day_actionType9_count', '北海道--旭川', 'action_678_count', 'action_678_count_c', '渥太华', '温莎', 'age_00', 'action_789_count', 'action_78_count', '澳门', '熊本市', '格拉纳达', '柏林', '亚特兰大', '捷克', '广岛', '云顶高原', 'tag_车内有异味', '开普敦', '开罗', '御殿场市', '德国', '惠灵顿', '慕尼黑', '成田市', '戛纳', '丹佛', '拉科鲁尼亚', '挪威', '摩洛哥', 'histord_sum_all', '摩纳哥', '斯克兰顿（宾夕法尼亚州）', '中国香港', '新加坡.1', '新北', '新疆', '新竹', '新西兰', '中国澳门', '中国', '星野度假村', '上海', '暹粒', '曼彻斯特', '万象'
         , 'latest_2day_actionType9_count', 'continent_e_1.0', 'continent_type1_count_avg', 'continent_e_4.0', '高雄', 'province_e_17.0', 'gender_e_1.0', 'latest_7day_actionType9_count', 'age_e_1.0', 'province_e_25.0', 'latest_5day_actionType8_count', 'province_e_9.0', 'province_e_22.0', 'province_e_13.0', '韩国', 'latest_6day_actionType9_count', '宁夏', '青海', 'action_5678_count_c', '奥地利', '布拉格', '多伦多', '哥本哈根', '斐济', '斯德哥尔摩', '北海道--札幌', '槟城', '佛罗伦萨', '丹麦', '波尔多', 'action_56789_count', '海南', '牛津', 'rating_mean', '皇后镇', 'tag_车辆和订单显示不符', 'tag_车况脏乱', '米兰', 'tag_私聊不回复', 'tag_没做景点介绍', '芬兰', '英国', '荷兰', '葡萄牙', 'tag_未举牌服务', 'tag_景点不介绍', 'tag_主动热情', 'tag_路线熟悉'
          ]]
print("训练集：", train.shape)
print("测试集：", test.shape)

train_x, val_x, train_y, val_y = train_test_split(train[feature], train["orderType"], test_size=0.2, random_state=0, stratify=train["orderType"])

class LGBClassifier(object):
    def __init__(self,params):
        self.params = params

    def fit(self, X, y, w=None):
        if w==None:
            w = np.ones(X.shape[0])
        # self.scaler = StandardScaler().fit(y)
        # y = self.scaler.transform(y)
        split = int(X.shape[0] * 0.8)
        indices = np.random.permutation(X.shape[0])
        train_id, test_id = indices[:split], indices[split:]
        x_train, y_train, w_train, x_valid, y_valid,  w_valid = X[train_id], y[train_id], w[train_id], X[test_id], y[test_id], w[test_id],
        d_train = lgb.Dataset(x_train, y_train, weight=w_train)
        d_valid = lgb.Dataset(x_valid, y_valid, weight=w_valid)
        partial_bst = lgb.train(self.params, d_train, 5000, valid_sets=[d_valid], early_stopping_rounds=50)
        num_round = partial_bst.best_iteration
        d_all = lgb.Dataset(X, label = y, weight=w)
        self.bst = lgb.train(self.params, d_all, num_round)

    def predict(self, X):
        return self.bst.predict(X)


class XGBClassifier(object):
    def __init__(self, params):
        self.params = params

    def fit(self, X, y, w=None):
        if w==None:
            w = np.ones(X.shape[0])
        split = int(X.shape[0] * 0.8)
        indices = np.random.permutation(X.shape[0])
        train_id, test_id = indices[:split], indices[split:]
        x_train, y_train, w_train, x_valid, y_valid,  w_valid = X[train_id], y[train_id], w[train_id], X[test_id], y[test_id], w[test_id],
        d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        partial_bst = xgb.train(self.params, d_train, 5000, early_stopping_rounds=50, evals = watchlist, verbose_eval=100)
        num_round = partial_bst.best_iteration
        d_all = xgb.DMatrix(X, label = y, weight=w)
        self.bst = xgb.train(self.params, d_all, num_round)

    def predict(self, X):
        test = xgb.DMatrix(X)
        return self.bst.predict(test)


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X_train, Y_train, X_test):
        X = X_train.values
        y = Y_train.values
        T = X_test.values

        X_train[X_train == np.inf] = np.nan
        X_test[X_test == np.inf] = np.nan
        X_fillna = X_train.fillna(X_train.min()).values
        T_fillna = X_test.fillna(X_test.min()).values

        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True))
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            print('Training base model ' + str(i+1) + '...')
            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                print('Training round ' + str(j+1) + '...')
                if clf not in [xgb1,lgb1]: # sklearn models cannot handle missing values.
                    X = X_fillna
                    T = T_fillna
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # w_holdout = w[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                if clf in [xgb1, xgb2, xgb3, xgb4, xgb5, lgb1, lgb2, lgb3, lgb4, lgb5, SVM]:
                    y_pred = clf.predict(X_holdout)
                    S_train[test_idx, i] = y_pred
                    S_test_i[:, j] = clf.predict(T)
                else:
                    y_pred = clf.predict_proba(X_holdout)[:,1]
                    S_train[test_idx, i] = y_pred
                    S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(1)
        self.S_train, self.S_test, self.y = S_train, S_test, y  # for diagnosis purpose
        self.corr = pd.concat([pd.DataFrame(S_train),Y_train],1).corr() # correlation of predictions by different models.
        # cv_stack = ShuffleSplit(n_splits=6, test_size=0.2)
        # score_stacking = cross_val_score(self.stacker, S_train, y, cv=cv_stack, n_jobs=1, scoring='neg_mean_squared_error')
        # print(np.sqrt(-score_stacking.mean())) # CV result of stacking
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict_proba(S_test)[:,1]
        # return y_pred
        return S_train, y, S_test, y_pred
lgb1_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 130,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': 4,
    'verbose': -1
}
lgb1 = LGBClassifier(lgb1_params)

lgb2_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 135,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': 4,
    'verbose': -1
}
lgb2 = LGBClassifier(lgb2_params)

lgb3_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 140,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 4,
    'scale_pos_weight': 4,
    'verbose': -1
}
lgb3 = LGBClassifier(lgb3_params)

lgb4_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 130,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'scale_pos_weight': 4,
    'verbose': -1
}
lgb4 = LGBClassifier(lgb4_params)

lgb5_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 130,
    'learning_rate': 0.01,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.9,
    'bagging_freq': 4,
    'scale_pos_weight': 4,
    'verbose': -1
}
lgb5 = LGBClassifier(lgb5_params)

params1 = {
    'booster': "gbtree",'objective': "binary:logistic",'eval_metric': "auc",
    'max_depth': 9,'subsample': 0.8,'colsample_bytree ': 0.9,'eta': 0.01,
    'lambda': 3,'silent': 0,'scale_pos_weight': 4,'min_child_weight': 1}
xgb1 = XGBClassifier(params1)
# params2 = {'booster': 'gblinear', 'alpha': 0,  # for gblinear, delete this line if change back to gbtree
#            'eta': 0.1, 'max_depth': 2, 'subsample': 1, 'colsample_bytree': 1, 'min_child_weight': 1,
#            'gamma': 0, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
# xgb2 = XGBClassifier(params2)
xgb2_params = {
    'booster': "gbtree",'objective': "binary:logistic",'eval_metric': "auc",
    'max_depth': 7,'subsample': 0.88,'colsample_bytree ': 0.8,'eta': 0.01,
    'lambda': 4,'silent': 0,'scale_pos_weight': 4,'min_child_weight': 1}
xgb2 = XGBClassifier(xgb2_params)


xgb3_params = {
    'booster': "gbtree",'objective': "binary:logistic",'eval_metric': "auc",
    'max_depth': 10,'subsample': 0.85,'colsample_bytree ': 0.9,'eta': 0.01,
    'lambda': 3,'silent': 0,'scale_pos_weight': 5,'min_child_weight': 1}
xgb3 = XGBClassifier(xgb3_params)

xgb4_params = {
    'booster': "gbtree",'objective': "binary:logistic",'eval_metric': "auc",
    'max_depth': 11,'subsample': 0.8,'colsample_bytree ': 0.85,'eta': 0.01,
    'lambda': 2,'silent': 0,'scale_pos_weight': 4,'min_child_weight': 1}
xgb4 = XGBClassifier(xgb4_params)

xgb5_params = {
    'booster': "gbtree",'objective': "binary:logistic",'eval_metric': "auc",
    'max_depth': 8,'subsample': 0.7,'colsample_bytree ': 0.9,'eta': 0.01,
    'lambda': 1,'silent': 0,'scale_pos_weight': 5,'min_child_weight': 1}
xgb5 = XGBClassifier(xgb5_params)

xgb_params={
    "learning_rate":0.01,"max_depth":8,"subsample":0.9,
    'colsample_bytree': 0.9,'objective': 'binary:logistic',
    'silent': 0, 'n_estimators':3000, 'gamma':1, 'min_child_weight':1}
xgb = xgb.XGBClassifier(**xgb_params)

cat1 = CatBoostClassifier(iterations=2580, loss_function='Logloss', eval_metric='AUC', od_type='Iter', od_wait=80, depth=8,nan_mode='Min',random_seed=2017,)
cat2 = CatBoostClassifier(iterations=2880, loss_function='Logloss', eval_metric='AUC', od_type='Iter', od_wait=80, depth=9,nan_mode='Min',random_seed=2017,)
cat3 = CatBoostClassifier(iterations=3080, loss_function='Logloss', eval_metric='AUC', od_type='Iter', od_wait=80, depth=7,nan_mode='Min',random_seed=2017,)
cat4 = CatBoostClassifier(iterations=3580, loss_function='Logloss', eval_metric='AUC', od_type='Iter', od_wait=80, depth=9,nan_mode='Min',random_seed=2017,)
cat5 = CatBoostClassifier(iterations=3880, loss_function='Logloss', eval_metric='AUC', od_type='Iter', od_wait=80, depth=8,nan_mode='Min',random_seed=2017,)
cat6 = CatBoostClassifier(iterations=3880, loss_function='Logloss', eval_metric='AUC', od_type='Iter', od_wait=80, depth=7,nan_mode='Min',random_seed=2017,)
RF = RandomForestClassifier(n_estimators=1000, oob_score=True)
ETR = ExtraTreesClassifier(n_estimators=1000)
Ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=200)
GBR = GradientBoostingClassifier(n_estimators=200, max_depth=9, max_features=0.5)
LR = LogisticRegression(n_jobs=-1, random_state=2017)
SVM = SVC(random_state=2017)

E = Ensemble(5, xgb, [xgb2, xgb3, xgb4, xgb5, lgb1, lgb2, lgb3, lgb4, lgb5, cat1, cat2, cat3, cat4, cat5, cat6, RF, ETR, Ada, GBR, LR])
# prediction = E.fit_predict(train[feature], train["orderType"], test[feature])
S_train, y, S_test, prediction = E.fit_predict(train[feature], train["orderType"], test[feature])


prob = test[['userid']]
prob['orderType'] = prediction
sub = pd.DataFrame(prob)
print("训练完成：", datetime.now())
k = 8
x = 'xgb'
sub.to_csv(dire + 'submit/stacking/sub_stacking_prob' + k + '_' + x + '.csv', index=False)


# xgb_params={
#     "learning_rate":0.001,"max_depth":5,"subsample":0.9,
#     'colsample_bytree': 0.9,'objective': 'binary:logistic',
#     'silent': 1, 'n_estimators':30000, 'gamma':1,
#     'min_child_weight':1
# }
# clf=xgb.XGBClassifier(**xgb_params,seed=2017)
# clf.fit(S_train,y)
# prediction=clf.predict_proba(S_test)[:,1]
# prob = test[['userid']]
# prob['orderType'] = prediction
# sub = pd.DataFrame(prob)
# print("训练完成：", datetime.now())
# sub.to_csv('results/sub_stacking_prob{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

