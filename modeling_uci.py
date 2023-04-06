<<<<<<< HEAD
import warnings
warnings.filterwarnings('ignore')

# 导入预处理过的数据
import pandas as pd
import pickle


target_var = 'default payment next month'
train_df=pickle.load(open('Preprocessing_binned_train_df_uci.pkl','rb'))
test_df=pickle.load(open('Preprocessing_binned_test_df_uci.pkl','rb'))
bin_strategies=pickle.load(open('Preprocessing_bin_strategies_uci.pkl','rb'))
[all_original_vars,binned_vars]=pickle.load(open('original_and_binned_vars_uci.pkl','rb'))
=======
import warnings

warnings.filterwarnings('ignore')

# 导入预处理过的数据
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, accuracy_score, confusion_matrix, \
    classification_report

target_var = 'default payment next month'
train_df = pickle.load(open('Preprocessing_binned_train_df_uci.pkl', 'rb'))
test_df = pickle.load(open('Preprocessing_binned_test_df_uci.pkl', 'rb'))
bin_strategies = pickle.load(open('Preprocessing_bin_strategies_uci.pkl', 'rb'))
[all_original_vars, binned_vars] = pickle.load(open('original_and_binned_vars_uci.pkl', 'rb'))

train_df_nona = train_df.fillna('bin_nan')


def CalcWOE(df, col, target):
    '''
    :param df: 包含需要计算WOE的变量和目标变量
    :param col: 需要计算WOE、IV的变量，必须是分箱后的变量，或者不需要分箱的类别型变量
    :param target: 目标变量，0、1表示好、坏
    :return: 返回WOE和IV
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index = True, right_index = True, how = 'left')
    regroup.reset_index(level = 0, inplace = True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis = 1)
    WOE_dict = regroup[[col, 'WOE']].set_index(col).to_dict(orient = 'index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.good_pcnt - x.bad_pcnt) * np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis = 1)
    IV = sum(IV)
    return {"WOE": WOE_dict, 'IV': IV}


WOE_dict = {}
IV_dict = {}
for var in binned_vars:
    woe_iv = CalcWOE(train_df_nona, var, target_var)
    WOE_dict[var] = woe_iv['WOE']
    IV_dict[var] = woe_iv['IV']

IV_dict_sorted = sorted(IV_dict.items(), key = lambda x: x[1], reverse = True)

# for iv_name, iv_value in IV_dict_sorted[:10]:
# print(iv_value, iv_name)

iv_threshould = 0.1
var_IV_selected = {k: IV_dict[k] for k in IV_dict.keys() if not IV_dict[k] < iv_threshould}
# print(var_IV_selected)
var_IV_selected_sorted = sorted(var_IV_selected.items(), key = lambda d: d[1], reverse = True)
var_IV_selected_sorted = [i[0] for i in var_IV_selected_sorted]
for var in var_IV_selected_sorted:
    woe_var = var + '_WOE'
    train_df_nona[woe_var] = train_df_nona[var].map(WOE_dict[var])

removed_var = []
cor_thresould = 0.5
for i in range(len(var_IV_selected_sorted) - 1):
    if var_IV_selected_sorted[i] not in removed_var:
        x1 = var_IV_selected_sorted[i] + "_WOE"
        for j in range(i + 1, len(var_IV_selected_sorted)):
            if var_IV_selected_sorted[j] not in removed_var:
                x2 = var_IV_selected_sorted[j] + "_WOE"
                cor_value = np.corrcoef([train_df_nona[x1], train_df_nona[x2]])[0, 1]
                if abs(cor_value) >= cor_thresould:
                    if IV_dict[var_IV_selected_sorted[i]] > IV_dict[var_IV_selected_sorted[j]]:
                        removed_var.append(var_IV_selected_sorted[j])
                        # print('提示：变量 {0} 与 {1} 之间的相关性为 {2}，删除 {1}'.format(x1, x2, str(cor_value)))
                    else:
                        removed_var.append(var_IV_selected_sorted[i])
                        # print('提示：变量 {0} 与 {1} 之间的相关性为 {2}，删除 {0}'.format(x1, x2, str(cor_value)))
retained_woe_vars = [i + "_WOE" for i in var_IV_selected_sorted if i not in removed_var]

for i in range(len(var_IV_selected_sorted) - 1):
    if var_IV_selected_sorted[i] in removed_var: continue
    x0 = train_df_nona[var_IV_selected_sorted[i] + '_WOE']
    x0 = np.array(x0)
    X_Col = [k + '_WOE' for k in var_IV_selected_sorted if k != var_IV_selected_sorted[i]]
    X = train_df_nona[X_Col]
    X = np.array(X)
    regr = LinearRegression()
    clr = regr.fit(X, x0)
    x_pred = clr.predict(X)
    R2 = 1 - ((x_pred - x0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()
    vif = 1 / (1 - R2)
    if vif > 10:
        print('注意：{0}的 vif 为 {1}'.format(var_IV_selected_sorted[i], vif))

X_train, y_train = train_df_nona[retained_woe_vars], train_df_nona[target_var]
LR = LogisticRegression()
LR.fit(X_train, y_train)
# print(LR.coef_)

X_train_temp = X_train.copy()
X_train_temp['intercept'] = [1] * X_train_temp.shape[0]
lr = sm.Logit(y_train, X_train_temp).fit()
summary = lr.summary2()
pvals = lr.pvalues.to_dict()
vars_into_clf = []
for k, v in pvals.items():
    if k == 'intercept': continue
    if v >= 0.1:
        print('注意：{0}的 p-value 为 {1}, 删除'.format(k, v))
    else:
        vars_into_clf.append(k)

X_train_into_clf = train_df_nona[vars_into_clf]
LR = LogisticRegression()
LR.fit(X_train_into_clf, y_train)

test_df_nona = test_df.fillna('bin_nan')
for woe_var in vars_into_clf:
    var = woe_var.replace('_WOE', '')
    test_df_nona[woe_var] = test_df_nona[var].map(WOE_dict[var])
X_test_into_clf, y_test = test_df_nona[vars_into_clf], test_df_nona[target_var]
pred = LR.predict(X_test_into_clf)

# 准确率
acc = accuracy_score(y_test, pred)
print("accuracy_score:", acc)
c_matrix = confusion_matrix(y_test, pred)
print(c_matrix)

tn, fp, fn, tp = c_matrix.ravel()
print(c_matrix)
print('tn={0},fp={1},fn={2},tp={3}'.format(tn, fp, fn, tp))
# 输出预测测试集的概率
y_prb_test = LR.predict_proba(X_test_into_clf)[:, 1]
# 得到误判率、命中率、门限
fpr, tpr, thresholds = roc_curve(y_test, y_prb_test)

# 计算auc
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

ks = max(tpr - fpr)
print("KS:", ks)

# 绘图
plt.plot(fpr, tpr, 'g', label = 'AUC = %0.2f' % (roc_auc))
plt.title('ROC curve')
# 设置x、y轴刻度范围
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.legend(loc = 'lower right')
# 绘制参考线
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')


plt.show()

def Prob2Score(prob, basePoint, PDO):
    # 将概率转化成分数且为正整数
    y = np.log(prob / (1 - prob))
    y2 = basePoint - PDO / np.log(2) * (y)
    scores = y2.astype("int")
    return scores


scores = Prob2Score(y_prb_test, 300, 100)
plt.hist(scores, bins = 100)
plt.show()
>>>>>>> 5286e72 (模型建设完成)
