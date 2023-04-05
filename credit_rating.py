import warnings
import pandas as pd
from auxiliary_functions import BinBadRate, MergeBad0, BadRateMonotone, ChiMerge, AssignBin, Monotone_Merge, \
    BadRateEncoding
import matplotlib.pyplot as plt
import numpy as np
import pickle
import numbers

warnings.filterwarnings('ignore')

total_df = pd.read_csv('D:\study\CDA_data\credit.csv')
target_var = 'default payment next month'
# 这里要额外提醒一下要保证数据用标签y的编码是0/1变量，且1代表违约样本
total_df[target_var] = total_df[target_var].astype(int)
train_df = total_df.loc[total_df['ID'] <= 20000]
test_df = total_df.loc[total_df['ID'] > 20000]
# print('训练集和测试集的数量分别为:', train_df.shape[0], test_df.shape[0])
drop_cols = ['ID']
retained_vars = [col for col in train_df.columns if not col in drop_cols and col != target_var]


# print('共有', len(retained_vars), '个原始指标')


def identify_large_percent_single_value_cols(train_data, retained_vars, large_percent_threshold = 0.9, show_num = 0):
    records_count = train_data.shape[0]
    col_most_values, col_large_value = {}, {}
    for col in retained_vars:
        # 计算每个变量的每个取值对应的数量，
        value_count = train_data[col].groupby(train_data[col]).count()
        # 计算最大数量对应的最大比例，并记录在col_most_values中
        col_most_values[col] = max(value_count) / records_count
        # 获得对应于最大比例的变量取值，并记录在col_large_value中
        large_value = value_count[value_count == max(value_count)].index[0]
        col_large_value[col] = large_value
    # 将col_most_values里的变量按照最大比例排序
    col_most_values_df = pd.DataFrame.from_dict(col_most_values, orient = 'index')
    col_most_values_df.columns = ['max percent']
    col_most_values_df = col_most_values_df.sort_values(by = 'max percent', ascending = False)
    # 如果要打印出“最大比例”最高的show_num个变量的话：将它们的“最大比例”打印出来
    if show_num > 0:
        pcnt = list(col_most_values_df[:show_num]['max percent'])
        vars = list(col_most_values_df[:show_num].index)
        plt.bar(range(len(pcnt)), height = pcnt)
        plt.title('Largest Percentage of Single Value in Each Variable')
        plt.show()
    large_percent_single_value_cols = list(
        col_most_values_df[col_most_values_df['max percent'] >= large_percent_threshold].index)
    return large_percent_single_value_cols


large_percent_single_value_cols = identify_large_percent_single_value_cols(train_df, retained_vars,
                                                                           large_percent_threshold = 0.9, show_num = 20)
retained_vars = [var for var in retained_vars if var not in large_percent_single_value_cols]


# print('识别出了', len(large_percent_single_value_cols), '个区别度过低的指标')
# print('剩余', len(retained_vars), '个指标')


def identify_large_percent_missing_value_cols(train_data, retained_vars, missing_pcnt_threshould = 0.8):
    large_percent_missing_value_cols = []
    for var in retained_vars:
        # 计算每个变量的缺失值个数
        missing_vals = train_data[var].map(lambda var: int(np.isnan(var)))
        # print(sum(missing_vals))
        # 计算每个变量的缺失值比例
        missing_rate = sum(missing_vals) * 1.0 / train_data.shape[0]
        if missing_rate > missing_pcnt_threshould:
            large_percent_missing_value_cols.append(var)
    return large_percent_missing_value_cols


large_percent_missing_value_cols = identify_large_percent_missing_value_cols(train_df, retained_vars)
retained_vars = [var for var in retained_vars if var not in large_percent_missing_value_cols]


# print('识别出了', len(large_percent_missing_value_cols), '个缺失率过高的指标')
# print('剩余', len(retained_vars), '个指标')


def identify_num_cat_vars(train_data, retained_vars):
    '''
    将变量分为数值型和类别型
    '''
    numerical_vars = []
    constant_vars = []
    for col in retained_vars:
        # 获得每个变量的取值集合
        uniq_valid_vals = [i for i in train_data[col] if i == i]
        uniq_valid_vals = list(set(uniq_valid_vals))
        # 若每个变量的取值数量不小于10，且取值为实数，那么算作数值型
        if len(uniq_valid_vals) >= 10 and isinstance(uniq_valid_vals[0], numbers.Real):
            numerical_vars.append(col)
    # 类别型变量为除掉数值型剩余的变量
    categorical_vars = [i for i in retained_vars if i not in numerical_vars]
    return numerical_vars, categorical_vars


numerical_vars, categorical_vars = identify_num_cat_vars(train_df, retained_vars)


# print('类别型变量有:', categorical_vars)

def identify_small_large_cat_vars(train_data, categorical_vars):
    small_cat_vars, large_cat_vars = [], []
    for var in categorical_vars:
        # 若取值数大于5，则算作取值较多的类别型变量
        if not len(train_data[var].unique()) > 5:
            small_cat_vars.append(var)
        else:
            large_cat_vars.append(var)
    return small_cat_vars, large_cat_vars


small_cat_vars, large_cat_vars = identify_small_large_cat_vars(train_df, categorical_vars)


# print('取值少的类别型变量有:', small_cat_vars)
# print('取值多的类别型变量有:', large_cat_vars)

def bining_small_cat_vars(train_data, target_var, small_cat_vars):
    needbe_merged_bin_dict = {}  # 存放需要合并的变量以及其合并方法
    for var in small_cat_vars:
        # print('为指标', var, '分箱中')
        bin_br = BinBadRate(train_data, var, target_var)[0]
        if min(bin_br.values()) == 0:  # 由于某个取值没有坏样本而进行合并
            # print(var, '由于0违约样本需要被合并优化')
            # 利用MergeBad0实现对变量var的分箱合并
            combine_bin = MergeBad0(train_data, var, target_var, direction = 'bad')
            # 分箱合并策略被存储到needbe_merged_bin_dict中，以便对新数据实现相同的分箱合并策略
            needbe_merged_bin_dict[var] = combine_bin
            # 经过分箱合并的分箱后变量设定一个新的变量名，这里以'_smallcatBin'作为后缀，所以在分箱后的数据集中如果有个变量是这个后缀的话，说明它是由取值少的变量得来的”
            new_var = var + '_smallcatBin'
            # 根据分箱合并策略实现分箱
            train_data[new_var] = train_data[var].map(combine_bin)
            continue
        '''
        下面两个操作与上面的类似
        '''
        if max(bin_br.values()) == 1:  # 由于某个取值没有好样本而进行合并
            # print(var, '由于0非违约样本需要被合并优化')
            combine_bin = MergeBad0(train_data, var, target_var, direction = 'good')
            needbe_merged_bin_dict[var] = combine_bin
            new_var = var + '_smallcatBin'
            train_data[new_var] = train_data[var].map(combine_bin)
            continue
        new_var = var + '_smallcatBin'
        train_data[new_var] = train_data[var]
    return train_data, needbe_merged_bin_dict


# 实现对取值少的变量进行分箱
binned_train_df, needbe_merged_bin_dict = bining_small_cat_vars(train_df, target_var, small_cat_vars)


# print(list(binned_train_df.columns))

def map_large_cat_vars_to_badrate_vars(train_data, target_var, large_cat_vars):
    br_encoding_dic = {}
    br_encoded_cat_vars = []
    var_num = 0
    for var in large_cat_vars:
        print('为指标', var, '数值编码（违约率）中')
        # 为违约率编码后的变量取一个新名字
        br_var = str(var) + '_br_encoding'
        # 利用BadRateEncoding获得对变量var的违约率编码信息，其中包含了已经编码后的变量'encoding'，以及编码对应的违约率词典'bad_rate'
        encoding_result = BadRateEncoding(train_data, var, target_var)
        # 将编码后的放进数据里，将编码对应的违约率词典保存到br_encoding_dic里。
        train_data[br_var], br_encoding_dic[var] = encoding_result['encoding'], encoding_result['bad_rate']
        # 记录数据里哪些变量名是属于“违约率编码后的变量”
        br_encoded_cat_vars.append(br_var)
    return train_data, br_encoding_dic, br_encoded_cat_vars


# 实现对取值较多的类别型变量做违约率编码
binned_train_df, br_encoding_dic, BRencoded_cat_vars = map_large_cat_vars_to_badrate_vars(binned_train_df, target_var,
                                                                                          large_cat_vars)


# 下面的函数实现递增方式来进行对连续型变量以及违约率编码后的变量实现卡方分箱，这是因为卡方分箱是一个相对于耗时的操作，所以实现以支持支持小批量递增实现的方式
def incre_bining_numerical_vars(train_data, target_var, numerical_vars, special_attribute, var_cutoff):
    # 因为是递增的方式，所以每次先获取一下哪些变量已经做好了分箱
    binned_numBin_vars = set([n for n in train_data.columns if n.find('_numBin') != -1])
    print('已经做好分箱的指标数:', len(binned_numBin_vars))
    var_num = 0
    for var in numerical_vars:
        var_num += 1
        # 数值型变量经过分箱后产生的新的变量的名字后缀为'_numBin'
        new_var = str(var) + '_numBin'
        if new_var in binned_numBin_vars: continue
        # 下面的代码对某个还没做好分箱的连续型变量进行分箱
        print("为第", str(var_num) + '/' + str(len(numerical_vars)), '个指标', var, '做分箱中')
        # 获得对应于卡方分箱方法的分箱策略（以切分点的形式表达）
        cutOffPoints = ChiMerge(train_data, var, target_var, special_attribute = special_attribute)
        # 保存切分点以实现对新的数据进行分箱
        var_cutoff[var] = cutOffPoints
        # 利用切分点以实现对变量分箱，并放到数据里（用新的变量名）
        train_data[new_var] = train_data[var].map(
            lambda x: AssignBin(x, cutOffPoints, special_attribute = special_attribute))
        # 利用辅助函数BadRateMonotone对单调性进行检查
        BRM = BadRateMonotone(train_data, new_var, target_var, special_attribute = special_attribute)
        # 不满足单调性则用Monotone_Merge进行合并
        # 这一段相对于比较复杂，阅读它请先熟悉分箱为单调性进行合并的算法以及熟悉pandas等工具包的操作，不然可以将其看做一个辅助函数的实现
        if not BRM:
            if special_attribute == []:
                bin_merged = Monotone_Merge(train_data, target_var, new_var)
                removed_index = []
                for bin in bin_merged:
                    if len(bin) > 1:  # 若发现发生了分箱合并
                        indices = [int(b.replace('Bin ', '')) for b in bin]  # 记录合并的分箱索引
                        removed_index = removed_index + indices[0:-1]  # 计划将前面n-1个索引移除掉
                # 移除因为分箱合并而需要移除的分箱对应的切分点
                removed_point = [cutOffPoints[k] for k in removed_index]
                for p in removed_point:
                    cutOffPoints.remove(p)
                # 将更新后的切分点覆盖之前的切分点
                var_cutoff[var] = cutOffPoints
                # 用更新后的切分点实现分箱
                train_data[new_var] = train_data[var].map(
                    lambda x: AssignBin(x, cutOffPoints, special_attribute = special_attribute))
            else:  # 下面的操作类似于上面的
                cutOffPoints2 = [i for i in cutOffPoints if i not in special_attribute]
                temp = train_data.loc[~train_data[var].isin(special_attribute)]
                bin_merged = Monotone_Merge(temp, 'target', col1)
                removed_index = []
                for bin in bin_merged:
                    if len(bin) > 1:
                        indices = [int(b.replace('Bin ', '')) for b in bin]
                        removed_index = removed_index + indices[0:-1]
                removed_point = [cutOffPoints2[k] for k in removed_index]
                for p in removed_point:
                    cutOffPoints2.remove(p)
                cutOffPoints2 = cutOffPoints2 + special_attribute
                var_cutoff[var] = cutOffPoints2
                train_data[new_var] = train_data[var].map(
                    lambda x: AssignBin(x, cutOffPoints2, special_attribute = special_attribute))
    return train_data, var_cutoff


# num_bin_var_cutoff = {}
# numbin_numerical_vars = numerical_vars
# numbin_numerical_vars.extend(BRencoded_cat_vars)
binned_train_df = pickle.load(open(r'binned_train_df.pkl', 'rb'))
num_bin_var_cutoff = pickle.load(open(r'num_bin_var_cutoff.pkl', 'rb'))
numbin_numerical_vars = pickle.load(open(r'numbin_numerical_vars.pkl', 'rb'))

interval = 10  # 递增运行的时候，每次运行10个变量上的卡方分箱
for begin_index in range(1, 5):
    end_index = begin_index * interval  # 即此次的运行将覆盖变量序列里的索引从0到(end_index-1)对应的变量
    print('此次迭代将为第', end_index, '之前的指标进行分箱')
    # 对索引从0到(end_index-1)对应的变量尝试做分箱（不过其中很多变量在以前的运行中已经做好了分箱了，函数中将会发现它们并越过它们，从而只对其中没做过分箱的变量做分箱）
    binned_train_df, num_bin_var_cutoff = incre_bining_numerical_vars(binned_train_df, target_var,
                                                                      numbin_numerical_vars[:end_index],
                                                                      special_attribute = [],
                                                                      var_cutoff = num_bin_var_cutoff)
    # 因为是递增的运行方式，所以需要保存每次运行的结果
    pickle.dump(numbin_numerical_vars, open('numbin_numerical_vars.pkl', 'wb'))
    pickle.dump(binned_train_df, open('binned_train_df.pkl', 'wb'))
    pickle.dump(num_bin_var_cutoff, open('num_bin_var_cutoff.pkl', 'wb'))

bin_strategies = {}
bin_strategies['needbe_merged_bin_dict'] = needbe_merged_bin_dict  # 保存取值少的类别型变量的分箱合并策略（只对应那些需要进行分箱合并的变量）
print('需要分箱的指标名:', needbe_merged_bin_dict)
bin_strategies['br_encoding_dic'] = br_encoding_dic  # 保存取值多的类别型变量对应的违约率编码策略
bin_strategies['num_bin_var_cutoff'] = num_bin_var_cutoff  # 保存数值型变量分箱对应的切分点（即卡方分箱策略）
bin_strategies['numerical_vars'] = numerical_vars  # 保存哪些变量是数值型变量的信息
bin_strategies['large_cat_vars'] = large_cat_vars  # 保存哪些变量是取值多的类别型变量的信息
bin_strategies['small_cat_vars'] = small_cat_vars  # 保存哪些变量是取值少的类别型变量的信息

pickle.dump(bin_strategies, open('Preprocessing_bin_strategies_uci.pkl', 'wb'))

# 先拿到所有原始变量对应的变量名信息
all_original_vars = [n for n in train_df.columns if n.find('Bin') == -1 and n.find('br_encoding') == -1]
all_original_vars.remove(target_var)


# 下面的函数支持对新的数据利用bin_strategies包含的信息做分箱
def bining_new_data(new_data, all_original_vars, bin_strategies):
    needbe_merged_bin_dict = bin_strategies['needbe_merged_bin_dict']
    br_encoding_dic = bin_strategies['br_encoding_dic']
    num_bin_var_cutoff = bin_strategies['num_bin_var_cutoff']
    numerical_vars = bin_strategies['numerical_vars']
    large_cat_vars = bin_strategies['large_cat_vars']
    small_cat_vars = bin_strategies['small_cat_vars']
    for var in all_original_vars:
        # 如果为取值少的类别型变量，则要拿到记录了分箱合并策略的needbe_merged_bin_dict实现分箱
        if var in small_cat_vars:
            if var in needbe_merged_bin_dict:  # 此变量是需要进行分箱合并的
                combine_bin = needbe_merged_bin_dict[var]  # 拿到分箱合并策略
                new_data[var + '_smallcatBin'] = new_data[var].map(combine_bin)  # 分箱
            else:
                new_data[var + '_smallcatBin'] = new_data[var]  # 否则，用它本身作为分箱后的变量
            continue
        # 如果为取值多的类别型变量，则要拿到记录了违约率编码的br_encoding_dic先做违约率编码，然后用记录了卡方分箱策略的num_bin_var_cutoff进行分箱
        if var in large_cat_vars:
            br_dict = br_encoding_dic[var]  # 拿到违约率编码词典
            new_data[var + '_br_encoding'] = new_data[var].apply(
                lambda x: br_dict[x] if x in br_dict.keys() else np.nan)  # 做违约率编码
            cutOffPoints = num_bin_var_cutoff[var + '_br_encoding']  # 用卡方分箱策略对违约率编码后的类别型变量（此时相当于一个数值型变量）做分箱
            new_data[var + '_br_encoding' + '_numBin'] = new_data[var + '_br_encoding'].map(
                lambda x: AssignBin(x, cutOffPoints))  # 做卡方分箱
            continue
        # 如果为取值多的类别型变量，则用记录了卡方分箱策略的num_bin_var_cutoff进行分箱
        if var in numerical_vars:
            cutOffPoints = num_bin_var_cutoff[var]
            new_data[var + '_numBin'] = new_data[var].map(lambda x: AssignBin(x, cutOffPoints))
            continue
    return new_data


# 对测试数据利用bin_strategies包含的信息做分箱
binned_test_df = bining_new_data(test_df, all_original_vars, bin_strategies)
pickle.dump(binned_train_df,open('Preprocessing_binned_train_df_uci.pkl','wb'))
pickle.dump(binned_test_df,open('Preprocessing_binned_test_df_uci.pkl','wb'))
binned_vars = [v for v in binned_train_df.columns if v.find('Bin')!=-1]
pickle.dump([all_original_vars,binned_vars],open('original_and_binned_vars_uci.pkl','wb'))