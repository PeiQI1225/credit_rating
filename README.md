# credit_rating
# 信用评分建模基础知识

## 信用评分背景介绍

### **信用评分的定义**

- 信用评分指的是向某一特定消费者发放贷款的风险。
- 信用评分主要被用于决策是否要给某个新的申请人贷款，其余的一些用途包括如何管理管理现有客户；包括是否要给现有客户增加信用额度等等

### **信用评分的历史**

20世纪30年代：为了使授信质量一致，开始找有经验的风险分析专家设计信用判断条件

- 20世纪50年代：回归分析等统计技术开始运用于信用评分，将人类经验与数学实证进行了结合。
- Now：信用评分模型已成为银行非常重要的风险评估工具。

### **信用评分的优点**

- 以科学方法将风险模式数据化
- 提供客观风险量尺，减少主观判断
- 提高风险管理效率，节省人力成本

### 常用评分类型

**申请评分**

- 用在贷前审核环节
- 主要用于评估放贷后是否会违约

**行为评分**

- 用在贷后监控环节
- 主要用于早期预警

**催收评分**

- 用在发生逾期后的管理环节
- 主要用于为催收工作提供指导

### 科学性体现

**建立在历史数据之上**

- 借由数据汇整、清理、分组及探勘等技术，从数据中总结出有用的风险因子。
- 借由近期申请同一产品的老刻画样本学习模型，即历史数据在某种程度 上具有与当前或未来数据的相似性。

> 如果因为是新产品或只有极少数消费者使用过目标产品，则可以根据少量样本或同类产品样本基于机器学习里的相关理论(例如迁移学习等)学习模型。

**建立在统计学、数据挖掘、机器学习等理论基础上**

- 上述理论中的分类、回归等方法帮助我们在信用评分的背景下对所使用的特征变量进行筛选和排序，并对所建立的模型的判别能力进行评估。
- 信用评分模型建立后，可将风险数据化。此时，可清楚地呈现客户违约概率及风险排序，使风险管理单位得以确切掌握客户风险，并且制定更为精准的授信政策。

## 信用评分案例（国外）

### FICO评分系统

美国应用最广泛的是由Fair Isaac公司推出的FICO信用评分模型

Fair Isaac公司开发了三种不同的FICO评分系统，三种评分系统分别由美国的三大信用管理局使用:

| Equifax(艾可菲)  | BEACON*                        |
| ---------------- | ------------------------------ |
| Experian(益博睿) | ExperianPFair Isaac Risk Model |
| TransUnion(全联) | FICO Risk Score,Classic        |

FICO评分系统得出的信用分数范围在300-850分之间。分数越高，说明客户的信用风险越小。贷款方通常会将分数作为参考:

\>680分：信用显著；620~680：作一进步的分析；<620分：增加担保/拒绝贷款

### 大数据风控

随着大数据时代来临，数据的维度和量级都呈现快速增长，这个时候有一些科技公司就在大数据的基础上面进行风控建模。例如：

- Zest Finance开发10个机器学习模型，1万条原始信息，7万个特征变量，5秒内完成。

> 利用机器学习和大数据技术，创立了一套和传统模式不太一样的信用评分方式，应用到的数据变量是传统模式的上百倍且采用非线性化的更前沿的技术进行分析，从而防止模型套利现象的出现，更准确的评估消费者信用风险

- Kabbage公司通过获取企业网店店主的销售、信用记录、顾客流量、评论、商品价格和存货等信息、以及在Facebook和Twitter上与客户的互动信息，借助数据挖掘技术，把这些店主分成不同的风险等级。

> 相对于Finance，Kabbage显得更为动态，因为它是通过对网商销售情况和资金流向的实时掌控，所以他能在第一时间对现金流紧张的网商做出预警，提高关注级别

## 信用评分案例（国内）

### 二代征信系统的个人信用评分

央行上线的二代征信报告中，将新增“个人信用报告数字解读”，推出针对个人的“信用评分”并给出该评分所处的“相对位置”

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=YTE4NWU0MzIwMjY1YjI1Yjc2NzYxMDI3YzZjNDA3ODlfUklxeTloZmwzS0ZSdHpPeTJWYzAyNEFxU21DVmpkc3JfVG9rZW46Ym94Y25WRFZCUjc4WENFQTV1OHRmMHRabzdlXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

> 个人信用报告解读栏推出了针对个人信用评分并月给出了该评分所处的相对位置。对于借贷机构用自己的一些数据来做评分，央行的评分所考虑的元素更多

综合考虑因素更多，更具权威性。

同样的我们国内也有类似Zest Finance的大数据风控公司

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=MjRhMDlkODg3N2QwODJjMTRjODRkOGUyYTIyZDFiZDZfelp2M2ZBSDBnYjB2ZTZiMjRacjJOcUV2R1F0UnJGUlRfVG9rZW46Ym94Y25nVDVxd2lvZzIxTjR4V21TNHRDRVZoXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

## 信用评分建模框架

![image-20230406203210592](C:\Users\佩棋\AppData\Roaming\Typora\typora-user-images\image-20230406203210592.png)

准备期期间，主要是定义一个任务，然后准备建模所需要的数据，有了任务和数据我们就可以进入包含数据分析、数据预处理、转换和划分，划分训练集、测试集、验证集，选择合适的变量和模型，进入模型学习与规划的模型期，模型检验的检验期（一般模型期和检验期都不止一遍），通过优化期针对模型的具体表现，对模型进行优化调整

### 信用评分建模前的准备(准备期）

**任务定义**

- 确定项目目标(通过模型判断客户风险程度，进而提高业务量，利润等)
- 确定观察期与表现期、违约及不确定的定义等

![image-20230406203240480](C:\Users\佩棋\AppData\Roaming\Typora\typora-user-images\image-20230406203240480.png)

> 假设我们需要生成一个带标签的样本（也就说我们要去对每个客户去确定他是不是一个违约客户），所谓的观察期就是指的就是这个申请时间点往前的一段时间区间，用来生成用户特征，所以这个观察期的区间不宜过长，也不宜太短；表现期指的是这个申请点往后的一段时间区间，用于定义用户是否不违约的区间，生成标签

关于违约的定义，常常根据用户发生M1+(即逾期超过一个月)，M2+，M3+的行为特点来定义。

- 确定项目规划(时程、成本、交付文件及格式、模型测试标准、项目验收标准等)

### 信用评分建模前的准备(数据期)

**收集数据**

通过不同的评分类型选择数据源进行数据收集

- 申请评分

△常用数据：个人信息、央行征信信息、申请行为信息、其他辅助信息

- 行为评分

△常用数据：贷后的还款行为、消费行为等。

- 催收评分

△常用数据：个人信息、贷后的还款行为、消费行为、联系人信息等

**数据质量把控**

- 是否有足够的违约样本

> 一般来说，违约样本对于总体样本的比例较小，这个在建模的时候可以利用到损失敏感、重采样本的方法进行处理，具体应该保证的数量要和多个因素有关

- 数据是否有代表性

> 我们统计建模是需要我们构建模型的训练数据中所包含的规律，和我们的测试数据中要发生的规律是有一定的关联的。如果模型代表性欠佳，我们可以使用迁移学习对模型进行优化

- 数据数值是否准确

> 对数据进行验证

# 模型建立---分箱操作

## 数据介绍

我们选取UIC上的一组公开数据进行程序的编写

> 数据源：https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=M2UwMWE0ZGM0OGYwNmRhZmRjZGZkYzA1YTk0MGY1M2NfNFVlTUpCY3lkYWw5bkhtR3dTZUVoUkw4dHdoOGtOREVfVG9rZW46Ym94Y25nQllpdzB1c0NpUEE5TTlXSG9wNk5mXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

数据源上的数据涵盖了贷款金额、性别、教育水平、婚姻状况、年龄……字段，数据既有类别型的也有数值型的

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=YzAwNzg5NmE4YmVhNWM1NGRkYTU5YzM1OWUwMzU0MDJfQ29iTmduV3M5TGRZM3NoanlkUU5FSWxmeVg4RllMaW5fVG9rZW46Ym94Y25GVjNlQ0xlRUFacEpHbHpZV0Y3Z0MzXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

> 标签数据中0代表未违约，1代表违约

我们通过程序对数据进行读取

```Python
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

total_df = pd.read_csv('D:\study\CDA_data\credit.csv')
target_var = 'default payment next month'
# 这里要额外提醒一下要保证数据用标签y的编码是0/1变量，且1代表违约样本
total_df[target_var] = total_df[target_var].astype(int)
train_df = total_df.loc[total_df['ID'] <= 20000]
test_df = total_df.loc[total_df['ID'] > 20000]
print('训练集和测试集的数量分别为:', train_df.shape[0], test_df.shape[0])
drop_cols = ['ID']
retained_vars = [col for col in train_df.columns if not col in drop_cols and col != target_var]
print('共有', len(retained_vars), '个原始指标')
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=ODVjNWUxYmMxYWRhZmY1MWRkOGNjODhkNjEyNGNlYTNfN2YxeTFsakQ0YTQ2ZG9ldHhVbGFYSmRGeHc5R1lvT2RfVG9rZW46Ym94Y25MdHJJT0duakZoSTEwMUFiMkxoUzJnXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

## 初步筛选数据

一些与运算相关的代码单独归置出来了

```Python
from auxiliary_functions import BinBadRate, MergeBad0, BadRateMonotone, ChiMerge, AssignBin, Monotone_Merge, BadRateEncoding
```

- BinBadRate: 计算每个指标里每个分箱的违约率，这将能够帮助我们对指标进行违约率编码         
- MergeBad0：分箱之后为了计算WOE,IV值，是需要每个分箱同时具备违约和非违约样本的，这个函数负责做这个检查，并在不满足的时候做分箱合并。         
- BadRateEncoding：通过搭配BinBadRate函数，实现对每个指标进行违约率编码         
- ChiMerge：这个函数对应于课程中介绍的卡方分箱，是相对于实现较为复杂的函数。阅读它请先熟悉算法以及熟悉pandas等工具包的操作。具体帮助阅读的注释请参考ChiMerge代码里的注释。    
- Chi2：根据卡方值的公式（请参考课件）计算某种分箱状态下的卡方值         
- AssignBin：根据输入的切分点实现对某个指标的某个取值分配分箱         
- AssignGroup: 获得某个变量的某个取值在输入的分箱策略下对应的分箱         
- Monotone_Merge：这个函数对应于课程中介绍到的根据单调性要求进行分箱合并，也相对于复杂。其具体功能是：将数据集df中，不满足坏样本率单调性的变量col进行合并，使得合并后的新的变量中，坏样本率单调，输出合并方案。阅读它请先熟悉分箱合并的算法思想以及熟悉pandas等工具包的操作。具体帮助阅读的注释请参考Monotone_Merge代码里的注释。         
- BadRateMonotone：检验某个指标的分箱是否满足单调性要求         

### 考虑利用"最大比例的值对应的比例"

```Python
def identify_large_percent_single_value_cols(train_data, retained_vars, large_percent_threshold=0.9, show_num = 0):
    records_count = train_data.shape[0]
    col_most_values,col_large_value = {},{}
    for col in retained_vars:
        # 计算每个变量的每个取值对应的数量
        value_count = train_data[col].groupby(train_data[col]).count()
        # 计算最大数量对应的最大比例，并记录在col_most_values中
        col_most_values[col] = max(value_count)/records_count
        # 获得对应于最大比例的变量取值，并记录在col_large_value中
        large_value = value_count[value_count== max(value_count)].index[0]
        col_large_value[col] = large_value
    # 将col_most_values里的变量按照最大比例排序
    col_most_values_df = pd.DataFrame.from_dict(col_most_values, orient = 'index')
    col_most_values_df.columns = ['max percent']
    col_most_values_df = col_most_values_df.sort_values(by = 'max percent', ascending = False)
    # 如果要打印出“最大比例”最高的show_num个变量的话：将它们的“最大比例”打印出来
    if show_num>0:
        pcnt = list(col_most_values_df[:show_num]['max percent'])
        vars = list(col_most_values_df[:show_num].index)
        plt.bar(range(len(pcnt)), height = pcnt)
        plt.title('Largest Percentage of Single Value in Each Variable')
        plt.show()
    large_percent_single_value_cols = list(col_most_values_df[col_most_values_df['max percent']>=large_percent_threshold].index)
    return large_percent_single_value_cols

large_percent_single_value_cols = identify_large_percent_single_value_cols(train_df,retained_vars, large_percent_threshold=0.9, show_num = 20)
retained_vars = [var for var in retained_vars if var not in large_percent_single_value_cols]
print('识别出了', len(large_percent_single_value_cols), '个区别度过低的指标')
print('剩余', len(retained_vars), '个指标')
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NzA0NjNlYTQ5YTg5ZTExODRkYTQ3OWVhNjBlNWRlM2JfMEVKNzJrMDFIUHNlM0FidnhsVm52NENRMDFHc01nNTdfVG9rZW46Ym94Y25VaTBJSkhlQXpaQ3pyRXBPQ2c5MWt0XzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=YzhmYWZlZDhiZWI4ZWZkNzhhYzJhOTAwN2QzYTIxZDZfSjZ0dXlkWm1DRWlCdHdmMDB1dmxmR0xuVDZnVk43RnlfVG9rZW46Ym94Y242c2ZTWTBxbU9OaklvMHFYUWpwWkRiXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

可以看出数据里面并不包含区别度特别低的指标，通过图我们可以看出区别度最高才0.6

### 通过缺失值占比来筛选变量

```Python
def identify_large_percent_missing_value_cols(train_data, retained_vars,missing_pcnt_threshould=0.8):
    large_percent_missing_value_cols = []
    for var in retained_vars:
        # 计算每个变量的缺失值个数
        missing_vals = train_data[var].map(lambda var: int(np.isnan(var)))
        #print(sum(missing_vals))
        # 计算每个变量的缺失值比例
        missing_rate = sum(missing_vals) * 1.0 / train_data.shape[0]
        if missing_rate > missing_pcnt_threshould:
            large_percent_missing_value_cols.append(var)
    return large_percent_missing_value_cols

large_percent_missing_value_cols = identify_large_percent_missing_value_cols(train_df,retained_vars)
retained_vars = [var for var in retained_vars if var not in large_percent_missing_value_cols]
print('识别出了', len(large_percent_missing_value_cols), '个缺失率过高的指标')
print('剩余', len(retained_vars), '个指标')
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=YTg1Njg1MmJiYTRlNjliYzU0ODMxNWM1YTgyNDcxZjZfWEROVWppa2p3R05DZmpEV1RNVnhUZTRxUHdzYXFNMnFfVG9rZW46Ym94Y25tUlFjZjlxN0Z0Rng0NFFuajFGRWViXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

## 数据类别划分

为了后续分箱工作，我们首先需要将保留的变量分为类别型和数值型

```Python
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
print('类别型变量有:', categorical_vars)
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NzA3MDM2YWVlMjU2YzE2MWJlN2Q3YmExYWY1OTU3YTlfR05sV2hPaU9aOGJuUTNiZVJuYmE1V25QR0syaTg0bXVfVG9rZW46Ym94Y25URDVJN21nVE1KV0RoaWhDR1JpQmZmXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

对于类别型变量我们还需要区分取值比较少的还是取值比较多的，因为取值较少的变量有时候是不需要分箱的，所以对于类别型变量还将分为取值较少和取值较多两种

```Python
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
print('取值少的类别型变量有:', small_cat_vars)
print('取值多的类别型变量有:', large_cat_vars)
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NTFlODQwYjM1NmUyOTE1N2JmYjc5OGFhMmJlMDM3ZTNfcERwV1pRMHpKdklJMjdEczRJUDU2RFZCd3FRbVNJeXlfVG9rZW46Ym94Y25wSE1WNGJERVZnWkNjbTV6aUZGYmxjXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

## 对取值少的离散型变量分箱

对于取值少的离散型变量一般情况下是不需要实行分箱操作的，除非变量里面有取值对应到的样本集合里面不包含非违约样本或不包含违约样本

```Python
def bining_small_cat_vars(train_data, target_var, small_cat_vars):
    needbe_merged_bin_dict = {}  # 存放需要合并的变量以及其合并方法
    for var in small_cat_vars:
        print('为指标', var, '分箱中')
        bin_br = BinBadRate(train_data, var, target_var)[0]
        if min(bin_br.values()) == 0:  # 由于某个取值没有坏样本而进行合并
            print(var, '由于0违约样本需要被合并优化')
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
            print(var, '由于0非违约样本需要被合并优化')
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
print(list(binned_train_df.columns))
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDY4N2Y3Y2FhOTIyY2U5NGZmODk4NTc2ZmI2ZjVmYTJfVm5YcXpLUTdhRXVWUXRGcUl1U0t6WEg1NmhrRlRETnpfVG9rZW46Ym94Y25jcThJSUtSdUtHcW1iNFFld1VxQUhoXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

## 对取值多的离散型变量分箱

卡方分箱是针对有序型变量进行操作的，因此我们需要对取值多的离散型变量进行编码，后进行分箱

### 先数值编码

针对于取值较多的类别型变量做违约率编码编码策略（针对于需要合并优化的变量）被保存在br_encoding_dic中可使用new_data[var].apply(lambda x: br_dict[x] if not np.isnan(x) else x)来对新样本对应变量进行编码产生的编码后新变量被保存在BRencoded_cat_vars中

```Go
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
```

### 对连续型变量以及违约率编码后的新变量做卡方分箱

针对于连续型变量以及违约率编码后的新变量做卡方分箱，编码策略（针对于需要合并优化的变量）被保存在num_bin_var_cutoff中可使用new_data[var].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))来对新样本对应变量进行编码

```Python
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
```

下面的3+3行代码分别用于初次分箱和非初次递增分箱

- 初次分箱使用前三行，注释掉后三行
- 非初次分箱则注释掉前三行，使用后三行

```Python
num_bin_var_cutoff = {}
numbin_numerical_vars = numerical_vars
numbin_numerical_vars.extend(BRencoded_cat_vars)
# binned_train_df = pickle.load(open(r'binned_train_df.pkl', 'rb'))
# num_bin_var_cutoff = pickle.load(open(r'num_bin_var_cutoff.pkl', 'rb'))
# numbin_numerical_vars = pickle.load(open(r'numbin_numerical_vars.pkl', 'rb'))

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
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=OGU0YjU2NDljMjA1MjUxZDU5N2Q3Y2I0MTJkYTEwZjhfMU95QURNMGN1UG94NW9lWUtQZ3d6U1VqREZLMEFzUjZfVG9rZW46Ym94Y244eWd5WWZBRUZZSmZtNGhmc2hvZHlkXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=OTFhZGQxZDlhMmMwNzVkYzZhZWNiMjdmYmU3NWNlZjVfOWoyS2dVUDNsblZSTDl6MjRxTUJObFVxZkRobkhlTXlfVG9rZW46Ym94Y250N3F5cDZ1TW5tZkRQcHFEbzdtM2liXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

## 保留分箱策略

对所有初检验后的变量进行好分箱操作后，我们需要把分箱的策略进行保留，以便对后来的数据实现分箱

```Python
bin_strategies = {}
bin_strategies['needbe_merged_bin_dict'] = needbe_merged_bin_dict  # 保存取值少的类别型变量的分箱合并策略（只对应那些需要进行分箱合并的变量）
print('需要分箱的指标名:', needbe_merged_bin_dict)
bin_strategies['br_encoding_dic'] = br_encoding_dic  # 保存取值多的类别型变量对应的违约率编码策略
bin_strategies['num_bin_var_cutoff'] = num_bin_var_cutoff  # 保存数值型变量分箱对应的切分点（即卡方分箱策略）
bin_strategies['numerical_vars'] = numerical_vars  # 保存哪些变量是数值型变量的信息
bin_strategies['large_cat_vars'] = large_cat_vars  # 保存哪些变量是取值多的类别型变量的信息
bin_strategies['small_cat_vars'] = small_cat_vars  # 保存哪些变量是取值少的类别型变量的信息

pickle.dump(bin_strategies, open('Preprocessing_bin_strategies_uci.pkl', 'wb'))
```

## 对测试数据进行分箱

```Python
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
```

## 存储分箱后的数据

```Python
pickle.dump(binned_train_df,open('Preprocessing_binned_train_df_uci.pkl','wb'))
pickle.dump(binned_test_df,open('Preprocessing_binned_test_df_uci.pkl','wb'))
binned_vars = [v for v in binned_train_df.columns if v.find('Bin')!=-1]
pickle.dump([all_original_vars,binned_vars],open('original_and_binned_vars_uci.pkl','wb'))
```

# 模型建立

## 导入预处理后的数据

```Python
import pandas as pd
import pickle
target_var = 'default payment next month'
train_df=pickle.load(open('Preprocessing_binned_train_df_uci.pkl','rb'))
test_df=pickle.load(open('Preprocessing_binned_test_df_uci.pkl','rb'))
bin_strategies=pickle.load(open('Preprocessing_bin_strategies_uci.pkl','rb'))
[all_original_vars,binned_vars]=pickle.load(open('original_and_binned_vars_uci.pkl','rb'))
```

## 缺失值处理

前面的分箱操作中，我们并未对缺失值进行分箱，是为了后面更灵活的处理缺失值，我们可以将缺失值处理成一个额外的分箱，如果缺失值分箱过程中未包含违约或非违约样本的话，我们需要进行分箱优化

> 一般采用分箱合并来保证对应的变量同时含有违约样本和非违约样本，但是**该数据集不含缺失值**，因此不进行分箱操作

```SQL
train_df_nona = train_df.fillna('bin_nan')
```

## 计算每个变量的WOE编码和IV值

```Python
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

for iv_name, iv_value in IV_dict_sorted[:10]:
    print(iv_value, iv_name)
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NTEwOTc5ZWYzMGU3MmYzOGY0MzYzZTgyMGNiZjJlYjFfcFhXRHpxZUV1SzVmREIzeExhWUtEOFV4dnJNR1BoUElfVG9rZW46WVR2dWJ3MER5bzNOOXZ4UElJeGNYYlhablhjXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

## **利用IV值进行变量筛选，对于IV值满足一定要求的变量进行WOE编码**

我们将计算好的IV值进行一个筛选，选择IV值不小于设定阈值的变量进行保留，对于保留下来的变量，用对应的WOE词典做WOE编码，形成一个新的变量加入到数据中

```Python
iv_threshould = 0.1
var_IV_selected = {k: IV_dict[k] for k in IV_dict.keys() if not IV_dict[k] < iv_threshould}
print(var_IV_selected)
var_IV_selected_sorted = sorted(var_IV_selected.items(), key = lambda d: d[1], reverse = True)
var_IV_selected_sorted = [i[0] for i in var_IV_selected_sorted]
for var in var_IV_selected_sorted:
    woe_var = var + '_WOE'
    train_df_nona[woe_var] = train_df_nona[var].map(WOE_dict[var])
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=MTRiNDdlYmFjNzhlZGFlNGQ2Y2FmZDAyZDA3NjQyNzNfUTlJYXNES2hSUlhxa0U1YmtaTGZjb0ZYN3lXQnlQUHBfVG9rZW46RllPOWJWc2tUb0dCb054NnFJMWM3T3FlbmdkXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

## 通过多变量分析进一步筛选变量

```Python
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
                        print('提示：变量 {0} 与 {1} 之间的相关性为 {2}，删除 {1}'.format(x1, x2, str(cor_value)))
                    else:
                        removed_var.append(var_IV_selected_sorted[i])
                        print('提示：变量 {0} 与 {1} 之间的相关性为 {2}，删除 {0}'.format(x1, x2, str(cor_value)))
retained_woe_vars = [i + "_WOE" for i in var_IV_selected_sorted if i not in removed_var]
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDM5YWFjZWVlMWQzZmY2NGEyMDBlOTlhMTAyMTIzNmNfeXZLZW1IbHhtR0htbERqeWF6YklCU3lJWlRMQ3FGZzhfVG9rZW46S3hzZGJrbE5Kb3cxRUd4MUNjcmNaYkpnbjJmXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

> 删除了两次两两之间相关性较强的变量

接下来我们还要检验一个变量与其他变量之间多重共线性的操作

```Python
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
```

> 本数据源中不含有方差膨胀因子大于10的变量

## 检测是否有系数符号不满足WOE编码的系数要求

```Go
X_train, y_train = train_df_nona[retained_woe_vars], train_df_nona[target_var]
LR = LogisticRegression()
LR.fit(X_train, y_train)
print(LR.coef_)
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=MzQ3NzhjNWMzY2UwZWQwMDA5YjA2NjEyNWI0OWY4NWNfcVJ6d2d3RWNwTjhheFNTUnlObUpBa3Zsakpoc2lwT3VfVG9rZW46WE9oRGJieUZwb2ZIenB4SzNjMmNqZFNQbjRnXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

> 该程序的WOE编码分子是非违约样本的比例，分母是违约样本的比例，所以一个变量的WOE值是和它非违约样本的比例成正比的

## 显著性筛选

我们希望模型里面的每一个变量都是显著的

```Python
import statsmodels.api as sm
X_train_temp = X_train.copy()
X_train_temp['intercept'] = [1]*X_train_temp.shape[0]
lr = sm.Logit(y_train, X_train_temp).fit()
summary = lr.summary2()
pvals = lr.pvalues.to_dict()
vars_into_clf = []
for k,v in pvals.items():
    if k=='intercept': continue
    if v>=0.1:
        print('注意：{0}的 p-value 为 {1}, 删除'.format(k, v))
    else:
        vars_into_clf.append(k)
```

本样本数据中不存在不显著的变量

## 分类模型

```Python
X_train_into_clf = train_df_nona[vars_into_clf]
LR = LogisticRegression()
LR.fit(X_train_into_clf, y_train)
```

## **导入测试样本，进行相同的缺失值分箱、WOE编码和变量筛选**

```Go
test_df_nona = test_df.fillna('bin_nan')
for woe_var in vars_into_clf:
    var = woe_var.replace('_WOE', '')
    test_df_nona[woe_var] = test_df_nona[var].map(WOE_dict[var])
X_test_into_clf, y_test = test_df_nona[vars_into_clf], test_df_nona[target_var]
pred = LR.predict(X_test_into_clf)
```

# 模型检验

```Python
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
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjNhOTljNDI5MWNhMTRjYTJkZTk2Y2VjZDM2NWUzNThfcDQzb0FKTmRTY1JNUTJoNVlsSWtjQ2czUGtzTW5EYXlfVG9rZW46T2llZWJxajZlb2s4UnB4SzJORmNpQ1RTbjllXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTM4NDU2NjEwN2VjZjM1M2RhYTZhZDQ1OTY4YWRkM2NfaWdYTUM5emJROHZJbXlXaDdMNU1tOXdDcG1wWXNMeE9fVG9rZW46SEMxMWJsdkI4b1JaZFl4Tkkxd2NWdXBHbjltXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)

尺度化

```Python
def Prob2Score(prob, basePoint, PDO):
    # 将概率转化成分数且为正整数
    y = np.log(prob / (1 - prob))
    y2 = basePoint - PDO / np.log(2) * (y)
    scores = y2.astype("int")
    return scores


scores = Prob2Score(y_prb_test, 300, 100)
plt.hist(scores, bins = 100)
plt.show()
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=MzAxNTYzMzNmZDNhOGQ5M2YyMjJlYTIxNGY5MWNlOThfZXBQem1XVXhNdThFRDlWcGZ5TmU2Ykl3ZTFHYVVLOG5fVG9rZW46Um1mNWJrRDFKb0JQQ0Z4UEFFQ2NQYldvbmFoXzE2ODA3ODQyMjU6MTY4MDc4NzgyNV9WNA)
