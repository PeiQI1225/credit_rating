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