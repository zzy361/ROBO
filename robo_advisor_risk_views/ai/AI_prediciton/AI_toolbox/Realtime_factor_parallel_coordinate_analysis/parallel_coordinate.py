import pandas as pd
import numpy as np

class parallel_coordinate:
    def __init__(self):
        pass
    def get_factor_data(self,data,back_days):
        self.factor_data = data.iloc[-back_days:,:]
    def label_mean_std_cal(self,data):
        label_split = data.groupby('Label')
        label_name = np.array(label_split)[:,0]
        temp_data = np.array(label_split)[:,1]
        for i in temp_data:
            del i['Label']
        df_mean = temp_data[0].mean().to_frame()
        df_std = temp_data[0].std().to_frame()
        for i in range(len(label_name)-1):
            df_mean = pd.merge(df_mean,temp_data[i+1].mean().to_frame())
            df_std = pd.merge(df_std, temp_data[i + 1].std().to_frame())
        df_mean.clumns = label_name
        df_std.columns = label_name
        self.df_mean = df_mean.T
        self.df_std = df_std.T
    def choose_label(self,confidence_param):
        df_upper_bound = self.df_mean+ confidence_param*self.df_std
        df_low_bound = self.df_mean - confidence_param * self.df_std
        result = []
        for i in range(df_upper_bound.shape[1]):
            temp_up_df = df_upper_bound.iloc[:,i].to_frame()
            temp_up_df.sort_values(by=list(temp_up_df.columns),inplace=True)
            temp_low_df = df_low_bound.reindex(list(temp_up_df.index))
            count = 0
            for j in range(temp_low_df.shape[0]-1):
                if temp_up_df.iloc[j+1,0] >temp_low_df.iloc[j,0]:
                    count += 1
            if count ==0:
                result.append(df_upper_bound.columns[i])
        return result

    def factor_select(self,data,back_days,confidence_param=0.2):
        self.get_factor_data(data,back_days)
        self.label_mean_std_cal(self.factor_data)
        choosen_label = self.choose_label(confidence_param)
        return choosen_label
