import numpy as np
import pandas as pd
import warnings
from scipy import interpolate
warnings.filterwarnings("ignore")

# Function goal: calculate the daily average and month average values of seven indexes
# Input parameters:
# df - ERA5 data after preprocessing
# First_year & End_year - wider than the statistical range
# First_count_year & End_count_year : statistic aim of typical months, for instance the typical month is selected to reflect the average condition from 1980 to 2010
# First_select_year & End_select_year :statistic range of typical months, for instance, the typical month is selected from 1990 to 2000
# Output
# Index_7_daily: daily average values of seven indexes
# Index_7_month: monthly average values of seven indexes
# df_index_mean: dataframe of monthly average values (12*7)
# df_index_std : dataframe of monthly std values (12*7)
# data_origin: data of statistic range of ERA5
def preprocess_daily_month_data(df, First_year, End_year, First_count_year, End_count_year, First_select_year, End_select_year):
    #filename = os.path.splitext(file)[0]
    #print("Processing the data of {filename}".format(filename = filename))
    #abspath = os.path.join(path,file)
    Total_year = End_year - First_year + 1 #start_year - end_year 
    #df = pd.read_csv(abspath)
    data = df[(df["Year"]>=First_year) & (df["Year"]<=End_year)].reset_index(drop = True) # cut the years we need
    data_year_list = []
    for year in list(data.Year.unique()):
        data_temp = data[data["Year"] == year]
        #figure out 2.29 and delete
        if year%4 == 0:
            data_temp = data_temp.drop(index = data_temp[(data_temp["Hour"]>=744) &(data_temp["Hour"]<=767)].index)
        data_temp.reset_index(inplace = True, drop = True)
        data_year_list.append(data_temp)
    # data need: First_year - End_year
    data = pd.concat(data_year_list,axis = 0)
    data.reset_index(inplace = True, drop = True)
    print("After processing, the total length of data is %d"%len(data))

    # final data need: End_select_year - First_select_year 
    total_year_select = End_select_year - First_select_year + 1
    data_origin = data[(data["Year"]>=First_select_year) & (data["Year"]<=End_select_year)].reset_index(drop = True)

    Data_Feature = data_origin[["Year","DryBulbT(C)","WaterVaporPressure(Pa)","TotalRadHori(W/m2)","GroundT(C)","WindSpeed(m/s)"]]
    HAN = data_origin["Enthalpy(kJ/kg.K)"]
    #print(len(data_origin))

    ##Seven indexes 
    Year_list = turn_into_daily(Data_Feature["Year"].values, "mean")
    AVE_temp_daily = turn_into_daily(Data_Feature["DryBulbT(C)"].values, "mean")
    Min_temp_daily = turn_into_daily(Data_Feature["DryBulbT(C)"].values, "min")
    Max_temp_daily = turn_into_daily(Data_Feature["DryBulbT(C)"].values, "max")
    AVE_VP_daily = turn_into_daily(Data_Feature["WaterVaporPressure(Pa)"].values, "mean")
    Sum_Radi_daily = turn_into_daily_radi(Data_Feature["TotalRadHori(W/m2)"].values)
    AVE_GST_daily = turn_into_daily(Data_Feature["GroundT(C)"].values, "mean")
    AVE_WS_daily = turn_into_daily(Data_Feature["WindSpeed(m/s)"].values, "mean")
    Han_daily = turn_into_daily(HAN.values, "mean")
    #merge the daily values
    Index_7_daily = pd.DataFrame({"Year":Year_list,"AVE_temp": AVE_temp_daily,"Min_temp":Min_temp_daily,"Max_temp":Max_temp_daily,
                                  "AVE_VP":AVE_VP_daily,"Sum_Radi":Sum_Radi_daily,
                                 "AVE_GST":AVE_GST_daily,"AVE_WS":AVE_WS_daily, "Han":Han_daily})
    #calculate month values
    monthnum = [31,28,31,30,31,30,31,31,30,31,30,31];
    Year_list_month = []
    AVE_temp_month = []
    Min_temp_month = []
    Max_temp_month = []
    AVE_VP_month = []
    Sum_Radi_month = []
    AVE_GST_month = []
    AVE_WS_month = []
    Han_month = []
    for year in Index_7_daily["Year"].unique():
        daily_data_temp = Index_7_daily[Index_7_daily["Year"] == year].reset_index(drop = True)
        #print(daily_data_temp)
        Year_list_t = daily_data_temp["Year"].values
        AVE_temp_daily_t = daily_data_temp["AVE_temp"].values
        Min_temp_daily_t = daily_data_temp["Min_temp"].values
        Max_temp_daily_t = daily_data_temp["Max_temp"].values
        AVE_VP_daily_t = daily_data_temp["AVE_VP"].values
        Sum_Radi_daily_t = daily_data_temp["Sum_Radi"].values
        AVE_GST_daily_t = daily_data_temp["AVE_GST"].values
        AVE_WS_daily_t = daily_data_temp["AVE_WS"].values
        Han_daily_t = daily_data_temp["Han"].values
        start = 0 #start
        #count monthly average values
        for num in monthnum:
            Year_list_month.append(np.mean(Year_list_t[start : start+num]))
            AVE_temp_month.append(np.mean(AVE_temp_daily_t[start : start+num]))
            #print(len(Min_temp_daily_t))
            Min_temp_month.append(np.mean(Min_temp_daily_t[start : start+num]))
            Max_temp_month.append(np.mean(Max_temp_daily_t[start : start+num]))
            AVE_VP_month.append(np.mean(AVE_VP_daily_t[start : start+num]))
            Sum_Radi_month.append(np.mean(Sum_Radi_daily_t[start : start+num]))
            AVE_GST_month.append(np.mean(AVE_GST_daily_t[start : start+num]))
            AVE_WS_month.append(np.mean(AVE_WS_daily_t[start : start+num]))
            Han_month.append(np.mean(Han_daily_t[start : start+num]))
            start = start + num 
    #Month_list
    Month_list = list(range(1,13))*total_year_select
    #seven indexes (12*30*7)
    Index_7_month = pd.DataFrame({"Year":Year_list_month,"Month":Month_list,"AVE_temp": AVE_temp_month,"Min_temp":Min_temp_month,"Max_temp":Max_temp_month,
                                  "AVE_VP":AVE_VP_month,"Sum_Radi":Sum_Radi_month,
                                 "AVE_GST":AVE_GST_month,"AVE_WS":AVE_WS_month, "Han":Han_month})

    #seven indexes monthly values (12*7)
    Index_count = Index_7_month[(Index_7_month["Year"]>=First_count_year)&(Index_7_month["Year"]<=End_count_year)].reset_index(drop = True)
    columns_save = ["AVE_temp","Min_temp","Max_temp","AVE_VP","Sum_Radi","AVE_GST","AVE_WS"]
    #print(Index_count)
    Index_mean = []
    Index_std = []
    for month in Index_7_month.Month.unique():
        df_temp = Index_count[Index_count["Month"] == month][columns_save]
        Index_mean.append(df_temp.mean().values)
        Index_std.append(df_temp.std().values)
    df_index_mean = pd.DataFrame(Index_mean)
    df_index_mean.columns = columns_save
    df_index_std = pd.DataFrame(Index_std)
    df_index_std.columns = columns_save
    #print(df_index_mean)
    #print(df_index_std)
    
    return(data_origin, Index_7_daily, Index_7_month, df_index_mean, df_index_std)

#count daily 
#input: values - orginal data; type - mean / max / min
def turn_into_daily(values,needtype):
    step = 24
    data = [values[i:i+step] for i in range(0,len(values),step)]
    data_daily = []
    for lis in data:
        if needtype == "mean":
            data_daily.append(np.mean(lis))
        elif needtype == "max":
            data_daily.append(np.max(lis))
        elif needtype == "min":
            data_daily.append(np.min(lis))
    return(data_daily)

#count daily value of radiation
#input: values - orginal data; type - mean / max / min
def turn_into_daily_radi(values):
    step = 24
    data = [values[i:i+step] for i in range(0,len(values),step)]
    data_daily = []
    for lis in data:
        data_daily.append(np.sum(lis)*60*60/1000000) #转化为日总辐射
    return(data_daily)

# function goal: calculate the std values of seven indexes and output the eta of each month and the typical month from which year
# input parameters: the average values of seven indexes(mean + std), columns, weights of different indexes, start year;
# output :
def generate_TMY(Index_7_month,Index_mean,Index_std,columns,weight,First_select_year):
    #calculate eta of seven indexes
    #for each month, select the eta of seven indexes less than 1 and save in the "select_index_12month"
    #Also, save the results of eta of seven indexes less than 1 in a dataframe "eta_12month"之内"
    select_index_12month = []
    eta_12month = []
    for month in Index_7_month.Month.unique():
        df_temp = Index_7_month[Index_7_month["Month"] == month][columns]
        df_temp_2 = df_temp.apply(lambda x: (x - np.mean(x))/np.std(x))
        eta_12month.append(df_temp_2[abs(df_temp_2)<1].dropna(axis = 0, how = "any"))
        select_index = df_temp_2[abs(df_temp_2)<1].dropna(axis = 0, how = "any").index.values
        select_index_12month.append(select_index)
    
    #find the most smallest eta and the corresponding year 
    eta_total_min_12month = []
    select_loc_12month = []
    for df_eta in eta_12month:
        eta_total_min = 100
        select_loc = 0
        for index in df_eta.index:
            count = 0
            eta_total = 0
            for col in df_eta.columns:
                eta_total = eta_total + abs(df_eta.loc[index,col])*weight[count]
                count = count + 1
            if eta_total <= eta_total_min:
                #print(eta_total,eta_total_min)
                eta_total_min = eta_total
                select_loc = index
        eta_total_min_12month.append(eta_total_min)
        select_loc_12month.append(select_loc)
    #return(Eta_7_month_cut,Possible_year_list,Possible_month_list,DM_list)#,df_DM,TMY_DM_value, TMY_DM_year
    select_12month_year = np.array(Index_7_month.loc[select_loc_12month,"Year"].values,dtype = int)
    return(select_index_12month,eta_12month,eta_total_min_12month,select_12month_year)

# function goal: according to the year and data, merge the final TMY
# input parameters: data; number of select years; the year of typical months
def combine_select_months_to_TMY(data_select,total_select_years, select_12month_year):
    month_list = []
    monthnum = [31,28,31,30,31,30,31,31,30,31,30,31];
    for k in range(len(monthnum)):
        num = monthnum[k]
        month_now = k+1
        for kk in range(24*num):
            month_list.append(month_now)
    month_list_final = month_list*total_select_years
    data_select["Month"] = month_list_final
    #According to the year, find the typical month
    data_TMY_month_list = []
    temp_value_list = []
    AH_value_list = []
    for kkk in range(len(select_12month_year)):
        month = kkk+1
        year = select_12month_year[kkk]
        print(month,year)
        df_s = data_select[(data_select["Year"]==year) & (data_select["Month"]==month)]
        temp_value_list.append(list(df_s["DryBulbT(C)"].values))# dry bulb temperature needs interpolation 
        AH_value_list.append(list(df_s["MoistureContent(g/kg.dra)"].values))#absolute humidity needs interpolation 
        data_TMY_month_list.append(df_s)
    data_TMY_final = pd.concat(data_TMY_month_list, axis = 0)
    data_TMY_final.reset_index(inplace = True, drop = True)
    temp_value_final = smooth_monthly(temp_value_list)# dry bulb temperature needs interpolation 
    AH_value_final = smooth_monthly(AH_value_list)#absolute humidity needs interpolation 
    data_TMY_final["DryBulbT(C)"] = temp_value_final
    data_TMY_final["MoistureContent(g/kg.dra)"] = AH_value_final
    return(data_TMY_final)

#Interpolation
def smooth_monthly(value_list):
    for ii in range(len(value_list)-1):
        last_value = value_list[ii][-12]
        sec_value = value_list[ii][-6]
        thi_value = value_list[ii+1][0]
        for_value = value_list[ii+1][6]
        final_value = value_list[ii+1][12]
        x_list = range(24)# interpolation of the 24 hours among the month
        x_p = [0,6,12,18,24]
        f_p = [last_value,sec_value,thi_value, for_value, final_value]
        print(f_p)
        tck = interpolate.splrep(x_p,f_p)
        y_list = interpolate.splev(x_list,tck, der = 0)
        #print(y_list)
        #print(ii)
        value_list[ii][-12:] = y_list[0:12]
        value_list[ii+1][0:12] = y_list[12:]
    value_final = []
    for lis in value_list:
        value_final = value_final + lis
    return(value_final)