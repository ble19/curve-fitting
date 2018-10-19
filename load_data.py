import pandas as pd
import os
import datetime

'''this is the method that shapes the data from the kaggle daily dataset to a useable format to predict household power 
 consumption.

Input: the Kaggle daily dataset for household power usage.
Output: A file which the transformed dataset ready for regression analysis.

Notes: I used the dataset to play around with simple regression analysis but if you haven't looked around on the kaggle page for the
dataset, there are interesting relationships to be explored including weather and government holidays.
'''
def load_dataset():

    #this selects the data we wish to compare from the batch files before rearranging them
    dataset = 'dataset file path here'
    list_dailyselection = []
    for file in os.listdir(dataset):
        df_daily = pd.read_csv(dataset+'\\'+file)
        df_daily_selection = df_daily[["LCLid", "day", "energy_sum"]]
        list_dailyselection.append(df_daily_selection)

    #the concat below provides an efficient way to combine the data
    df_dailyselection = pd.concat(list_dailyselection, axis=0)
    df_dailyselection["day"] = pd.to_datetime(df_dailyselection["day"])
    df_count = df_dailyselection.groupby(["day"]).count()

    #the date range was selected because the most households were enrolled between this period
    start_date = datetime.datetime(year=2013, month=1, day=1)
    end_date = datetime.datetime(year=2014, month=1, day=1)
    df_dailyselection_zoom = df_dailyselection[
        (df_dailyselection["day"] >= start_date) & (df_dailyselection["day"] < end_date)]



    # create usable data format. In this case, label = household id, and the rest of the entry is the measure
    # this is using java logic, does not work because row values not tabled. Also doesn't use python efficiently

    reformatted_datalist = []
    '''for row in df_dailyselection_zoom.iterrows():
            if check_none([row-1]) or row != row-1:  # check whether first row or new household id
                entry = []
                np.append(entry, row[0])
                np.append(entry, row[1])
            elif row[0] == [row-1][0]:               # check for continuing value
                np.append(entry, row[1])
                if row+1 is None:                  # handle entry
                    break
                elif row[0] != row+1[0]:           # handle last household id in series
                    np.append(reformatted_data, entry)'''

    ''' The important part here is the pivot which puts that data into the household id followed by the year's data.
    Again, concat made combining the rows a breeze. The data has to be written to a preexisting file so make sure
    you have one ready. This saves time from having to reformat the data everytime.'''
    HouseholdIDS = pd.unique(df_dailyselection_zoom.LCLid)

    for index in HouseholdIDS:
        entrytable = df_dailyselection_zoom.loc[df_dailyselection_zoom['LCLid'] == index]
        entrytable = pd.DataFrame(entrytable).pivot(index='LCLid', columns='day', values='energy_sum')
        reformatted_datalist.append(entrytable)
    result = pd.concat(reformatted_datalist)

    result.to_csv('~/Documents/reformatted_data.csv')
    return result
