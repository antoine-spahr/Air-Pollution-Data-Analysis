# Air Pollution & Climate Change - Pollution Data Analysis
#
# author: Antoine Spahr
#
# 28.02.19
#
# Analysis of air pollutant in two locations : Sion and Tanikon
# Data source : Pollutant : NABEL
#               Wind : MeteoSuisse
#_______________________________________________________________________________

#_______________________________________________________________________________
# %% Import Library
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy import stats
from scipy import signal
import numpy as np
from numpy.polynomial.polynomial import polyfit

# <*> Functions + params
# %%

data_folder = r'./data/'
plt.rc('font', family='sans-serif')
plt.rcParams.update({'font.size': 10})
figure_res = 300

def mean_direction(direction, r=1):
    """ Get the mean of a periodic value by taking the vectorial mean """
    direction_rad = direction * np.pi/180
    x = np.mean(r * np.cos(direction_rad))
    y = np.mean(r * np.sin(direction_rad))
    direction_deg = np.arctan2(y,x)*180/np.pi

    if np.sign(direction_deg) < 0:
        direction_deg = (direction_deg + 360) %360

    return direction_deg

def std_direction(direction, r=1):
    """ Get the std of a periodic value by taking the vectorial sdt """
    direction_rad = direction * np.pi/180
    x = np.std(r * np.cos(direction_rad))
    y = np.std(r * np.sin(direction_rad))
    direction_deg = np.arctan2(y,x)*180/np.pi

    if np.sign(direction_deg) < 0:
        direction_deg = (direction_deg + 360) %360

    return direction_deg

def getSeason(date):
    """ get the season associated with a date
        INPUT : the date as a datetime object
        OUTPUT : the season as a string"""
    month = date.month
    day = date.day
    seasons = {1:'winter', 2:'spring', 3:'summer', 4:'autumn'}

    if (not month in [3, 6, 9, 12]) or (day >= 21):
        return seasons[(month%12 + 3)//3]
    elif (month == 3) and (day < 21):
        return seasons[1]
    elif (month == 6) and (day < 21):
        return seasons[2]
    elif (month == 9) and (day < 21):
        return seasons[3]
    elif (month == 12) and (day < 21):
        return seasons[4]

def dirAngle2str(angle):
    dir_map = {0:'N',1:'NE',2:'E',3:'SE',4:'S',5:'SW',6:'W',7:'NW'}
    return dir_map[int((angle+22)/45)%8]

def getDaytype(date):
    """ get the type of day at a given day (weekend of weekday)
        INPUT: a date as a datetime object
        OUTPUT: the type of day (weekend or weekday), as a string """

    day = date.weekday() # 0:Monday, ... , 4:Friday, 5:Saturday, 6:Sunday
    if day > 4:
        return 'weekend'
    else:
        return 'weekday'

def loadData(data_filename, winddata_filename):
    """
        Load data from NABEL pollutant data, and wind data from meteo suisse
        return a pandas Dataframe with the to data merged based on date (index = date)
    """
    # import hourly data of the various pollutant (O3, NO2, NOX, PM10), and meteorological factor (Temperature, precipitation, and Radiation)
    pol = pd.read_csv(data_folder+data_filename, sep=';' , skiprows=5, encoding="ISO-8859-1")
    pol.columns = ['Date','O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3]', 'Temp [C]', 'Prec [mm]', 'RAD [W/m2]']
    pol['Date']  = pd.to_datetime(pol['Date'], format='%d.%m.%Y %H:%M')

    # import the wind data and get the date as a datetime object
    wind = pd.read_csv(data_folder+winddata_filename, names=['year', 'month', 'day', 'hour', 'minute', 'wind direction [°N]', 'windspeed [m/s]'], header=None, skiprows=15, delim_whitespace=True)
    # change the hour = 24 in hour = 0 to build the datetime object
    wind.loc[wind['hour'] == 24, 'hour'] = 0
    wind['Date'] = wind[['year', 'month', 'day', 'hour', 'minute']].apply(lambda row: dt.datetime(row.year, row.month, row.day, row.hour, row.minute), axis=1)
    wind.drop(columns={'year', 'month', 'day', 'hour', 'minute'}, inplace=True)
    wind.set_index('Date').resample('H').agg({'wind direction [°N]':mean_direction,'windspeed [m/s]':'mean'}).reindex().fillna(method='ffill')

    # merge the two data based on the common dates
    data = pd.merge(pol, wind, left_on='Date', right_on='Date', how='inner')

    # get a new season column
    data['Season'] = data['Date'].apply(getSeason)

    # get the day type (weekend or weekday)
    data['Day type'] = data['Date'].apply(getDaytype)

    return data

def check_24h_limit(df_24h_means, limit_daily, pol, loc):
    # get row above the limit
    df = df_24h_means[df_24h_means[pol+' [ug/m3]'] > limit_daily].reset_index()
    # compute the time delta between the rows to identify the different time periodes
    df['time_delta'] = (df.Date - df.Date.shift(1)).astype('timedelta64[h]')
    # if there are some value above the limit
    if df.shape[0] != 0:
        # get the number of time delta larger than 1 (= new period)
        nbr_period = df.loc[df['time_delta'] > 1, 'time_delta'].count()+1
    else:
        nbr_period = 0
    nbr_hour = df.shape[0]

    print('In {0} : 24h averages of {1} has exceed the legal limit of {2} [ug/m3] during {3} periodes of times and for a total of {4} hours.'.format(loc,pol,limit_daily,nbr_period,nbr_hour))

def check_hourly_limit(df_hourly, limit_hourly, pol, loc, periods_above_hourly, hour_above_hourly):
    df = df_hourly[df_hourly[pol+' [ug/m3]'] > limit_hourly].reset_index()
    # compute the time delta between the rows to identify the different time periodes
    df['time_delta'] = (df.Date - df.Date.shift(1)).astype('timedelta64[h]')

    nbr_period = 0

    # if there are some value above the limit
    if df.shape[0] != 0:
        # get the number of time delta larger than 1 (= new period)
        nbr_period = df.loc[df['time_delta'] > 1, 'time_delta'].count()+1
        periods_above_hourly[loc+'_'+pol] = nbr_period

    nbr_hour = df.shape[0]
    hour_above_hourly[loc+'_'+pol] = nbr_hour

    print('In {0} : Hourly averages of {1} has exceed the legal limit of {2} [ug/m3] during {3} periodes of times and for a total of {4} hours.'.format(loc,pol,limit_hourly,nbr_period,nbr_hour))

def check_annual_limit(df, limit_annual, pol, loc):
    if df[pol+' [ug/m3]'] > limit_annual:
        print('In {0} : the annual average of {1} has exceeded the legal limit of {2} [ug/m3], by {3:.2f} [ug/m3]'.format(loc, pol, limit_annual, df[pol+' [ug/m3]']-limit_annual))
    else:
        print('In {0} : the annual average of {1} has NOT exceeded the legal limit of {2} [ug/m3], by {3:.2f} [ug/m3]'.format(loc, pol, limit_annual, limit_annual-df[pol+' [ug/m3]']))

def comparisonTests(data_weekday, data_weekend, index_names, cols, test='mannwhitneyu'):
    """ perform the Mann-Whitney-U test between weekend/weekday data of both Sion and Tanikon, for all element in index names """

    col_names = ['SIO weekday\n vs\n SIO weekend', 'SIO weekday\n vs\n TAE weekday', 'SIO weekday\n vs\n TAE weekend', 'SIO weekend\n vs\n TAE weekday', 'SIO weekend\n vs\n TAE weekend', 'TAE weekday\n vs\n TAE weekend']
    pval_df = pd.DataFrame(index=index_names, columns=col_names)

    for idx in range(len(index_names)):
        p_vals = []
        pairs = [(data_weekday[cols[idx]+' SIO'], data_weekend[cols[idx]+' SIO']), \
                 (data_weekday[cols[idx]+' SIO'], data_weekday[cols[idx]+' TAE']), \
                 (data_weekday[cols[idx]+' SIO'], data_weekend[cols[idx]+' TAE']), \
                 (data_weekend[cols[idx]+' SIO'], data_weekday[cols[idx]+' TAE']), \
                 (data_weekend[cols[idx]+' SIO'], data_weekend[cols[idx]+' TAE']), \
                 (data_weekday[cols[idx]+' TAE'], data_weekend[cols[idx]+' TAE'])]

        for p in pairs:
            if test=='mannwhitneyu':
                _ , p_val = stats.mannwhitneyu(p[0], p[1])
                p_vals.append(p_val)
            elif test=='wilcoxon':
                _ , p_val = stats.wilcoxon(p[0], p[1])
                p_vals.append(p_val)
            elif test=='kolmogorovsmirnov':
                _ , p_val = stats.ks_2samp(p[0], p[1])
                p_vals.append(p_val)
            else:
                raise ValueError('Wrong test type!')

        pval_df.iloc[idx,:] = p_vals

    return pval_df

def crossCorr(data, padding=(12,12)):
    """
        Compute the cross-correlation between the columns of data on the shift defined by padding
        and return two dataframes : the highest correlation (in absolute values) and the corresponding lag
    """
    col_names = data.columns

    corr_array = np.zeros((len(col_names),len(col_names)))
    lag_array = np.zeros((len(col_names),len(col_names)))

    for i, col1 in enumerate(col_names):
        for j, col2 in enumerate(col_names):
            s1 = np.array(data[col1].fillna(0))
            s2 = np.pad(np.array(data[col2].fillna(0)), padding, 'linear_ramp')#, 'constant', constant_values=(0, 0))
            s1 = (s1 - np.mean(s1)) / (np.std(s1) * np.std(s2) * len(s1))
            s2 = (s2 - np.mean(s2)) / (np.std(s1) * np.std(s2) * len(s2))

            cross_cor = signal.correlate(s1, s2, mode='valid')
            best_lag = np.argmax(np.abs(cross_cor)) # span with max correlation
            lag_array[i,j] = best_lag - padding[0]
            corr_array[i,j] = cross_cor[best_lag]

    corr_df = pd.DataFrame(data=corr_array, index=col_names, columns=col_names)
    lag_df = pd.DataFrame(data=lag_array, index=col_names, columns=col_names)

    return corr_df, lag_df

def testNormality(data, test='shapiro'):
    pval = []
    for col in data.columns:
        if test=='shapiro':
            pval.append(stats.shapiro(data[col])[1])
        elif test == 'normaltest':
            pval.append(stats.normaltest(data[col])[1])
        else:
            raise ValueError('Specied Test not valid')

    return pd.Series(data=pval)

def conf_int_bootstrap(data, N=10000, a=0.05, agg_dict=None):
    """ compute a confidence interal for the mean of the columns on data (data is a pandas Dataframe)"""
    means = []
    for i in range(N):
        sub_data = data.sample(frac=0.9, replace=True)
        if agg_dict is None:
            means.append(sub_data.mean(axis=0))
        else:
            means.append(sub_data.groupby(np.ones(len(sub_data))).agg(agg_dict, axis=0).iloc[0,:])

    means_df = pd.concat(means, axis=1).transpose()
    return means_df.quantile([a/2, 1-a/2], axis=0)

# <*!>

# <*> Import Sion Data
# %%

data_sion = loadData('SIO_Hourly.csv', 'SIO_Wind_MM10_18.txt')
#data_sion.describe()

# <*!>

# <*> Import Tanikon Data
# %%

data_tanikon = loadData('TAE_Hourly.csv', 'TAE_Wind_MM10_18.txt')
#data_tanikon.describe()

# <*!>

#<*> PART 1
#_______________________________________________________________________________
# <*> defining function to apply for the mean in a groupBy
# --> special mean for wind
# --> sum for precipitation (get cumulative precipitation)
# %%

mean_dict_meteo = {'Temp [C]':'mean', \
                   'Prec [mm]':'sum', \
                   'RAD [W/m2]':'mean', \
                   'wind direction [°N]':mean_direction, \
                   'windspeed [m/s]':'mean'}

std_dict_meteo = {'Temp [C]':'std',  \
                   'RAD [W/m2]':'std', \
                   'wind direction [°N]':std_direction, \
                   'windspeed [m/s]':'std'}

mean_dict = {'O3 [ug/m3]':'mean', \
             'NO2 [ug/m3]':'mean', \
             'PM10 [ug/m3]':'mean', \
             'NOX [ug/m3]':'mean', \
             'Temp [C]':'mean', \
             'Prec [mm]':'sum', \
             'RAD [W/m2]':'mean', \
             'wind direction [°N]':mean_direction, \
             'windspeed [m/s]':'mean'}

std_dict = {'O3 [ug/m3]':'std', \
             'NO2 [ug/m3]':'std', \
             'PM10 [ug/m3]':'std', \
             'NOX [ug/m3]':'std', \
             'Temp [C]':'std', \
             'RAD [W/m2]':'std', \
             'wind direction [°N]':std_direction, \
             'windspeed [m/s]':'std'}
# <*!>

#_______________________________________________________________________________
# <*> Meteorological Parameters by Seasons

# %% Grouping by Season
#_______________________________________________________________________________

# Get the meteorological data into a single dataframe for both locations group by season
# Get the mean by season as well as the standrad deviation by season

data_meteo_sion_season = data_sion[['Season', 'Temp [C]', 'Prec [mm]', 'RAD [W/m2]', 'wind direction [°N]', 'windspeed [m/s]']].groupby('Season').agg(mean_dict_meteo)
data_meteo_tanikon_season = data_tanikon[['Season', 'Temp [C]', 'Prec [mm]', 'RAD [W/m2]', 'wind direction [°N]','windspeed [m/s]']].groupby('Season').agg(mean_dict_meteo)

meteo_sion = data_sion[['Temp [C]', 'Prec [mm]', 'RAD [W/m2]', 'wind direction [°N]', 'windspeed [m/s]']]
meteo_tanikon = data_tanikon[['Temp [C]', 'Prec [mm]', 'RAD [W/m2]', 'wind direction [°N]', 'windspeed [m/s]']]

#Confidence intervals Sion
ci_inf_sion = pd.DataFrame(index=None, columns=meteo_sion.columns)
ci_sup_sion = pd.DataFrame(index=None, columns=meteo_sion.columns)

for season in data_sion.Season.unique():
    ci_95 = conf_int_bootstrap(meteo_sion[data_sion['Season']==season], agg_dict=mean_dict_meteo)
    ci_inf_sion = ci_inf_sion.append(ci_95.iloc[0,:].rename(season))
    ci_sup_sion = ci_sup_sion.append(ci_95.iloc[1,:].rename(season))

# Confidence interval Tanikon
ci_inf_tanikon = pd.DataFrame(index=None, columns=meteo_tanikon.columns)
ci_sup_tanikon = pd.DataFrame(index=None, columns=meteo_tanikon.columns)

for season in data_tanikon.Season.unique():
    ci_95 = conf_int_bootstrap(meteo_tanikon[data_tanikon['Season']==season], agg_dict=mean_dict_meteo)
    ci_inf_tanikon = ci_inf_tanikon.append(ci_95.iloc[0,:].rename(season))
    ci_sup_tanikon = ci_sup_tanikon.append(ci_95.iloc[1,:].rename(season))

# Merging
data_meteo_season = pd.merge(data_meteo_sion_season, data_meteo_tanikon_season, left_index=True, right_index=True, how='inner', suffixes=(' SIO', ' TAE'))
data_meteo_season = data_meteo_season.reindex(index=['winter', 'spring', 'summer', 'autumn'])

data_meteo_season_inf = pd.merge(ci_inf_sion, ci_inf_tanikon, left_index=True, right_index=True, how='inner', suffixes=(' SIO', ' TAE'))
data_meteo_season_inf = data_meteo_season_inf.reindex(index=['winter', 'spring', 'summer', 'autumn'])
data_meteo_season_inf = -data_meteo_season_inf + data_meteo_season

data_meteo_season_sup = pd.merge(ci_sup_sion, ci_sup_tanikon, left_index=True, right_index=True, how='inner', suffixes=(' SIO', ' TAE'))
data_meteo_season_sup = data_meteo_season_sup.reindex(index=['winter', 'spring', 'summer', 'autumn'])
data_meteo_season_sup = data_meteo_season_sup - data_meteo_season

# %% Plot the results in a grouped barplot
#_______________________________________________________________________________

col = data_meteo_season.columns.to_list()
colors = ['lightskyblue','yellowgreen','lightcoral','burlywood']
barwidth = 0.9
titles = ['Temperature','Precipitation','Radiation','Wind direction', 'Wind speed']

fig1_2, axs = plt.subplots(1,5,figsize=(15,4))

for idx, ax in enumerate(axs.reshape(-1)):
    data_meteo_season[col[idx::5]].transpose().plot(kind='bar', ax = ax, color=colors, width=barwidth, \
                                                    yerr=np.array([data_meteo_season_inf[col[idx::5]].values, data_meteo_season_sup[col[idx::5]].values]).transpose([1,0,2]), \
                                                    error_kw=dict(elinewidth=2, ecolor='darkgray', capsize=3))
    if idx == 1:
        ax.set_ylabel('Cumulative ' + col[idx].replace(' SIO', ''))
    else:
        ax.set_ylabel(col[idx].replace(' SIO', ''))

    ax.set_title(titles[idx])
    #ax.set_xlabel('Seasons')
    ax.set_xticklabels(['Sion', 'Tanikon'])
    ax.xaxis.set_tick_params(rotation=0)
    ax.legend().set_visible(False)

handles, _ = axs[0].get_legend_handles_labels()
labels = ['Winter', 'Spring', 'Summer', 'Autumn']
lgd = fig1_2.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.06), bbox_transform=fig1_2.transFigure, ncol=4)

fig1_2.tight_layout()
fig1_2.savefig('./Figures/'+'season2_avg.png', dpi=figure_res, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

# <*!>

# %% _______________________________________________________________________________
# <*> Monthly Evolution

pollutant_list = ['O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3]']
meteo_list = ['Temp [C]', 'Prec [mm]', 'RAD [W/m2]', 'windspeed [m/s]','wind direction [°N]']

# group the data by month for both location and merge them
monthly_sion = data_sion[['Date']+pollutant_list+meteo_list]
monthly_sion_std = monthly_sion.groupby(monthly_sion.Date.dt.month).agg(std_dict)
monthly_sion = monthly_sion.groupby(monthly_sion.Date.dt.month).agg(mean_dict)
monthly_sion['wind direction [°N]'] = monthly_sion['wind direction [°N]'].apply(dirAngle2str)
monthly_sion = monthly_sion.reindex(columns=pollutant_list+meteo_list)

monthly_tanikon = data_tanikon[['Date']+pollutant_list+meteo_list]
monthly_tanikon_std = monthly_tanikon.groupby(monthly_tanikon.Date.dt.month).agg(std_dict)
monthly_tanikon = monthly_tanikon.groupby(monthly_tanikon.Date.dt.month).agg(mean_dict)
monthly_tanikon['wind direction [°N]'] = monthly_tanikon['wind direction [°N]'].apply(dirAngle2str)
monthly_tanikon = monthly_tanikon.reindex(columns=pollutant_list+meteo_list)

data_monthly = pd.merge(monthly_sion, monthly_tanikon, how='inner', right_index=True, left_index=True, suffixes=(' SIO', ' TAE'))
data_monthly_std = pd.merge(monthly_sion_std, monthly_tanikon_std, how='inner', right_index=True, left_index=True, suffixes=(' SIO', ' TAE'))

# %% Plot all the pollutant timeseries for both location in a single plot (same color for pollutant) same linestyle for locations
#_______________________________________________________________________________
width = 2.5
col = data_monthly.columns.to_list()
colors = ['sandybrown','mediumaquamarine']
titles = ['O3','NO2','PM10','NOX','Temperature','Precipitation','Radiation','Wind speed']
ftsize = 9

fig2_1, axs = plt.subplots(2,4,figsize=(14,7))

for idx, ax in enumerate(axs.reshape(-1)):
    if not idx in [5,6]:
        ax.fill_between(data_monthly.index, data_monthly[col[idx]]+data_monthly_std[col[idx]], data_monthly[col[idx]]-data_monthly_std[col[idx]], facecolor=colors[0], alpha=0.3)
        ax.fill_between(data_monthly.index, data_monthly[col[idx+9]]+data_monthly_std[col[idx+9]], data_monthly[col[idx+9]]-data_monthly_std[col[idx+9]], facecolor=colors[1], alpha=0.3)
    elif idx == 6:
        ax.fill_between(data_monthly.index, data_monthly[col[idx]]+data_monthly_std[col[idx]], facecolor=colors[0], alpha=0.3)
        ax.fill_between(data_monthly.index, data_monthly[col[idx+9]]+data_monthly_std[col[idx+9]], facecolor=colors[1], alpha=0.3)


    data_monthly[col[idx::9]].plot(ax=ax, color=colors, linewidth=width)
    ax.set_title(titles[idx])
    ax.set_xlabel(None)

    if idx == 7:
        for i, xy in enumerate(zip(data_monthly.index, data_monthly[col[idx]])):
            if i == 0:
                ax.annotate(data_monthly.loc[i+1,col[idx+1]], xy=xy, xytext=(xy[0]+0.3, xy[1]+0.3), color=colors[0], fontsize=ftsize)
            elif i == 11:
                ax.annotate(data_monthly.loc[i+1,col[idx+1]], xy=xy, xytext=(xy[0]-0.4, xy[1]+0.3), color=colors[0], fontsize=ftsize)
            else:
                ax.annotate(data_monthly.loc[i+1,col[idx+1]], xy=xy, xytext=(xy[0], xy[1]+0.3), color=colors[0], fontsize=ftsize)
            #ax.annotate(data_monthly.loc[i+1,col[idx+1]], xy=xy, va='center', ha='center')
        for i, xy in enumerate(zip(data_monthly.index, data_monthly[col[idx+9]])):
            if i == 0:
                ax.annotate(data_monthly.loc[i+1,col[idx+10]], xy=xy, xytext=(xy[0]+0.3, xy[1]+0.4), color=colors[1], fontsize=ftsize)
            elif i == 11:
                ax.annotate(data_monthly.loc[i+1,col[idx+10]], xy=xy, xytext=(xy[0]-0.8, xy[1]+0.5), color=colors[1], fontsize=ftsize)
            else:
                ax.annotate(data_monthly.loc[i+1,col[idx+10]], xy=xy, xytext=(xy[0]-0.2, xy[1]-0.5), color=colors[1], fontsize=ftsize)

    if idx == 5:
        ax.set_ylabel('Cumulative ' + col[idx].replace(' SIO', ''))
    else:
        ax.set_ylabel(col[idx].replace(' SIO', ''))

    ax.set_xticklabels(['','Feb','Apr','Jun','Aug','Oct','Dec'])
    ax.legend().set_visible(False)

axs[1,2].set_ylim(bottom=0)
handles, _ = axs[0,0].get_legend_handles_labels()
labels = ['Sion', 'Tanikon']
lgd = fig2_1.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04), bbox_transform=fig2_1.transFigure, ncol=2)

fig2_1.tight_layout()
fig2_1.savefig('./Figures/'+'monthly_avg2.png', dpi=figure_res, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

# %% Precipitation vs PM10
fig2_2, axs = plt.subplots(1,2,figsize=(14,5))

data_sion.set_index('Date')[['PM10 [ug/m3]', 'Prec [mm]']].plot(ax=axs[0], color=['darkgray', 'cornflowerblue'])
#data_sion[['Date', 'Temp [C]']].set_index('Date').rolling(window='24h').mean().plot(ax=axs[0], color='coral', linewidth=1)
axs[0].lines[0].set_linewidth(0.8)
axs[0].lines[1].set_linewidth(1.5)
axs[0].set_ylabel('Precipitation [mm/hour] / PM10 [ug/m3]')
axs[0].set_xlabel(None)
axs[0].set_title('Sion')
axs[0].legend().set_visible(False)

#data_tanikon[['Date', 'Temp [C]']].set_index('Date').rolling(window='24h').mean().plot(ax=axs[1], color='coral')
data_tanikon.set_index('Date')[['PM10 [ug/m3]', 'Prec [mm]']].plot(ax=axs[1], color=['darkgray', 'cornflowerblue'])
axs[1].lines[0].set_linewidth(0.8)
axs[1].lines[1].set_linewidth(1.7)
axs[1].set_ylabel('Precipitation [mm/hour] / PM10 [ug/m3]')
axs[1].set_xlabel(None)
axs[1].set_title('Tanikon')
axs[1].legend().set_visible(False)

handles, _ = axs[0].get_legend_handles_labels()
labels = ['Precipitation [mm/hour]', 'PM10 [ug/m3]']
lgd = fig2_2.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.06), bbox_transform=fig2_2.transFigure, ncol=4)

fig2_2.tight_layout()
fig2_2.savefig('./Figures/'+'Prec_vs_PM10.png', dpi=figure_res, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

# %% Spearman correlation coefficients
#_______________________________________________________________________________

corr = [monthly_sion.corr(method='spearman'), monthly_tanikon.corr(method='spearman')]
loc = ['Sion', 'Tänikon']

fig2_3, axs = plt.subplots(1,2,figsize=(13, 6))

for idx, ax in enumerate(axs):
    ax.matshow(corr[idx], cmap='RdBu_r', alpha=0.8, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr[idx].columns)))
    ax.set_yticks(range(len(corr[idx].columns)))
    ax.set_xticklabels(corr[idx].columns, rotation = 30, ha='right');
    ax.set_yticklabels(corr[idx].columns);
    ax.set_title('Spearman correlation map for '+loc[idx])
    ax.xaxis.tick_bottom()

    for (i, j), z in np.ndenumerate(corr[idx]):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=11)

fig2_3.tight_layout()
fig2_3.savefig('./Figures/'+'monthly_corr.png', dpi=figure_res, bbox_inches='tight')
plt.show()

# <*!>

#_______________________________________________________________________________
# <*> Hourly Evolution - Weekend vs Weekday

pollutant_list = ['O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3]']
meteo_list = ['Temp [C]', 'Prec [mm]', 'RAD [W/m2]', 'windspeed [m/s]','wind direction [°N]']

# filter weekday and groupby Date.hour to get the hourly average of a weekday fro Sion
data_weekday_sion = data_sion[data_sion['Day type'] == 'weekday'].groupby(data_sion.Date.dt.hour).agg(mean_dict)
data_weekday_sion_std = data_sion[data_sion['Day type'] == 'weekday'].groupby(data_sion.Date.dt.hour).agg(std_dict)
data_weekday_sion['wind direction [°N]'] = data_weekday_sion['wind direction [°N]'].apply(dirAngle2str)
data_weekday_sion = data_weekday_sion.reindex(columns=pollutant_list+meteo_list)

# filter weekend and groupby Date.hour to get the hourly average of a weekend for Sion
data_weekend_sion = data_sion[data_sion['Day type'] == 'weekend'].groupby(data_sion.Date.dt.hour).agg(mean_dict)
data_weekend_sion_std = data_sion[data_sion['Day type'] == 'weekend'].groupby(data_sion.Date.dt.hour).agg(std_dict)
data_weekend_sion['wind direction [°N]'] = data_weekend_sion['wind direction [°N]'].apply(dirAngle2str)
data_weekend_sion = data_weekend_sion.reindex(columns=pollutant_list+meteo_list)

# merge the two temporaray dataframe
data_week_sion = pd.merge(data_weekday_sion, data_weekend_sion, how='inner', left_index=True, right_index=True, suffixes=(' weekday', ' weekend'))
data_week_sion_std = pd.merge(data_weekday_sion_std, data_weekend_sion_std, how='inner', left_index=True, right_index=True, suffixes=(' weekday', ' weekend'))

# filter weekday and groupby Date.hour to get the hourly average of a weekday for Tanikon
data_weekday_tanikon = data_tanikon[data_tanikon['Day type'] == 'weekday'].groupby(data_tanikon.Date.dt.hour).agg(mean_dict)
data_weekday_tanikon_std = data_tanikon[data_tanikon['Day type'] == 'weekday'].groupby(data_tanikon.Date.dt.hour).agg(std_dict)
data_weekday_tanikon['wind direction [°N]'] = data_weekday_tanikon['wind direction [°N]'].apply(dirAngle2str)
data_weekday_tanikon = data_weekday_tanikon.reindex(columns=pollutant_list+meteo_list)

# filter weekend and groupby Date.hour to get the hourly average of a weekend for Tanikon
data_weekend_tanikon = data_tanikon[data_sion['Day type'] == 'weekend'].groupby(data_tanikon.Date.dt.hour).agg(mean_dict)
data_weekend_tanikon_std = data_tanikon[data_sion['Day type'] == 'weekend'].groupby(data_tanikon.Date.dt.hour).agg(std_dict)
data_weekend_tanikon['wind direction [°N]'] = data_weekend_tanikon['wind direction [°N]'].apply(dirAngle2str)
data_weekend_tanikon = data_weekend_tanikon.reindex(columns=pollutant_list+meteo_list)

# merge the two temporaray dataframes
data_week_tanikon = pd.merge(data_weekday_tanikon, data_weekend_tanikon, how='inner', left_index=True, right_index=True, suffixes=(' weekday', ' weekend'))
data_week_tanikon_std = pd.merge(data_weekday_tanikon_std, data_weekend_tanikon_std, how='inner', left_index=True, right_index=True, suffixes=(' weekday', ' weekend'))

# get the final dataframe for plotting
data_week = pd.merge(data_week_sion, data_week_tanikon, how='inner', left_index=True, right_index=True, suffixes=(' SIO', ' TAE'))
data_week_std = pd.merge(data_week_sion_std, data_week_tanikon_std, how='inner', left_index=True, right_index=True, suffixes=(' SIO', ' TAE'))

# %% plot the data
#_______________________________________________________________________________

col = data_week.columns.to_list()
colors_2 = ['lightcoral','skyblue']
colors_text=['lightcoral', 'skyblue']
titles = ['O3 Sion','O3 Tanikon','NO2 Sion','NO2 Tanikon','PM10 Sion','PM10 Tanikon','NOX Sion','NOX Tanikon','Temperature Sion','Temperature Tanikon','Precipitation Sion','Precipitation Tanikon','Radiation Sion','Radiation Tanikon','Wind speed Sion','Wind speed Tanikon']
ftsize = 9
# reordonate columns name for easier plotting
col = [name for sublist in [col[i:18:9]+col[18+i:36:9] for i in range(9)] for name in sublist]

fig3_1, axs = plt.subplots(4,4,figsize=(14,14))

for idx, ax in enumerate(axs.reshape(-1)):
    if not idx in [10,11]:
        ax.fill_between(data_week.index, data_week[col[2*idx]]+data_week_std[col[2*idx]], data_week[col[2*idx]]-data_week_std[col[2*idx]], facecolor=colors_2[0], alpha=0.3)
        ax.fill_between(data_week.index, data_week[col[2*idx+1]]+data_week_std[col[2*idx+1]], data_week[col[2*idx+1]]-data_week_std[col[2*idx+1]], facecolor=colors_2[1], alpha=0.3)
        ax.set_ylabel(col[2*idx].replace(' weekday',''))
    else:
        ax.set_ylabel('Cumulative ' + col[2*idx].replace(' weekday',''))

    data_week[col[2*idx:2*idx+2]].plot(ax=ax, color=colors_2, linewidth=width)
    ax.set_title(titles[idx])
    ax.set_xlabel(None)
    ax.legend().set_visible(False)
    ax.set_xticks(np.arange(1,24,4))
    ax.set_xticklabels([str(i)+'h' for i in np.arange(1,24,4)])

    if idx in [14,15]:
        for i, xy in enumerate(zip(data_week.index, data_week[col[28+2*int(idx/15)]])):
            if i%2 != 0 and i%23 != 0:
                ax.annotate(data_week.loc[i,col[32+2*int(idx/15)]], xy=xy, xytext=(xy[0]+0.2, xy[1]+0.6), color=colors_text[0], fontsize=ftsize)

        for i, xy in enumerate(zip(data_week.index, data_week[col[28+2*int(idx/15)+1]])):
            if i%2 != 0 and i%23 != 0:
                ax.annotate(data_week.loc[i,col[32+2*int(idx/15)+1]], xy=xy, xytext=(xy[0]+0.2, xy[1]-0.6), color=colors_text[1], fontsize=ftsize)


for i in range(4):
    for j in range(0,3,2):
        axs[i,j].get_shared_y_axes().join(axs[i,j], axs[i,j+1])
        axs[i,j].autoscale(axis='y')

handles, labels = axs[0,0].get_legend_handles_labels()
labels = [lbl.replace('O3 [ug/m3] ','').replace(' SIO','') for lbl in labels]
lgd = fig3_1.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), bbox_transform=fig3_1.transFigure, ncol=2)

fig3_1.tight_layout()
fig3_1.savefig('./Figures/'+'daily_avg.png', dpi=figure_res, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

# %% Correlation Plots

corr = [data_week_sion.corr(method='spearman'), data_week_tanikon.corr(method='spearman')]
loc = ['Sion', 'Tänikon']

fig3_2, axs = plt.subplots(1,2,figsize=(20, 9))

for idx, ax in enumerate(axs):
    ax.matshow(corr[idx], cmap='RdBu_r', alpha=0.8)
    ax.set_xticks(range(len(corr[idx].columns)))
    ax.set_yticks(range(len(corr[idx].columns)))
    ax.set_xticklabels(corr[idx].columns, rotation = 30, ha='right');
    ax.set_yticklabels(corr[idx].columns);
    ax.set_title('Spearman correlation map for '+loc[idx])
    ax.xaxis.tick_bottom()

    for (i, j), z in np.ndenumerate(corr[idx]):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=11)

fig3_2.tight_layout()
fig3_2.savefig('./Figures/'+'daily_corr.png', dpi=figure_res, bbox_inches='tight')
plt.show()

# <*!>

#_______________________________________________________________________________
# <*> Legal Limits overpass

# %%
limits_annual = {'NO2':30,'PM10':20}
limits_daily = {'NO2':80,'PM10':50}
limits_hourly = {'O3':120}

# average on 24h periods
sion_24h_mean = data_sion[['Date', 'O3 [ug/m3]','NO2 [ug/m3]','PM10 [ug/m3]','NOX [ug/m3]', 'Temp [C]']].set_index('Date').rolling(window='24h').mean()
tanikon_24h_mean = data_tanikon[['Date', 'O3 [ug/m3]','NO2 [ug/m3]','PM10 [ug/m3]','NOX [ug/m3]']].set_index('Date').rolling(window='24h').mean()

# annual means
sion_annual_mean = data_sion[['O3 [ug/m3]','NO2 [ug/m3]','PM10 [ug/m3]','NOX [ug/m3]']].mean(axis=0)
tanikon_annual_mean = data_tanikon[['O3 [ug/m3]','NO2 [ug/m3]','PM10 [ug/m3]','NOX [ug/m3]']].mean(axis=0)

# Hourly mean is simply the data for the pollutant
sion_hourly_mean = data_sion[['Date','O3 [ug/m3]','NO2 [ug/m3]','PM10 [ug/m3]','NOX [ug/m3]']]
tanikon_hourly_mean = data_tanikon[['Date','O3 [ug/m3]','NO2 [ug/m3]','PM10 [ug/m3]','NOX [ug/m3]']]

# Check periods where limits are exceeded on the 24h averages : for NO2 and PM10

check_24h_limit(sion_24h_mean, limits_daily['PM10'], 'PM10', 'Sion')
check_24h_limit(sion_24h_mean, limits_daily['NO2'], 'NO2', 'Sion')
check_24h_limit(tanikon_24h_mean, limits_daily['PM10'], 'PM10', 'Tanikon')
check_24h_limit(tanikon_24h_mean, limits_daily['NO2'], 'NO2', 'Tanikon')

# Check if limits are exceeded on the annual averages : for NO2 and PM10
check_annual_limit(sion_annual_mean, limits_annual['PM10'], 'PM10', 'Sion')
check_annual_limit(sion_annual_mean, limits_annual['NO2'], 'NO2', 'Sion')
check_annual_limit(tanikon_annual_mean, limits_annual['PM10'], 'PM10', 'Tanikon')
check_annual_limit(tanikon_annual_mean, limits_annual['NO2'], 'NO2', 'Tanikon')

# Check if the limits are exceed on the hourly averages for O3
periods_above_hourly = dict()
hour_above_hourly = dict()

check_hourly_limit(sion_hourly_mean, limits_hourly['O3'], 'O3', 'Sion', periods_above_hourly, hour_above_hourly)
check_hourly_limit(tanikon_hourly_mean, limits_hourly['O3'], 'O3', 'Tanikon', periods_above_hourly, hour_above_hourly)

periods_above_daily = {'Sion_NO2':1,'Sion_PM10':0,'Tanikon_NO2':0,'Tanikon_PM10':2}
hour_above_daily = {'Sion_NO2':11,'Sion_PM10':0,'Tanikon_NO2':0,'Tanikon_PM10':39}

# %% Plots
#______________________________________________________________________________
fig5_2, axs = plt.subplots(3,2,figsize=(15,15), sharey='row')

df = [sion_hourly_mean.set_index('Date'), tanikon_hourly_mean.set_index('Date')]
titles = ['O3 limit in Sion', 'O3 limit in Tanikon']
keys = ['Sion_O3','Tanikon_O3']

for idx, ax in enumerate(axs[0,:]):
    df[idx].loc[:,'O3 [ug/m3]'].plot(ax=ax, color='gray', linewidth=.3)
    ax.axhline(y=limits_hourly['O3'], linestyle='--', color='black', linewidth=1.5, label='Legal limit')
    ax.fill_between(df[idx].index, df[idx].loc[:,'O3 [ug/m3]'], limits_hourly['O3'], where=(df[idx].loc[:,'O3 [ug/m3]'])>=limits_hourly['O3'], color='red', alpha=0.8, label='Overpassed limit')
    ax.set_title(titles[idx])
    ax.set_xlabel(None)
    ax.set_ylabel('Hourly mean of O3 [ug/m3]')

    ax.text(0.02,0.9,'Number of periods above {0} [ug/m3] : {1} \nNumber of hours above {0} [ug/m3] : {2}h'.format(limits_hourly['O3'], periods_above_hourly[keys[idx]],hour_above_hourly[keys[idx]]), transform=ax.transAxes)

# -----------
df = [sion_24h_mean, tanikon_24h_mean]
titles = ['NO2 limit in Sion', 'NO2 limit in Tanikon']
keys = ['Sion_NO2','Tanikon_NO2']

for idx, ax in enumerate(axs[1,:]):
    df[idx].loc[:,'NO2 [ug/m3]'].plot(ax=ax, color='gray', linewidth=.9)
    ax.axhline(y=limits_daily['NO2'], linestyle='--', color='black', linewidth=1.5, label='Legal limit')
    ax.fill_between(df[idx].index, df[idx].loc[:,'NO2 [ug/m3]'], limits_daily['NO2'], where=(df[idx].loc[:,'NO2 [ug/m3]'])>=limits_daily['NO2'], color='red', alpha=0.8, label='Overpassed limit')
    ax.set_title(titles[idx])
    ax.set_xlabel(None)
    ax.set_ylabel('24 mean of NO2 [ug/m3]')

    ax.text(0.02,0.9,'Number of periods above {0} [ug/m3] : {1} \nNumber of hours above {0} [ug/m3] : {2}h'.format(limits_daily['NO2'], periods_above_daily[keys[idx]],hour_above_daily[keys[idx]]), transform=ax.transAxes)
axs[1,0].set_ylim(0,100)
# ------------
df = [sion_24h_mean, tanikon_24h_mean]
titles = ['PM10 limit in Sion', 'PM10 limit in Tanikon']
keys = ['Sion_PM10','Tanikon_PM10']

for idx, ax in enumerate(axs[2,:]):
    df[idx].loc[:,'PM10 [ug/m3]'].plot(ax=ax, color='gray', linewidth=.9)
    ax.axhline(y=limits_daily['PM10'], linestyle='--', color='black', linewidth=1.5, label='Legal limit')
    ax.fill_between(df[idx].index, df[idx].loc[:,'PM10 [ug/m3]'], limits_daily['PM10'], where=(df[idx].loc[:,'PM10 [ug/m3]'])>=limits_daily['PM10'], color='red', alpha=0.8, label='Overpassed limit')
    ax.set_title(titles[idx])
    ax.set_xlabel(None)
    ax.set_ylabel('24 mean of PM10 [ug/m3]')

    ax.text(0.02,0.9,'Number of periods above {0} [ug/m3] : {1} \nNumber of hours above {0} [ug/m3] : {2}h'.format(limits_daily['PM10'], periods_above_daily[keys[idx]],hour_above_daily[keys[idx]]), transform=ax.transAxes)
axs[2,0].set_ylim(0,70)

handles, labels = axs[2,0].get_legend_handles_labels()
labels[0] = 'Pollutant [ug/m3]'
lgd = fig5_2.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), bbox_transform=fig5_2.transFigure, ncol=3)

fig5_2.tight_layout()
fig5_2.savefig('./Figures/'+'limit.png', dpi=figure_res, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
# <*!>

#<*!>

# <*> PART 2

# <*> Mean Weekend-Weekday
# %% extract subdataframe of weekend and weekday values
data_weekend = pd.merge(data_sion.loc[data_sion['Day type'] == 'weekend', ['Date', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3]']], \
                        data_tanikon.loc[data_tanikon['Day type'] == 'weekend', ['Date', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3]']], \
                        how='inner', right_on='Date', left_on='Date', suffixes=(' SIO', ' TAE')).drop(columns={'Date'})

data_weekday = pd.merge(data_sion.loc[data_sion['Day type'] == 'weekday', ['Date', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3]']], \
                        data_tanikon.loc[data_tanikon['Day type'] == 'weekday', ['Date', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3]']], \
                        how='inner', right_on='Date', left_on='Date', suffixes=(' SIO', ' TAE')).drop(columns={'Date'})

confint_weekend = conf_int_bootstrap(data_weekend)
confint_weekday = conf_int_bootstrap(data_weekday)

# %% Boxplot for each pollutant
boxdict = dict(linestyle='-', linewidth=2, color='dimgray')
whiskerdict = dict(linestyle='-', linewidth=2, color='dimgray')
mediandict = dict(linestyle='-', linewidth=1.5, color='tomato')
meandict = dict(marker='o' ,markerfacecolor='forestgreen', markeredgecolor=None, markersize=5, linewidth=0)

tickslabels = ['Sion weekday', 'Sion weekend', 'Tänikon weekday', 'Tänikon weekend']
titles = ['$\mathregular{O_3}$','$\mathregular{NO_2}$','PM10','$\mathregular{NO_X}$']
ylab = ['$\mathregular{O_3 \ [\mu g/m^3]}$','$\mathregular{NO_2 \ [\mu g/m^3]}$','$\mathregular{PM10 \ [\mu g/m^3]}$','$\mathregular{NO_X \ [\mu g/m^3]}$']
dot_size = 25
pos = [1,2,3.2,4.2]

plt.rcParams.update({'font.size': 14})

fig6_1, axs = plt.subplots(nrows=2,ncols=4,figsize=(24,15), gridspec_kw={'height_ratios':[3,1]})

for idx, (ax1, ax2) in enumerate(zip(axs[0,:].reshape(-1), axs[1,:].reshape(-1))):
    values = [data_weekday.iloc[:,idx].dropna(), data_weekend.iloc[:,idx].dropna(), data_weekday.iloc[:,idx+4].dropna(), data_weekend.iloc[:,idx+4].dropna()]
    lower = [confint_weekday.iloc[0,idx], confint_weekend.iloc[0,idx], confint_weekday.iloc[0,idx+4], confint_weekend.iloc[0,idx+4]]
    upper = [confint_weekday.iloc[1,idx], confint_weekend.iloc[1,idx], confint_weekday.iloc[1,idx+4], confint_weekend.iloc[1,idx+4]]
    means = [val.mean(axis=0) for val in values]
    err_means_lower = [m-l for m,l in zip(means, lower)]
    err_means_upper = [u-m for m,u in zip(means, upper)]

    ax1.boxplot(values, \
                positions=pos, notch=True, bootstrap=5000, \
                widths = 0.7, showfliers=False, showcaps=False, showmeans=False, boxprops=boxdict, whiskerprops=whiskerdict, medianprops=mediandict, meanprops=meandict)
    ax2.boxplot(values, \
                positions=pos, notch=True, bootstrap=5000, \
                widths = 0.7, showfliers=False, showcaps=False, showmeans=False, boxprops=boxdict, whiskerprops=whiskerdict, medianprops=mediandict, meanprops=meandict)
    ax1.errorbar(pos, means, yerr=[err_means_lower, err_means_upper], elinewidth=2.2, capsize=2, **meandict)
    ax2.errorbar(pos, means, yerr=[err_means_lower, err_means_upper], elinewidth=3, capsize=3, capthick=3, **meandict)

    for k in range(len(values)):
        ax1.scatter(np.random.normal(pos[k], 0.05, values[k].shape[0]), values[k], c='darkgray', alpha=0.5, marker='.', s=dot_size, lw = 0)
        ax2.scatter(np.random.normal(pos[k], 0.05, values[k].shape[0]), values[k], c='darkgray', alpha=0.5, marker='.', s=dot_size, lw = 0)

    ax2.set_xticklabels(tickslabels, rotation=45, ha="right")
    ax1.set_xticklabels(4*[None])
    ax1.grid(True, 'both', 'y')
    ax1.set_title(titles[idx])
    ax1.set_ylabel(ylab[idx])
    ax1.set_ylim(bottom=0)
    #d = max([m+e for m,e in zip(means, err_means_upper)])-min([m-e for m,e in zip(means, err_means_lower)])
    d = max(upper)-min(lower)
    ax2.set_ylim([min(lower)-0.05*d, max(upper)+0.05*d])

handles = [plt.Line2D((0,1),(0,0), **mediandict), plt.Line2D((0,1),(0,0), **meandict)]
labels = ['median','mean']
lgd = fig6_1.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), bbox_transform=fig6_1.transFigure, ncol=2)

fig6_1.tight_layout()
fig6_1.savefig('./Figures/'+'boxplots_pollutants.png', dpi=figure_res, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()


# %% Statistical test to compare means : Wilcoxon signed rank testtest --> tests if x and y are sampled from the same distributions --> compare two independant means without assuming normal distribution of the samples
# if samples are dependant -> use Mann-Whitney-U test

index_names = ['O3', 'NO2', 'PM10', 'NOX']
cols = ['O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3]']

pval_df_pollutant = comparisonTests(data_weekday, data_weekend, index_names, cols, 'mannwhitneyu')
pval_df_pollutant_dist = comparisonTests(data_weekday, data_weekend, index_names, cols, 'kolmogorovsmirnov')

# %% same for meteo
data_weekend = pd.merge(data_sion.loc[data_sion['Day type'] == 'weekend', ['Date', 'Temp [C]', 'Prec [mm]', 'RAD [W/m2]', 'wind direction [°N]', 'windspeed [m/s]']], \
                        data_tanikon.loc[data_tanikon['Day type'] == 'weekend', ['Date', 'Temp [C]', 'Prec [mm]', 'RAD [W/m2]', 'wind direction [°N]', 'windspeed [m/s]']], \
                        how='inner', right_on='Date', left_on='Date', suffixes=(' SIO', ' TAE')).drop(columns={'Date'})

data_weekday = pd.merge(data_sion.loc[data_sion['Day type'] == 'weekday', ['Date', 'Temp [C]', 'Prec [mm]', 'RAD [W/m2]', 'wind direction [°N]', 'windspeed [m/s]']], \
                        data_tanikon.loc[data_tanikon['Day type'] == 'weekday', ['Date', 'Temp [C]', 'Prec [mm]', 'RAD [W/m2]', 'wind direction [°N]', 'windspeed [m/s]']], \
                        how='inner', right_on='Date', left_on='Date', suffixes=(' SIO', ' TAE')).drop(columns={'Date'})

confint_weekend = conf_int_bootstrap(data_weekend)
confint_weekday = conf_int_bootstrap(data_weekday)
confint_weekday
confint_weekend

# %% Boxplot for each meteorological parameters
boxdict = dict(linestyle='-', linewidth=2, color='dimgray')
whiskerdict = dict(linestyle='-', linewidth=2, color='dimgray')
mediandict = dict(linestyle='-', linewidth=1.5, color='tomato')
meandict = dict(marker='o' ,markerfacecolor='forestgreen', markeredgecolor=None, markersize=5, linewidth=0)

tickslabels = ['Sion weekday', 'Sion weekend', 'Tänikon weekday', 'Tänikon weekend']
titles = ['Temperature','Precipitation','Radiation','Wind direction', 'Wind speed']
ylab = ['Temperature [°C]','Precipitation [mm/h]','Radiation $\mathregular{[W/m2]}$','Wind direction [°N]', 'Wind speed [m/s]']
dot_size = 25
pos = [1,2,3.2,4.2]

fig6_2, axs = plt.subplots(2,5,figsize=(24,15), gridspec_kw={'height_ratios':[3,1]})

for idx, (ax1, ax2) in enumerate(zip(axs[0,:].reshape(-1), axs[1,:].reshape(-1))):
    values = [data_weekday.iloc[:,idx].dropna(), data_weekend.iloc[:,idx].dropna(), data_weekday.iloc[:,idx+5].dropna(), data_weekend.iloc[:,idx+5].dropna()]
    lower = [confint_weekday.iloc[0,idx], confint_weekend.iloc[0,idx], confint_weekday.iloc[0,idx+5], confint_weekend.iloc[0,idx+5]]
    upper = [confint_weekday.iloc[1,idx], confint_weekend.iloc[1,idx], confint_weekday.iloc[1,idx+5], confint_weekend.iloc[1,idx+5]]
    means = [val.mean(axis=0) for val in values]
    err_means_lower = [m-l for m,l in zip(means, lower)]
    err_means_upper = [u-m for m,u in zip(means, upper)]

    ax1.boxplot(values, \
                positions=pos, notch=True, bootstrap=5000, \
                widths = 0.7, showfliers=False, showcaps=False, showmeans=False, boxprops=boxdict, whiskerprops=whiskerdict, medianprops=mediandict, meanprops=meandict)
    ax2.boxplot(values, \
                positions=pos, notch=True, bootstrap=5000, \
                widths = 0.7, showfliers=False, showcaps=False, showmeans=False, boxprops=boxdict, whiskerprops=whiskerdict, medianprops=mediandict, meanprops=meandict)
    ax1.errorbar(pos, means, yerr=[err_means_lower, err_means_upper], elinewidth=2.2, capsize=2, **meandict)
    ax2.errorbar(pos, means, yerr=[err_means_lower, err_means_upper], elinewidth=3, capsize=3, capthick=3, **meandict)

    for k in range(len(values)):
        ax1.scatter(np.random.normal(pos[k], 0.05, values[k].shape[0]), values[k], c='darkgray', alpha=0.5, marker='.', s=dot_size, lw = 0)
        ax2.scatter(np.random.normal(pos[k], 0.05, values[k].shape[0]), values[k], c='darkgray', alpha=0.5, marker='.', s=dot_size, lw = 0)

    ax2.set_xticklabels(tickslabels, rotation=45, ha="right")
    ax1.set_xticklabels(4*[None])
    ax1.grid(True, 'both', 'y')
    ax1.set_title(titles[idx])
    ax1.set_ylabel(ylab[idx])
    ax1.set_ylim(bottom=0)
    d = max(upper)-min(lower)
    ax2.set_ylim([min(lower)-0.05*d, max(upper)+0.05*d])

handles = [plt.Line2D((0,1),(0,0), **mediandict), plt.Line2D((0,1),(0,0), **meandict)]
labels = ['median','mean']
lgd = fig6_2.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), bbox_transform=fig6_2.transFigure, ncol=2)

fig6_2.tight_layout()
fig6_2.savefig('./Figures/'+'boxplots_meteo.png', dpi=figure_res, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

# %% Statistical test to compare means : Mann-Whitney-U test --> tests if x and y are sampled from the same distributions --> compare two independant means without assuming normal distribution of the samples
# if samples are dependant -> use Wilcoxon signed rank test
index_names = ['Temperature', 'Precipitation', 'Radiation', 'Wind direction', 'Wind speed']
cols = ['Temp [C]', 'Prec [mm]', 'RAD [W/m2]', 'wind direction [°N]', 'windspeed [m/s]']

pval_df_meteo = comparisonTests(data_weekday, data_weekend, index_names, cols, 'mannwhitneyu')
pval_df_meteo_dist = comparisonTests(data_weekday, data_weekend, index_names, cols, 'kolmogorovsmirnov')

# %% merge the two p_val dataframes and save it in csv
# for mannwhitneyu test
pval_df = pval_df_pollutant.append(pval_df_meteo).astype(float)
pd.set_option('display.float_format', '{:.3g}'.format)
pval_df
pval_df.to_csv('data_output/mannwhitneyu_pval.csv', sep=',', encoding='utf-8', float_format='%.3g')

# for kolmogorov-smirnov test
pval_df_dist = pval_df_pollutant_dist.append(pval_df_meteo_dist).astype(float)
pd.set_option('display.float_format', '{:.3g}'.format)
pval_df_dist
pval_df_dist.to_csv('data_output/ks_pval.csv', sep=',', encoding='utf-8', float_format='%.3g')

# <*!>

# <*> Correlation & Lagged correlations

# %% Correlation (and lagged correlation) map with the whole data (year) for Sion and Tanikon
c1, l1 = crossCorr(data_sion.drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(0,0))
c2, l2 = crossCorr(data_sion.drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(12,12))
c3, l3 = crossCorr(data_tanikon.drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(0,0))
c4, l4 = crossCorr(data_tanikon.drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(12,12))

corr = [c1,c2,c3,c4]
lag = [l1,l2,l3,l4]
titles = ['Correlation', 'Lagged correlation']
labels = ['Sion', '', 'Tänikon']

fig7_1, axs = plt.subplots(2,2,figsize=(16,16))

for idx, ax in enumerate(axs.reshape(-1)):
    ax.matshow(corr[idx], cmap='RdBu_r', alpha=0.8, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr[idx].columns)))
    ax.set_yticks(range(len(corr[idx].columns)))

    if idx in [2,3]:
        ax.set_xticklabels(corr[idx].columns, rotation = 30, ha='right')
    else:
        ax.set_xticklabels([None]*8)
        ax.set_title(titles[idx])
    if idx in [0,2]:
        ax.set_yticklabels(corr[idx].columns)
        ax.set_ylabel(labels[idx], fontsize=18)
    else:
        ax.set_yticklabels([None]*8)

    ax.xaxis.tick_bottom()

    for (i, j), z in np.ndenumerate(corr[idx]):
        ax.text(i, j, '{0:0.2f}\nlag {1}h'.format(z, int(lag[idx].iloc[i,j])), ha='center', va='center', fontsize=11)

fig7_1.tight_layout()
fig7_1.savefig('./Figures/yearly_correlations.png', dpi=figure_res)
plt.show()

# %% scatterplot of O3 in function of others by season for SIO and TAE
seasons = ['winter', 'spring', 'summer', 'autumn']
colors = ['lightskyblue','yellowgreen','lightcoral','burlywood']
prop_dict = dict(linewidth=0, marker='o', markersize=3, markeredgewidth=0.0, alpha=0.9)
cols = data_sion.columns.drop(['Season', 'Date', 'Day type', 'wind direction [°N]', 'O3 [ug/m3]'])
labels = ['$\mathregular{NO_2 \ [\mu g/m^3]}$', '$\mathregular{PM10 \ [\mu g/m^3]}$', '$\mathregular{NO_x \ [\mu g/m^3]}$', '$\mathregular{Temperature \ [\degree C]}$', '$\mathregular{Precipitation \ [mm]}$', '$\mathregular{Radiation \ [W/m^2]}$', '$\mathregular{wind speed \ [m/s]}$']

fig7_2, axs = plt.subplots(2,7, figsize=(20,6), sharex='col', sharey=True)

for idx in range(axs.shape[1]):
    for season, color in zip(seasons, colors):
        axs[0,idx].plot(data_sion.loc[data_sion['Season']==season, cols[idx]], data_sion.loc[data_sion['Season']==season, 'O3 [ug/m3]'], markerfacecolor=color, **prop_dict)
        axs[1,idx].plot(data_tanikon.loc[data_tanikon['Season']==season, cols[idx]], data_tanikon.loc[data_tanikon['Season']==season, 'O3 [ug/m3]'], markerfacecolor=color, **prop_dict)
        axs[1,idx].set_xlabel(labels[idx])

axs[0,0].set_ylabel('Sion \n\n$\mathregular{O_3 \ [\mu g/m^3]}$')
axs[1,0].set_ylabel('Tänikon \n\n$\mathregular{O_3 \ [\mu g/m^3]}$')

#handles, _ = axs[0,0].get_legend_handles_labels()
handles = [mpatch.Patch(facecolor=c) for c in colors]
labels = seasons
lgd = fig7_2.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), bbox_transform=fig7_2.transFigure, ncol=4)

fig7_2.tight_layout()
#fig7_2.savefig('./Figures/'+'scatter_ozone.png', dpi=figure_res, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()

# %% Plot O3 and NO2 scatter plots
seasons = ['winter', 'spring', 'summer', 'autumn']

corr_sion = [crossCorr(data_sion[data_sion['Season']==season].drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(0,0))[0] for season in seasons]
corr_sion.append(crossCorr(data_sion.drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(0,0))[0])
corr_tanikon = [crossCorr(data_tanikon[data_tanikon['Season']==season].drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(0,0))[0] for season in seasons]
corr_tanikon.append(crossCorr(data_tanikon.drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(0,0))[0])

colors = ['lightskyblue','yellowgreen','lightcoral','burlywood']
prop_dict = dict(linewidth=0, marker='o', markersize=2, markeredgewidth=0.0, alpha=0.9)
colsO3 = data_sion.columns.drop(['Season', 'Date', 'Day type', 'wind direction [°N]', 'O3 [ug/m3]'])
colsNO2 = data_sion.columns.drop(['Season', 'Date', 'Day type', 'wind direction [°N]', 'NO2 [ug/m3]'])
labelsO3 = ['$\mathregular{NO_2 \ [\mu g/m^3]}$', '$\mathregular{PM10 \ [\mu g/m^3]}$', '$\mathregular{NO_x \ [\mu g/m^3]}$', '$\mathregular{Temperature \ [\degree C]}$', '$\mathregular{Precipitation \ [mm]}$', '$\mathregular{Radiation \ [W/m^2]}$', '$\mathregular{wind speed \ [m/s]}$']
labelsNO2 = ['$\mathregular{O_3 \ [\mu g/m^3]}$', '$\mathregular{PM10 \ [\mu g/m^3]}$', '$\mathregular{NO_x \ [\mu g/m^3]}$', '$\mathregular{Temperature \ [\degree C]}$', '$\mathregular{Precipitation \ [mm]}$', '$\mathregular{Radiation \ [W/m^2]}$', '$\mathregular{wind speed \ [m/s]}$']

def textCorr(ax, corr, col1, col2, colors):
    x=0.02
    text = []
    for s, c in zip([corr[i].loc[col1,col2] for i in range(len(corr))], colors+['darkgray']):
        text.append(ax.text(x, 1.05, str('{:2.2f}'.format(s.round(2))) + " ", color=c, fontsize=9, fontweight='bold',  transform=ax.transAxes))
        x += 0.2
    return text

fig7_3, axs = plt.subplots(4,7, figsize=(18,12), sharey='row')
txt = []
for idx in range(axs.shape[1]):
    for season, color in zip(seasons, colors):
        # O3
        axs[0,idx].plot(data_sion.loc[data_sion['Season']==season, colsO3[idx]], data_sion.loc[data_sion['Season']==season, 'O3 [ug/m3]'], markerfacecolor=color, **prop_dict)
        axs[1,idx].plot(data_tanikon.loc[data_tanikon['Season']==season, colsO3[idx]], data_tanikon.loc[data_tanikon['Season']==season, 'O3 [ug/m3]'], markerfacecolor=color, **prop_dict)
        axs[1,idx].set_xlabel(labelsO3[idx])
        # NO2
        axs[2,idx].plot(data_sion.loc[data_sion['Season']==season, colsNO2[idx]], data_sion.loc[data_sion['Season']==season, 'NO2 [ug/m3]'], markerfacecolor=color, **prop_dict)
        axs[3,idx].plot(data_tanikon.loc[data_tanikon['Season']==season, colsNO2[idx]], data_tanikon.loc[data_tanikon['Season']==season, 'NO2 [ug/m3]'], markerfacecolor=color, **prop_dict)
        axs[3,idx].set_xlabel(labelsNO2[idx])

    txt += textCorr(axs[0,idx], corr_sion, 'O3 [ug/m3]', colsO3[idx], colors)
    _ = textCorr(axs[1,idx], corr_tanikon, 'O3 [ug/m3]', colsO3[idx], colors)
    _ = textCorr(axs[2,idx], corr_sion, 'NO2 [ug/m3]', colsNO2[idx], colors)
    _ = textCorr(axs[3,idx], corr_tanikon, 'NO2 [ug/m3]', colsNO2[idx], colors)

axs[0,0].set_ylabel('Sion \n\n$\mathregular{O_3 \ [\mu g/m^3]}$')
axs[1,0].set_ylabel('Tänikon \n\n$\mathregular{O_3 \ [\mu g/m^3]}$')
axs[2,0].set_ylabel('Sion \n\n$\mathregular{NO_2 \ [\mu g/m^3]}$')
axs[3,0].set_ylabel('Tänikon \n\n$\mathregular{NO_2 \ [\mu g/m^3]}$')

for i in range(0,3,2):
    for j in range(7):
        axs[i,j].get_shared_x_axes().join(axs[i,j], axs[i+1,j])
        axs[i,j].autoscale(axis='x')

#handles, _ = axs[0,0].get_legend_handles_labels()
handles = [mpatch.Patch(facecolor=c) for c in colors]
labels = seasons
lgd = fig7_3.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), bbox_transform=fig7_3.transFigure, ncol=4)

fig7_3.tight_layout()
fig7_3.savefig('./Figures/'+'scatter_O3_NO2.png', dpi=figure_res, bbox_extra_artists=(lgd,*txt), bbox_inches='tight')
plt.show()

# %%
seasons = ['winter', 'spring', 'summer', 'autumn']
corr_sion = [crossCorr(data_sion[data_sion['Season']==season].drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(0,0))[0] for season in seasons]
corr_sion.append(crossCorr(data_sion.drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(0,0))[0])
corr_tanikon = [crossCorr(data_tanikon[data_tanikon['Season']==season].drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(0,0))[0] for season in seasons]
corr_tanikon.append(crossCorr(data_tanikon.drop(columns={'Season', 'Day type', 'Date', 'wind direction [°N]'}), padding=(0,0))[0])

corr = [corr_sion, corr_tanikon]
data = [data_sion, data_tanikon]
colors = ['lightskyblue','yellowgreen','lightcoral','burlywood']
prop_dict = dict(linewidth=0, marker='o', markersize=2, markeredgewidth=0.0, alpha=0.9)
pollutant = ['O3 [ug/m3]','NO2 [ug/m3]']
cols = [data_sion.columns.drop(['Season', 'Date', 'Day type', 'wind direction [°N]', 'O3 [ug/m3]']), \
        data_sion.columns.drop(['Season', 'Date', 'Day type', 'wind direction [°N]', 'NO2 [ug/m3]'])]
xlabels = [['$\mathregular{NO_2 \ [\mu g/m^3]}$', '$\mathregular{PM10 \ [\mu g/m^3]}$', '$\mathregular{NO_x \ [\mu g/m^3]}$', '$\mathregular{Temperature \ [\degree C]}$', '$\mathregular{Precipitation \ [mm]}$', '$\mathregular{Radiation \ [W/m^2]}$', '$\mathregular{wind speed \ [m/s]}$'],\
           ['$\mathregular{O_3 \ [\mu g/m^3]}$', '$\mathregular{PM10 \ [\mu g/m^3]}$', '$\mathregular{NO_x \ [\mu g/m^3]}$', '$\mathregular{Temperature \ [\degree C]}$', '$\mathregular{Precipitation \ [mm]}$', '$\mathregular{Radiation \ [W/m^2]}$', '$\mathregular{wind speed \ [m/s]}$']]
ylabels = [['Sion \n\n$\mathregular{O_3 \ [\mu g/m^3]}$', 'Tänikon \n\n$\mathregular{O_3 \ [\mu g/m^3]}$'], \
           ['Sion \n\n$\mathregular{NO_2 \ [\mu g/m^3]}$', 'Tänikon \n\n$\mathregular{NO_2 \ [\mu g/m^3]}$']]

fig = plt.figure(figsize=(20, 20))

outer_big = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)
for p in range(outer_big._nrows):
    outer = gridspec.GridSpecFromSubplotSpec(2, 7, subplot_spec=outer_big[p], wspace=0.05, hspace=0.2)
    for r in range(outer._nrows):
        for c in range(outer._ncols):
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[r*7+c], hspace=0.1, height_ratios=[4,9])

            ax1 = plt.Subplot(fig, inner[0])
            ax2 = plt.Subplot(fig, inner[1])

            ax1.bar([i for i in range(len(corr[r]))], [corr[r][j].loc[pollutant[p],cols[p][c]] for j in range(len(corr[r]))], color=colors+['darkgray'])
            for j in range(len(corr[r])):
                val = corr[r][j].loc[pollutant[p],cols[p][c]]
                pos=(j,-0.25)
                if val <= 0 :
                    pos = (j,0.2)
                ax1.annotate(str('{:2.2f}'.format(val)), pos, color=(colors+['darkgray'])[j], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)

            ax1.plot([-1,5], [0,0], linewidth=0.5, linestyle='-', color='black')
            ax1.set_ylim([-1,1])
            ax1.set_xlim([-0.5,4.5])
            ax1.get_xaxis().set_visible(False)

            for season, color in zip(seasons, colors):
                ax2.plot(data[r].loc[data[r]['Season']==season, cols[p][c]], data[r].loc[data[r]['Season']==season, pollutant[p]], markerfacecolor=color, **prop_dict)

            if c!=0 :
                ax1.set_yticklabels([])
                ax2.set_yticklabels([])
            else:
                ax1.set_ylabel('Correlation')
                ax2.set_ylabel(ylabels[p][r])

            if r%2!=0: ax2.set_xlabel(xlabels[p][c])

            fig.add_subplot(ax1)
            fig.add_subplot(ax2)

handles = [mpatch.Patch(facecolor=c) for c in colors+['darkgray']]
labels = seasons+['yearly']
lgd = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.06), bbox_transform=fig.transFigure, ncol=5)

fig.savefig('./Figures/'+'scatter_hist_O3_NO2.png', dpi=figure_res, bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.show()

# <*!>

# <*> Analysis of July month
# %%
# get data for the month of july and for the pollutants concentration only
sion_july_hourly = data_sion.loc[data_sion['Date'].dt.month==7, ['Date', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3]']].set_index('Date')
tanikon_july_hourly = data_tanikon.loc[data_tanikon['Date'].dt.month==7, ['Date', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3]']].set_index('Date')

# get daily avergaes for both locations
sion_july_daily = sion_july_hourly.groupby(sion_july_hourly.index.day).mean()
tanikon_july_daily = tanikon_july_hourly.groupby(tanikon_july_hourly.index.day).mean()

# test for each pollutant if they are lognormally distributed
# --> take the log of each element and then test is normally distributed
# test normality with a Shapiro-Wilk test - H : the sample is normally distributed ;  A : the sample is not normally distributed

sion_july_hourly = sion_july_hourly.apply(np.log).dropna()
tanikon_july_hourly = tanikon_july_hourly.apply(np.log).dropna()
sion_july_daily = sion_july_daily.apply(np.log).dropna()
tanikon_july_daily = tanikon_july_daily.apply(np.log).dropna()

# %% QQ-plots
titles = ['Sion hourly', 'Tänikon hourly', 'Sion daily', 'Tänikon daily']
ylables = ['$\mathregular{O_3}$ \n Ordered Values', '$\mathregular{NO_2}$ \n Ordered Values', '$\mathregular{PM10}$ \n Ordered Values', '$\mathregular{NO_x}$ \n Ordered Values']

fig8_1, axs = plt.subplots(4,4, figsize=(12,11))

for idx2, (data, title) in enumerate(zip([sion_july_hourly, tanikon_july_hourly, sion_july_daily, tanikon_july_daily], titles)):
    for idx1, (c, ylabel) in enumerate(zip(list(sion_july_daily.columns), ylables)):
        res = stats.probplot(data[c], dist='norm', plot=axs[idx1,idx2]) #, sparams=(2.5,)
        axs[idx1,idx2].get_lines()[0].set_marker('o')
        axs[idx1,idx2].get_lines()[0].set_markerfacecolor('darkgray')
        axs[idx1,idx2].get_lines()[0].set_markeredgecolor('dimgray')
        axs[idx1,idx2].get_lines()[0].set_markersize(6.0)
        axs[idx1,idx2].get_lines()[1].set_linewidth(2.0)
        axs[idx1,idx2].get_lines()[1].set_color('red')

        if idx1 != 3:
            axs[idx1,idx2].set_xlabel(None)

        if idx2 != 0:
            axs[idx1,idx2].set_ylabel(None)
        else:
            axs[idx1,idx2].set_ylabel(ylabel)

        if idx1 == 0:
            axs[idx1,idx2].set_title(title)
        else:
            axs[idx1,idx2].set_title(None)

fig8_1.tight_layout()
fig8_1.savefig('./Figures/'+'QQplots.png', dpi=figure_res)
plt.show()

# %%
test = 'shapiro'

s1 = testNormality(sion_july_hourly, test=test)
s2 = testNormality(tanikon_july_hourly, test=test)
s3 = testNormality(sion_july_daily, test=test)
s4 = testNormality(tanikon_july_daily, test=test)

col = pd.MultiIndex.from_product([['hourly', 'daily'],['Sion', 'Tänikon']])
idx = ['O3','NO2','PM10','NOX']

pvals_df = pd.concat([s1,s2,s3,s4], axis=1)
pvals_df.columns = col
pvals_df.index = idx
pd.set_option('display.float_format', '{:.3g}'.format)
pvals_df

pvals_df.to_csv('data_output/'+test+'_pval.csv', sep=',', encoding='utf-8', float_format='%.3g')

# <*!>

# <*> Unusual Patterns

# %% plot time series to select unsual periods/peaks
# -> PM10 during winter in TAE ?
# -> Christmas Sion ?
#
# What is above a given limits ? point event ? or meteorological event (daily high values) ? only outlier (Dixon test) ?

fig, axs = plt.subplots(4,2,figsize=(15, 10))
cols = data_sion.columns

for idx, ax in enumerate(np.reshape(axs[:,0],-1)):
    ax.plot(data_sion.loc[:,'Date'], data_sion.loc[:,cols[idx+1]], linewidth=.4, color='darkgray', marker=None)
    ax.set_title(cols[idx+1])

for idx, ax in enumerate(np.reshape(axs[:,1],-1)):
    ax.plot(data_tanikon.loc[:,'Date'], data_tanikon.loc[:,cols[idx+1]], linewidth=.4, color='darkgray', marker=None)
    ax.set_title(cols[idx+1])

fig.tight_layout()
plt.show()
 # %%
fig, axs = plt.subplots(4,2,figsize=(15,10))
cols = data_tanikon.columns

for idx, ax in enumerate(np.reshape(axs[:,0],-1)):
    ax.plot(data_sion.loc[:,'Date'], data_sion.loc[:,cols[idx+5]], linewidth=.4, color='darkgray', marker=None)
    ax.set_title(cols[idx+5])

for idx, ax in enumerate(np.reshape(axs[:,1],-1)):
    ax.plot(data_tanikon.loc[:,'Date'], data_tanikon.loc[:,cols[idx+5]], linewidth=.4, color='darkgray', marker=None)
    ax.set_title(cols[idx+5])
fig.tight_layout()
plt.show()

# %% check when PM10 overpass 50 ug/m3 on daily average and 100 ug/m3 on hourly values
dates = data_sion.loc[data_sion['PM10 [ug/m3]'] > 100,'Date']

#%% PM10 in Sion hourly
cmap = cm.get_cmap('Oranges')
color_idx = [0.5+i/(2*len(dates)) for i in range(len(dates))]
fig, ax = plt.subplots(1,1,figsize=(8,6))
for date, color_idx in zip(dates, color_idx):
    d = date.day
    m = date.month
    ax.plot(range(24),data_sion.loc[(data_sion['Date']>=dt.datetime(2018,m,d,0)) & (data_sion['Date']<dt.datetime(2018,m,d+1,0)),'PM10 [ug/m3]'], lw=2, color=cmap(color_idx))
ax.plot([0,23],[100,100],linewidth=2, linestyle='--', color='black')
ax.set_xlim([0,23])
handles, _ = ax.get_legend_handles_labels()
labels = ['2018-{month:02d}-{day:02d}'.format(month=date.month, day=date.day) for date in dates]
lgd = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 0.86), bbox_transform=fig.transFigure, ncol=1)
plt.show()

#%%
data_sion_daily = data_sion.groupby([data_sion.Date.dt.month, data_sion.Date.dt.day]).agg(mean_dict)
data_sion_daily[data_sion_daily['PM10 [ug/m3]'] > 40]

#%% PM10 Tänikon hourly
dates = data_tanikon.loc[data_tanikon['PM10 [ug/m3]'] > 100, 'Date']

cmap = cm.get_cmap('Oranges')
color_idx = [0.5+i/(2*len(dates)) for i in range(len(dates))]
fig, ax = plt.subplots(1,1,figsize=(8,6))
for date, color_idx in zip(dates, color_idx):
    d = date.day
    m = date.month
    ax.plot(range(24),data_tanikon.loc[(data_tanikon['Date']>=dt.datetime(2018,m,d,0)) & (data_tanikon['Date']<dt.datetime(2018,m,d+1,0)),'PM10 [ug/m3]'], lw=2, color=cmap(color_idx))
ax.plot([0,23],[100,100],linewidth=2, linestyle='--', color='black')
ax.set_xlim([0,23])
handles, _ = ax.get_legend_handles_labels()
labels = ['2018-{month:02d}-{day:02d}'.format(month=date.month, day=date.day) for date in dates]
lgd = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 0.86), bbox_transform=fig.transFigure, ncol=1)
plt.show()

#%% PM10 Tanikon daily -> feb-mar high PM10
data_tanikon_daily = data_tanikon.groupby([data_tanikon.Date.dt.month, data_tanikon.Date.dt.day]).agg(mean_dict)
dates = data_tanikon_daily[data_tanikon_daily['PM10 [ug/m3]'] > 40]

# Test if PM10 is significantly higher than the whole data and if Temp is significantly lowert than the whole data
data_high = data_tanikon.loc[(data_tanikon['Date']>=dt.datetime(2018,2,19,0)) & (data_tanikon['Date']<dt.datetime(2018,3,4,0)),['PM10 [ug/m3]','Temp [C]']]
_ , p_val_PM10 = stats.mannwhitneyu(data_tanikon.loc[data_tanikon['Season']=='winter','PM10 [ug/m3]'], data_high['PM10 [ug/m3]'], alternative='less') # H : mean(PM10_all) = mean(PM10_period) A: mean(PM10_all) < mean(PM10_period)
_ , p_val_T = stats.mannwhitneyu(data_tanikon.loc[data_tanikon['Season']=='winter','Temp [C]'], data_high['Temp [C]'], alternative='greater') # H : mean(Temp_all) = mean(Temp_period) A: mean(Temp_all) > mean(Temp_period)
print('H : mean(PM10_all) = mean(PM10_period) \nA: mean(PM10_all) < mean(PM10_period) \n >>> p-value = {0:.3E}'.format(p_val_PM10))
print('H : mean(Temp_all) = mean(Temp_period) \nA: mean(Temp_all) > mean(Temp_period) \n >>> p-value = {0:.3E}'.format(p_val_T))

# %% PLot overview
fig = plt.figure(figsize=(10,9))
gs = gridspec.GridSpec(2,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,:])

# diurnal plot
cmap = cm.get_cmap('Greys')
color_idx = [0.4+i/(2.5*len(dates)) for i in range(len(dates))]

ax1.plot([0,23],[40,40],linewidth=1.5, linestyle='--', color='black')
dates = [dt.datetime(2018, date[0], date[1]) for date in dates.index]
dates = dates[2:6] # keep only the relevant dates (the period feb-mar)
for date, color_idx in zip(dates, color_idx):
    d = date.day
    m = date.month
    ax1.plot(range(24),data_tanikon.loc[(data_tanikon['Date']>=dt.datetime(2018,m,d,0)) & (data_tanikon['Date']<dt.datetime(2018,m,d,0)+dt.timedelta(days=1)),'PM10 [ug/m3]'], lw=2.5, color=cmap(color_idx))

ax1.set_xlim([0,23])
ax1.set_xlabel('Hour of the day')
ax1.set_ylabel('$\mathregular{PM10 \ [\mu g/m^3]}$')
ax1.set_title('Diurnal PM10 concentration above 40 $\mathregular{[\mu g/m^3]}$', loc='left')
handles, _ = ax1.get_legend_handles_labels()
labels = ['2018-{month:02d}-{day:02d}'.format(month=date.month, day=date.day) for date in dates]
lgd = fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 0), bbox_transform=ax1.transAxes, ncol=1)

# scatter plot
d1 = dt.datetime(2018,2,17,0)
d2 = dt.datetime(2018,3,10,0)
df = data_tanikon.loc[(data_tanikon['Date']>=d1) & (data_tanikon['Date']<d2),:]
pm10_T_corr = df[['PM10 [ug/m3]','Temp [C]']].corr(method='pearson')
b, m = polyfit(df['PM10 [ug/m3]'], df['Temp [C]'], 1)

ax2.plot(df['PM10 [ug/m3]'], df['Temp [C]'], lw=0, markersize=7, marker='o', markerfacecolor='gray', markeredgewidth=0, alpha=0.6)
ax2.plot(df['PM10 [ug/m3]'], b+m*df['PM10 [ug/m3]'], '-', color='indianred', linewidth=2.5)
ax2.annotate('$R^2 = $ {0:.2f}'.format(pm10_T_corr.iloc[0,1]), (0.7, 0.9), xycoords=ax2.transAxes, fontsize=13)
ax2.set_xlabel('PM10 $\mathregular{[\mu g/m^3]}$')
ax2.set_ylabel('Temperature [°C]')
ax2.set_title('PM10 vs Temperature', loc='left')

# overview plot
upprval = 75
date_list = [d1 + dt.timedelta(days=x) for x in range(0, (d2-d1).days)]
h = [upprval for i in range((d2-d1).days)]
dates_ext = set(dates + [date+dt.timedelta(days=1) for date in dates])
unsual_period = [(date in dates_ext) for date in date_list]

ax3b = ax3.twinx()
ax3b.plot(df['Date'], df['Temp [C]'], lw=2, color='coral')
ax3b.plot(df['Date'], df['Prec [mm]'], lw=2, color='lightskyblue')
ax3.plot(df['Date'], df['PM10 [ug/m3]'], lw=2.3, color='darkgray')
ax3.plot([d1, d2], [data_tanikon['PM10 [ug/m3]'].mean(), data_tanikon['PM10 [ug/m3]'].mean()], color='gray', linestyle=':',label='mean PM10')
ax3.fill_between(date_list, h, where=unsual_period, color='tomato', linestyle=':', alpha=0.1, label='unsual periods')
ax3.set_ylim([0,upprval])
ax3b.set_ylim([-15,15])
ax3.set_xlim([d1, d2])

ax3.set_ylabel('$\mathregular{PM10 \ [\mu g/m^3]}$')
ax3b.set_ylabel('Temperature [°C] / Precipitation [mm/h]')
ax3.set_title('PM10, Temperature and Precipitation in Tänikon over the unsual period', loc='left')

handles = ax3.get_legend_handles_labels()[0] + ax3b.get_legend_handles_labels()[0]
labels = ['PM10','Mean PM10','Unsual period','Temperature','Precipitation']
lgd = fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax3.transAxes, ncol=1)

fig.tight_layout()
fig.savefig('./Figures/UnsualPeriodPM10Tanikon.png', dpi=figure_res)
plt.show()

# %% check when NO2 overpass 80 ug/m3 on daily average and 110 ug/m3 on hourly measures
dates = data_sion.loc[data_sion['NO2 [ug/m3]'] > 110, 'Date']

cmap = cm.get_cmap('Oranges')
color_idx = [0.5+i/(2*len(dates)) for i in range(len(dates))]
fig, ax = plt.subplots(1,1,figsize=(8,6))
for date, color_idx in zip(dates, color_idx):
    d = date.day
    m = date.month
    ax.plot(range(24),data_sion.loc[(data_sion['Date']>=dt.datetime(2018,m,d,0)) & (data_sion['Date']<dt.datetime(2018,m,d+1,0)),'NO2 [ug/m3]'], lw=2, color=cmap(color_idx))
ax.plot([0,23],[100,100],linewidth=2, linestyle='--', color='black')
ax.set_xlim([0,23])
handles, _ = ax.get_legend_handles_labels()
labels = ['2018-{month:02d}-{day:02d}'.format(month=date.month, day=date.day) for date in dates]
lgd = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.3, 0.88), bbox_transform=fig.transFigure, ncol=1)
plt.show()

# <*!>

# <*!>
