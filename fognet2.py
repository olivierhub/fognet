#!/usr/bin/python

'''
Based on https://www.kaggle.com/justdoit/rossmann-store-sales/xgboost-in-python-with-rmspe/code
Public Score :  not submitted
Private Validation Score :  0.114779
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator
import matplotlib
#matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt
import seaborn as sns
import locale
#locale.setlocale(locale.LC_ALL,'en_US')
import h5py
import cPickle, gzip, os, glob, os.path
import cPickle as pickle
import copy

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import MiniBatchKMeans, KMeans     
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from operator import itemgetter
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor

import sys
from psycopg2._psycopg import Column
from numexpr.necompiler import evaluate
np.random.seed(0)

from fognet_utils import toBinary, rmse_scoring, report, WindParse, WindParse2, CloudCoverParse, CloudDensityParse, WeatherParse, TruncateTimeStamp2Hours, TruncateTimeStamp4Hours, TruncateTimeStampHours
  
from fognet_utils import GenerateGroupsBy, compute_rmse, plot_errors,  compare_evaluations, CloudHeigthParse, process_model, remove_correlated_features
from fognet_utils import add_eval_sets
import fognet_utils
    
## Start of main script

if 1:
    
    if 1:
        print("Loading sidi sdata")
        
        sidi = pd.read_csv("fognet_sidi.csv",index_col = 0, parse_dates=[0])
        
        
        
        sidi.loc[sidi['WW'] == 'Fog or ice fog, sky visible, has begun or has become thicker during the preceding hour.', 'RainLevel'] = 'Fog'
        sidi.loc[sidi['WW'] == 'Mist', 'RainLevel'] = 'Mist'
        sidi.loc[sidi['W1'] == 'Drizzle', 'RainLevel'] = 'Drizzle'
        sidi.loc[sidi['W1'] == 'Thunderstorm(s) with or without precipitation.', 'RainLevel'] = 'Thunder'
        sidi.loc[sidi['W1'] == 'Rain', 'RainLevel'] = 'Rain'
        sidi.loc[sidi['W1'] == 'Fog or ice fog or thick haze.', 'RainLevel'] = 'Fog'        
        sidi.loc[sidi['W1'] == 'Shower(s).', 'RainLevel' ] = 'Rain'
        sidi.loc[sidi['W2'] == 'Fog or ice fog or thick haze.', 'RainLevel' ] = 'Fog'
        sidi.loc[sidi['W2'] == 'Shower(s).', 'RainLevel' ] = 'Rain'
        sidi.loc[sidi['W2'] == 'Drizzle', 'RainLevel'] = 'Drizzle'
        sidi.loc[sidi['W2'] == 'Rain', 'RainLevel'] = 'Rain'
        sidi.loc[sidi['RainLevel'].isnull(), 'RainLevel'] = 'Dry'
        
        
        
        toBinary('RainLevel', sidi)
        sidi.drop(['RainLevel'], axis=1,inplace=True)
        
        
        
        # drop garbage columns
        sidi.drop(["WW","W1","W2","E'", "sss","E"],axis=1,inplace = True)
        
        # These one should not be useful
        sidi.drop(['Tn','Tx'],axis=1,inplace = True)
        
        # todo
        #sidi.drop(['Tg'],axis=1,inplace = True)
        
        # Fix cloud cover..
        
        sidi.loc[sidi['N'] == 'no clouds', 'Cl'] = 'no clouds'
        sidi.loc[sidi['N'] == 'no clouds', 'Nh'] = 'no clouds'
        sidi.loc[sidi['N'] == 'no clouds', 'H'] = '2500 or more, or no clouds'
        sidi.loc[sidi['N'] == 'no clouds', 'Cm'] = 'no clouds'
        sidi.loc[sidi['N'] == 'no clouds', 'Ch'] = 'no clouds'
        
        sidi.loc[sidi['Cl'] == 'No Stratocumulus, Stratus, Cumulus or Cumulonimbus.', 'Cl' ] = 'no clouds'
        sidi.loc[sidi['Cm'] == 'No Altocumulus, Altostratus or Nimbostratus.', 'Cm' ] = 'no clouds'
        sidi.loc[sidi['Ch'] == 'No Cirrus, Cirrocumulus or Cirrostratus.', 'Ch' ] = 'no clouds'
        
        
        sidi.loc[(sidi['H'] == '2500 or more, or no clouds.') & sidi['Cl'].isnull(), 'Cl' ] = 'no clouds'
        sidi.loc[(sidi['H'] == '2500 or more, or no clouds.') & sidi['Cm'].isnull(), 'Cm' ] = 'no clouds' 
        sidi.loc[(sidi['H'] == '2500 or more, or no clouds.') & sidi['Ch'].isnull(), 'Ch' ] = 'no clouds'
        
        # We don't see them.. They are not there..
        sidi.loc[(sidi['Nh'] == '100%') & sidi['Ch'].isnull() , 'Ch' ] = 'no visible clouds'
        sidi.loc[(sidi['Nh'] == '90  or more, but not 100%') & sidi['Ch'].isnull(), 'Ch' ] = 'no visible clouds'        
        sidi.loc[(sidi['N'] == '100%') & sidi['Ch'].isnull(), 'Ch' ] = 'no visible clouds'
        sidi.loc[(sidi['N'] == '90  or more, but not 100%') & sidi['Ch'].isnull(), 'Ch' ] = 'no visible clouds'
        
        sidi.loc[(sidi['Nh'] == '100%') & sidi['Cm'].isnull() , 'Cm' ] = 'no visible clouds'
        sidi.loc[(sidi['Nh'] == '90  or more, but not 100%') & sidi['Cm'].isnull(), 'Cm' ] = 'no visible clouds'        
        sidi.loc[(sidi['N'] == '100%') & sidi['Cm'].isnull(), 'Cm' ] = 'no visible clouds'
        
        
        # Fix lazy public servant's work
        sidi.loc[(sidi['N'] == 'no clouds') & sidi['Nh'].isnull(), 'Nh'] = 'no clouds'
        sidi.loc[(sidi['N'] == 'no clouds') & sidi['H'].isnull(), 'H'] = '2500 or more, or no clouds.'
        
        
        
        sidi.loc[sidi['Cl'] == 'Cumulus mediocris or congestus, with or without Cumulus of species fractus or humilis or Stratocumulus, all having their bases at the same level.', 'Cl' ] = 'mediocris'                       
        sidi.loc[sidi['Cl'] == 'Stratocumulus other than Stratocumulus cumulogenitus.', 'Cl'] = 'Stratocumulus'
        sidi.loc[sidi['Cl'] == 'Stratus fractus or Cumulus fractus of bad weather, or both (pannus), usually below Altostratus or Nimbostratus.', 'Cl' ] = 'fractus'
        sidi.loc[sidi['Cl'] == 'Cumulonimbus calvus, with or without Cumulus, Stratocumulus or Stratus.', 'Cl' ] = 'calvus' 
        sidi.loc[sidi['Cl'] == 'Cumulonimbus capillatus (often with an anvil), with or without Cumulonimbus calvus, Cumulus, Stratocumulus, Stratus or pannus.', 'Cl' ] = 'capillatus' 
        sidi.loc[sidi['Cl'] == 'Stratus nebulosus or Stratus fractus other than of bad weather, or both.', 'Cl' ] = 'nebulosus'
        sidi.loc[sidi['Cl'] == 'Cumulus humilis or Cumulus fractus other than of bad weather, or both.', 'Cl' ] = 'humilis'
        sidi.loc[sidi['Cl'] == 'Cumulus and Stratocumulus other than Stratocumulus cumulogenitus, with bases at different levels.', 'Cl' ] = 'cumulus' 
        
        
        sidi.loc[sidi['Cm'] == 'no visible clouds', 'Cm' ] = 'novisible'
        sidi.loc[sidi['Cm'] == 'Altocumulus translucidus at a single level.', 'Cm' ] = 'translucidus'
        sidi.loc[sidi['Cm'] == 'Altocumulus translucidus or opacus in two or more layers, or Altocumulus opacus in a single layer, not progressively invading the sky, or Altocumulus with Altostratus or Nimbostratus.', 'Cm' ] = 'opacus'
        sidi.loc[sidi['Cm'] == 'Altocumulus castellanus or floccus.', 'Cm' ] = 'floccus'
        
        sidi.loc[sidi['Ch'] == 'no visible clouds', 'Ch' ] = 'novisible'
        sidi.loc[sidi['Ch'] == 'Cirrus spissatus, in patches or entangled sheaves, which usually do not increase and sometimes seem to be the remains of the upper part of a Cumulonimbus; or Cirrus castellanus or floccus.', 'Ch' ] = 'spissatus'
        sidi.loc[sidi['Ch'] == 'Cirrus (often in bands) and Cirrostratus, or Cirrostratus alone, progressively invading the sky; they generally thicken as a whole, but the continuous veil does not reach 45 degrees above the horizon.', 'Ch' ] = 'cirrostratus'
        sidi.loc[sidi['Ch'] == 'Cirrus fibratus, sometimes uncinus, not progressively invading the sky.', 'Ch' ] = 'fibratus'
        
        #print(sidi.loc[sidi['Ch'].isnull(),['Cl','Cm','Ch','N','Nh']])
        #print(sidi['Ch'].unique())
        print(sidi.columns)            
        
        # Trace of precipitation is .. small precipitation
        sidi.loc[sidi['RRR'] == "Trace of precipitation", 'RRR' ] = 0.01
        sidi['RRR'] = sidi['RRR'].astype(np.float32)
                                         
        # At  6:00 AM there should be report of rain for the last 24h
        # But maybe there is no data if there was no rain for 24h
        
        # Make sure we have data at 12:00, 18:00, 6:00 with rain for the last 6,12,24 hours
        sidi.loc[sidi['tR'].isnull() & (sidi.index.hour == 6), 'tR' ] = 24
        sidi.loc[sidi['RRR'].isnull() & (sidi.index.hour == 6), 'RRR' ] = 0.0
        
        sidi.loc[sidi['tR'].isnull() & (sidi.index.hour == 18), 'tR' ] = 12
        sidi.loc[sidi['RRR'].isnull() & (sidi.index.hour == 18), 'RRR' ] = 0.0
        
        sidi.loc[sidi['tR'].isnull() & (sidi.index.hour == 12), 'tR' ] = 6
        sidi.loc[sidi['RRR'].isnull() & (sidi.index.hour == 12), 'RRR' ] = 0.0
        
        sidi.loc[sidi['RRR'].isnull(),'RRR'] = 0.0
        
        
        sidi['day'] = sidi.index.values.astype('<M8[D]')
                
        sidi['day_before'] = sidi['day'] - np.timedelta64(1,'D')
        
        # At 6h00 retrieve the volume the previous day at 18h so that
        # We know the qtity of rain during the night
        
        sidi_night_rain = pd.merge(sidi.loc[sidi.index.hour == 6, ['day_before','RRR']].reset_index(), sidi.loc[sidi.index.hour == 18, ['day','RRR']], left_on = 'day_before', right_on = 'day' ).set_index('Local time in Sidi Ifni')
        
        sidi_night_rain['avg_night_rain_3h'] = (sidi_night_rain['RRR_x'] - sidi_night_rain['RRR_y']) / 4.0
        
        # At 18h00 retrieve the volume at 12h so that
        # We know the qtity of rain between 12h and 18h
        
        sidi_18_rain = pd.merge(sidi.loc[sidi.index.hour == 18, ['day','RRR']].reset_index(), sidi.loc[sidi.index.hour == 12, ['day','RRR']], left_on = 'day', right_on = 'day' ).set_index('Local time in Sidi Ifni')
        
        sidi_18_rain['avg_18_rain_3h'] = (sidi_18_rain['RRR_x'] - sidi_18_rain['RRR_y']) / 2.0
        
        sidi = sidi.join(sidi_night_rain['avg_night_rain_3h'], how='left')
        sidi = sidi.join(sidi_18_rain['avg_18_rain_3h'], how='left')
        
        sidi.loc[sidi.index.hour == 6, 'avg_rain_3h'] = sidi.loc[sidi.index.hour == 6, 'avg_night_rain_3h'] 
        sidi.loc[sidi.index.hour == 18, 'avg_rain_3h'] = sidi.loc[sidi.index.hour == 18, 'avg_18_rain_3h']
        sidi.loc[sidi.index.hour == 12, 'avg_rain_3h'] = sidi.loc[sidi.index.hour == 12, 'RRR'] / 2.0
        
        sidi.drop(['avg_night_rain_3h','avg_18_rain_3h','RRR','tR','day','day_before'], axis=1, inplace=True)
        
        col_shift_list = ['T','Po','U','DD','Ff','Cl','Nh','Ch']
        
        print(sidi.isnull().sum(axis=0))
        for x_offset in xrange(3):
                    sidi = sidi.join(sidi.shift(periods=1, axis=0)[col_shift_list],rsuffix='_minus_1',how='left')
                    
                    
                    for col in col_shift_list:
                        sidi.loc[sidi[col].isnull(), col] = sidi['%s_minus_1' % col]
                    
                    sidi.loc[(sidi["DD"] == 'variable wind direction'), 'DD'] = sidi['%s_minus_1' % 'DD']                    
                    
                    for col in col_shift_list:
                        sidi.drop(['%s_minus_1'%col], axis=1, inplace=True)
        
        
        print(sidi.isnull().sum(axis=0))
        
        #print(sidi.loc[sidi['DD'].isnull(),'DD'])
                        
        #WindParse(sidi,"DD")                        
        WindParse2(sidi,"DD")
        CloudCoverParse(sidi,"N")
        CloudCoverParse(sidi,"Nh",'CloudCover2')
        CloudHeigthParse(sidi,'H')
        
        print(sidi.loc[sidi['CloudHeight'].isnull(),'H'])
        sidi_wind = sidi[np.logical_not(sidi['Ff'].isnull())][['Ff','WindDirection1','WindDirection2','WindDirection3']].copy()
        
        sidi.drop(['DD','N','Ff','Nh','H','WindDirection1','WindDirection2','WindDirection3'], axis=1, inplace = True)
        
        for col_cat in ['Cl','Cm','Ch']:
            newcols = toBinary(col_cat, sidi)
            sidi.drop([col_cat], axis=1, inplace = True)
        
        for col_cat in ['WindDirection1','WindDirection2','WindDirection3']:
            newcols = toBinary(col_cat, sidi_wind)
            sidi_wind.drop([col_cat], axis=1, inplace = True)
            
        sidi_pa = sidi[np.logical_not(sidi['Pa'].isnull() )][['Pa']].copy()
        
        sidi_Tg = sidi[np.logical_not(sidi['Tg'].isnull())][['Tg']].copy()
        
        sidi_avg_rain = sidi[np.logical_not(sidi['avg_rain_3h'].isnull())][['avg_rain_3h']].copy()
        
        sidi.drop(['Pa','Tg','avg_rain_3h'], axis=1, inplace = True)
        print(sidi.isnull().sum(axis=0))
        #sys.exit(0)                
        sidi.dropna(inplace=True)
        sidi_wind.dropna(inplace=True)
        
        print(sidi_wind.isnull().sum(axis=0))        
        print(sidi.isnull().sum(axis=0))
        
        
        print("Describe sidi",sidi.describe())
        print("Describe sidi_wind",sidi_wind.describe())
        print("Describe sidi_pa",sidi_pa.describe())
        print("Describe sidi_Tg",sidi_Tg.describe())
        print("Describe sidi_avg_rain",sidi_avg_rain.describe())
        
        

    if 1:
        print("Loading guelmin sdata")
        
        guelmin = pd.read_csv("fognet_macro_guelmin.csv",index_col = 0, parse_dates=[0])
        
        
        
        # drop garbage columns
        guelmin.drop(['ff10'], axis=1, inplace = True)
        
        guelmin.loc[guelmin['WW'].isnull(), 'WW'] = guelmin["W'W'"]
        
        WeatherParse(guelmin,"WW")
        
        print(guelmin.loc[guelmin['RainLevel'].isnull(),["WW"]])
        guelmin.loc[guelmin["c"] == "29970 m9, few clouds (10-30%) 450 m", "c"] = "Few clouds (10-30%) 450 m"
        guelmin.loc[guelmin["c"] == "29970 m9, few clouds (10-30%) 480 m", "c"] = "Few clouds (10-30%) 480 m"
        guelmin.loc[guelmin["c"] == "29970 m9, few clouds (10-30%) 600 m", "c"] = "Few clouds (10-30%) 600 m"
        guelmin.loc[guelmin["c"] == "Scattered clouds (40-50%)30, cumulonimbus clouds , broken clouds (60-90%) 3000 m","c" ] = "Scattered clouds (40-50%) 30 m, cumulonimbus clouds , broken clouds (60-90%) 3000 m"
        guelmin.loc[guelmin["c"] == "8 less than 30 m , few clouds (10-30%) 480 m , 8 less than 984 feet , few clouds (10-30%) 480 m","c" ] = "Few clouds (10-30%) 480 m"
        
        guelmin.loc[guelmin["c"] == "Scattered clouds (40-50%) 000 000 18420 m/13", "c" ] = np.nan
        guelmin.loc[guelmin["c"] == "Altocumulus clouds T 690 m, cumulus congestus of great vertical extent , few clouds (10-30%) 780 m, cumulonimbus clouds , scattered clouds (40-50%) 3000 m", "c" ] = "Few clouds (10-30%) 780 m"
        
        
        guelmin.drop(["WW","W'W'"],axis=1,inplace = True)
        # append value 1 hour before for DD and ff
        guelmin.loc[(guelmin["DD"] == 'variable wind direction') & (guelmin["Ff"] <= 1) , 'DD'] = 'Calm, no wind'
        
        guelmin.loc[guelmin['VV'] == '10.0 and more', 'VV'] = 10
        
        guelmin['VV'] = guelmin['VV'].astype(np.float32) 
        
        col_shift_list = ['DD','Ff','T','P0','P','VV','Td','U','c','RainLevel']
        
        print(guelmin.isnull().sum(axis=0))
        for x_offset in xrange(3):
                    guelmin = guelmin.join(guelmin.shift(periods=1, axis=0)[col_shift_list],rsuffix='_minus_1',how='left')
                    
                    
                    for col in col_shift_list:
                        guelmin.loc[guelmin[col].isnull(), col] = guelmin['%s_minus_1' % col]
                    
                    guelmin.loc[(guelmin["DD"] == 'variable wind direction'), 'DD'] = guelmin['%s_minus_1' % 'DD']
                    guelmin.loc[(guelmin["c"] == 'Overcast (100%)'), 'c'] = guelmin['%s_minus_1' % 'c']
                    
                    for col in col_shift_list:
                        guelmin.drop(['%s_minus_1'%col], axis=1, inplace=True)
                
        
        print(guelmin.isnull().sum(axis=0))
        #WindParse(guelmin,"DD")                        
        WindParse2(guelmin,"DD")
        
        
        guelmin['cloud_array'] = guelmin.apply(lambda x: str(x['c']).split(",",1)[0].replace("Scattered ","Scattered_").replace("Few ","Few_").replace("Broken ","Broken_").replace("No Significant Clouds","No_Clouds_(0-0%) 10000 m").replace("No clouds","No_Clouds_(0-0%) 19999 m").replace("less than 30","15").replace(" (","_("), axis=1)
        guelmin['cloud_density'] = guelmin.apply(lambda x: x['cloud_array'].split(" ")[0], axis=1)
        
        CloudDensityParse(guelmin,'cloud_density')
        
        guelmin['cloud_distance'] = guelmin.apply(lambda x: x['cloud_array'].split(" ")[1], axis=1)
        
        guelmin['cloud_distance'] = guelmin['cloud_distance'].astype(np.float32)
        
        #CloudCoverParse(guelmin,"N")
        
        #print guelmin.loc[guelmin['WindDirection'].isnull(),'DD']
        print(guelmin.columns)            
        
        print(guelmin.isnull().sum(axis=0))
        
        
        guelmin_wind = guelmin[np.logical_not(guelmin['Ff'].isnull())][['Ff','WindDirection1','WindDirection2','WindDirection3']].copy()
        
        for col_cat in ['WindDirection1','WindDirection2','WindDirection3']:
            newcols = toBinary(col_cat, guelmin_wind)
            guelmin_wind.drop([col_cat], axis=1, inplace = True)
            
        guelmin.drop(['DD','Ff','WindDirection1','WindDirection2','WindDirection3','c','cloud_density','cloud_array'], axis=1, inplace = True)        
        guelmin.dropna(inplace=True)
        guelmin_wind.dropna(inplace=True)
        
        print(guelmin.isnull().sum(axis=0))
        print("Describe guelmin", guelmin.describe())
        print(guelmin_wind.isnull().sum(axis=0))
        print("Describe guelmin_wind", guelmin_wind.describe())
        #sys.exit(0)
        
    if 1:
        print("Loading agadir data")
        
        agadir = pd.read_csv("fognet_agadir.csv",index_col = 0, parse_dates=[0])
        
        # drop garbage columns
        agadir.drop(["ff10"],axis=1,inplace = True)

        agadir.loc[agadir['WW'].isnull(), 'WW'] = agadir["W'W'"]        
        WeatherParse(agadir,"WW")

        agadir.drop(['WW',"W'W'"],axis=1,inplace = True)

        agadir.loc[agadir["c"] == "29970 m9, few clouds (10-30%) 480 m", "c"] = "Few clouds (10-30%) 480 m"
        agadir.loc[agadir["c"] == "29970 m9, few clouds (10-30%) 600 m", "c"] = "Few clouds (10-30%) 600 m"
        agadir.loc[agadir["c"] == "Scattered clouds (40-50%)30, cumulonimbus clouds , broken clouds (60-90%) 3000 m","c" ] = "Scattered clouds (40-50%) 30 m, cumulonimbus clouds , broken clouds (60-90%) 3000 m"
        
        agadir.loc[agadir["c"] == "Scattered clouds (40-50%) 000 000 18420 m/13", "c" ] = np.nan


        # fix the crazy "variable wind direction.."        
        agadir.loc[(agadir["DD"] == 'variable wind direction') & (agadir["Ff"] <= 1) , 'DD'] = 'Calm, no wind'
        
        #agadir = agadir.applymap(lambda x: np.nan if isinstance(x, basestring) and x == "variable wind direction" else x)
        #agadir.replace("variable wind direction",np.nan, inplace=True)

        
        print(agadir.isnull().sum(axis=0))


        
        col_shift_list = ["T","P0","P","U","DD","Ff","c","VV","Td"]
        
        for x_offset in xrange(17):
            agadir = agadir.join(agadir.shift(periods=1, axis=0)[col_shift_list],rsuffix='_minus_1',how='left')
            
            
            for col in col_shift_list:
                agadir.loc[agadir[col].isnull(), col] = agadir['%s_minus_1' % col]
    
            agadir.loc[(agadir["DD"] == 'variable wind direction'), 'DD'] = agadir['%s_minus_1' % 'DD']
            agadir.loc[(agadir["c"] == 'Overcast (100%)'), 'c'] = agadir['%s_minus_1' % 'c']
            
            for col in col_shift_list:
                agadir.drop(['%s_minus_1'%col], axis=1, inplace=True)


        agadir.loc[agadir['VV'] == '10.0 and more', 'VV'] = 10
        
        agadir['VV'] = agadir['VV'].astype(np.float32) 

        agadir['cloud_array'] = agadir.apply(lambda x: str(x['c']).split(",",1)[0].replace("Scattered ","Scattered_").replace("Few ","Few_").replace("Broken ","Broken_").replace("No Significant Clouds","No_Clouds_(0-0%) 10000 m").replace("No clouds","No_Clouds_(0-0%) 19999 m").replace("less than 30","15").replace(" (","_("), axis=1)
        agadir['cloud_density'] = agadir.apply(lambda x: x['cloud_array'].split(" ")[0], axis=1)
        
        CloudDensityParse(agadir,'cloud_density')
        
        agadir['cloud_distance'] = agadir.apply(lambda x: x['cloud_array'].split(" ")[1], axis=1)
        
        agadir['cloud_distance'] = agadir['cloud_distance'].astype(np.float32)
        

        agadir.drop(['c','cloud_array','cloud_density'],axis=1,inplace = True)


        #WindParse(agadir,"DD")
        WindParse2(agadir,"DD")
        
        for col_cat in ['WindDirection1','WindDirection2','WindDirection3']:
            newcols = toBinary(col_cat, agadir)
            agadir.drop([col_cat], axis=1, inplace = True)

        agadir.drop(['DD'], axis=1, inplace = True)

        print(agadir.isnull().sum(axis=0))
        agadir.dropna(inplace=True)
        print(agadir.isnull().sum(axis=0))
        print(agadir.count(axis=0))
        print(agadir.describe())
        
        
        if 0:
            fig, axs = plt.subplots(nrows=agadir.shape[1], ncols=1, sharex=True) #, figsize=(16, 18))
    
            columns = agadir.columns
            for i, ax in list(enumerate(axs)):
                
                col = columns[i]
                print("Graph column", col)
                ax.plot_date(agadir.index, agadir[col], ms=1.5, label='train')
                ax.set_ylabel(col)
                
                if i == 0:
                    ax.legend(loc='upper right', markerscale=10, fontsize='xx-large')
            
            plt.show()
            
        

    train_micro_5mn = pd.read_csv("fognet_train_micro_5mn.csv", index_col = 0, parse_dates=[0])            
    train_micro_2h = pd.read_csv("fognet_train_micro_2h.csv", index_col = 0, parse_dates=[0])
    test_micro_5mn = pd.read_csv("fognet_test_micro_5mn.csv", index_col = 0, parse_dates=[0])            
    test_micro_2h = pd.read_csv("fognet_test_micro_2h.csv", index_col = 0, parse_dates=[0])
    
    train_target = pd.read_csv("fognet_target.csv", index_col = 0, parse_dates=[0])
    train_target['is_train'] = True
    
    test_submit = pd.read_csv("fognet_submit.csv", index_col = 0, parse_dates=[0])
    test_submit['is_train'] = False
    
    all_micro_5mn = pd.concat([train_micro_5mn, test_micro_5mn]).sort_index()

    
    
    if 0:
        fig, axs = plt.subplots(nrows=all_micro_5mn.shape[1], ncols=1, sharex=True) #, figsize=(16, 18))

        columns = all_micro_5mn.columns
        for i, ax in list(enumerate(axs)):
            
            col = columns[i]
            print("Graph column", col)
            ax.plot_date(all_micro_5mn.index, all_micro_5mn[col], ms=1.5, label='train')
            ax.set_ylabel(col)
            
            if i == 0:
                ax.legend(loc='upper right', markerscale=10, fontsize='xx-large')
        
        plt.show()

    all_micro_2h = pd.concat([train_micro_2h, test_micro_2h]).sort_index()
    print(all_micro_2h.isnull().sum(axis=0))
    
    
    all_micro_2h_460 = all_micro_2h[np.logical_not(all_micro_2h['leafwet460_min'].isnull())][['leafwet460_min']].copy()
    all_micro_2h.drop(['leafwet460_min'],axis=1,inplace=True)
    
    col_shift_list = ["percip_mm","humidity","temp","leafwet450_min","leafwet_lwscnt","gusts_ms","wind_dir","wind_ms"]
    
    for x_offset in xrange(21):
        all_micro_2h = all_micro_2h.join(all_micro_2h.shift(periods=1, axis=0)[col_shift_list],rsuffix='_minus_1',how='left')
                
        for col in col_shift_list:
            all_micro_2h.loc[all_micro_2h[col].isnull(), col] = all_micro_2h['%s_minus_1' % col]
        
        for col in col_shift_list:
            all_micro_2h.drop(['%s_minus_1'%col], axis=1, inplace=True)
    
    
    print(all_micro_2h.isnull().sum(axis=0))
    
    if 0:
        fig, axs = plt.subplots(nrows=(all_micro_2h.shape[1] + 1), ncols=1, sharex=True) #, figsize=(16, 18))

        columns = all_micro_2h.columns
        for i, ax in list(enumerate(axs)):
            
            if i < all_micro_2h.shape[1]:
                col = columns[i]
                print("Graph column", col)
                ax.plot_date(all_micro_2h.index, all_micro_2h[col], ms=1.5, label='train')                        
                ax.set_ylabel(col)
            else:
                ax.plot_date(train_target.index, train_target['yield'], ms=1.5, label='target')
            
            if i == 0:
                ax.legend(loc='upper right', markerscale=10, fontsize='xx-large')
        
        
        plt.show()


    # Compte diffs
    
    full_period_1h=pd.date_range('2013-11-23 16:00:00',periods=18524, freq='1H')
    pd_full_period_1h = pd.DataFrame(index=full_period_1h)
    agadir_full_p_1h = pd_full_period_1h.join(agadir[['T','P','U']])
    
    col_shift_list = ["T","P","U"]
    
    for x_offset in xrange(2):
        agadir_full_p_1h = agadir_full_p_1h.join(agadir_full_p_1h.shift(periods=1, axis=0)[col_shift_list],rsuffix='_minus_1',how='left')
        
        for col in col_shift_list:
            agadir_full_p_1h.loc[agadir_full_p_1h[col].isnull(), col] = agadir_full_p_1h['%s_minus_1' % col]
        
        for col in col_shift_list:
            agadir_full_p_1h.drop(['%s_minus_1'%col], axis=1, inplace=True)
    
    full_period=pd.date_range('2013-11-23 16:00:00',periods=9262, freq='2H')
    pd_full_period = pd.DataFrame(index=full_period)
    agadir_full_p = pd_full_period.join(agadir_full_p_1h)
    agadir_offset = agadir_full_p[['T','P','U']].join(agadir_full_p[['T','P','U']].shift(periods=1, axis=0), rsuffix='_minus_1')
    agadir_offset['P_diff'] = agadir_offset['P'] - agadir_offset['P_minus_1']
    agadir_offset['T_diff'] = agadir_offset['T'] - agadir_offset['T_minus_1']
    agadir_offset['U_diff'] = agadir_offset['U'] - agadir_offset['U_minus_1']
    agadir_offset.drop(['T','P','U','T_minus_1','P_minus_1','U_minus_1'],axis=1,inplace = True)
    agadir_offset.dropna(inplace=True)



    all_micro_full_p = pd.concat([train_micro_2h, test_micro_2h]).sort_index()
    all_micro_full_p = pd_full_period.join(all_micro_full_p[['temp','leafwet_lwscnt']])
    
    all_micro_offset = all_micro_full_p[['temp','leafwet_lwscnt']].join(all_micro_full_p[['temp','leafwet_lwscnt']].shift(periods=1, axis=0), rsuffix='_minus_1')
    all_micro_offset['temp_diff'] = all_micro_offset['temp'] - all_micro_offset['temp_minus_1']
    all_micro_offset['leaf_diff'] = all_micro_offset['leafwet_lwscnt'] - all_micro_offset['leafwet_lwscnt_minus_1']
    all_micro_offset.drop(['temp','leafwet_lwscnt','temp_minus_1','leafwet_lwscnt_minus_1'], axis=1,inplace=True)
    all_micro_offset.dropna(inplace=True)



    date_model = pd.concat([train_target,test_submit]).sort_index()
    if 0:            
        date_model['woy'] = date_model.index.weekofyear
        date_model['m'] = date_model.index.month
        date_model['h'] = date_model.index.hour
    sun_model = fognet_utils.sun_at_fognets()
    
    date_model = date_model.join(sun_model)
    
    
    if 1:
        # Model based only on date
        print("Linear model on date")
        lin_model = date_model.copy()
                        
        
        lr = LinearRegression(fit_intercept=True,normalize=True,copy_X=True,n_jobs=6)
    
        print("fit")
        #lr.fit(lin_model.loc[date_model['is_train'] == True, ['m','woy']].as_matrix().astype(np.float32), lin_model.loc[date_model['is_train'] == True, 'yield'].as_matrix().astype(np.float32))
        lr.fit(lin_model.loc[date_model['is_train'] == True, ['sun_angle','hours_of_sun','hours_of_night','adjusted_hours_of_sun','is_day']].as_matrix().astype(np.float32), lin_model.loc[date_model['is_train'] == True, 'yield'].as_matrix().astype(np.float32))

        print("predict")
        #predicted_yield = lr.predict(lin_model.loc[date_model['is_train'] == False, ['m','woy']].as_matrix().astype(np.float32))
        predicted_yield = lr.predict(lin_model.loc[date_model['is_train'] == False, ['sun_angle','hours_of_sun','hours_of_night','adjusted_hours_of_sun','is_day']].as_matrix().astype(np.float32))
        
        lin_model.loc[date_model['is_train'] == False, 'yield_lin'] = predicted_yield
        lin_model.to_hdf('fognet_datemodel.hdf','datemodel',mode='w',complib='blosc')
        
        #print(lin_model)     

    if 1:
        # Model based only on agadir
        print("Agadir model")

        agadir_model = date_model.copy()
        print(agadir_model.isnull().sum(axis=0))
        print(agadir_model.count().sum(axis=0))

        if 1:
            group_list = GenerateGroupsBy(agadir, "agadir", hours_grouped = 4)
            for grouped_result in group_list:
                agadir_model = agadir_model.join(grouped_result)
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))

            if 0:
                group_list = GenerateGroupsBy(guelmin, "guelmin_24", hours_grouped = 24)            
                for grouped_result in group_list:
                    agadir_model = agadir_model.join(grouped_result)
                    print(agadir_model.isnull().sum(axis=0))
                    print(agadir_model.count(axis=0))
                
            if 0:
                group_list = GenerateGroupsBy(sidi, "sidi_24", hours_grouped = 24)
                for grouped_result in group_list:
                    agadir_model = agadir_model.join(grouped_result, how='left')
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))                                                

            if 0:
                group_list = GenerateGroupsBy(guelmin, "guelmin_12", hours_grouped = 12)            
                for grouped_result in group_list:
                    agadir_model = agadir_model.join(grouped_result)
                    print(agadir_model.isnull().sum(axis=0))
                    print(agadir_model.count(axis=0))
                
            if 0:
                group_list = GenerateGroupsBy(sidi, "sidi_12", hours_grouped = 12)
                for grouped_result in group_list:
                    agadir_model = agadir_model.join(grouped_result, how='left')
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))

            if 0:
                group_list = GenerateGroupsBy(sidi_wind, "sidi_wind_24", hours_grouped = 24)
                for grouped_result in group_list:
                    agadir_model = agadir_model.join(grouped_result, how='left')
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))

            if 0:
                group_list = GenerateGroupsBy(sidi_pa, "sidi_pa_24", hours_grouped = 24)
                for grouped_result in group_list:
                    agadir_model = agadir_model.join(grouped_result, how='left')
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))

            if 0:
                group_list = GenerateGroupsBy(sidi_Tg, "sidi_tg_24", hours_grouped = 24)
                for grouped_result in group_list:
                    agadir_model = agadir_model.join(grouped_result, how='left')
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))

            if 0:
                group_list = GenerateGroupsBy(sidi_avg_rain, "sidi_avg_24", hours_grouped = 24)
                for grouped_result in group_list:
                    agadir_model = agadir_model.join(grouped_result, how='left')
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))
                                           

        print(agadir_model.isnull().sum(axis=0))
        agadir_model.dropna(inplace=True)
        #print(agadir_model.isnull().sum(axis=0))
        print(agadir_model.count(axis=0))
        print(agadir_model.describe())
        #sys.exit(0)

        
        if 0:
            for c in agadir_model.columns:
                print("Processing column",c)
                if c == "yield":
                    continue
                if c == "is_train":
                    continue
                    
                nb_nulls =  np.sum(agadir_model[c].isnull())
                print ("There are", nb_nulls, "nulls in this column")
                if nb_nulls > 0:
                    agadir_model["%s_nulls" % c] = agadir_model[c].isnull()
                    agadir_model[c].fillna(-2, inplace=True)


        columns_selection = list(agadir_model.columns)
        #for c in columns_selection:
        #    print(c)
        #sys.exit(0)
        columns_selection.remove('yield')
        columns_selection.remove('is_train')
        #columns_selection.remove('h')
        #columns_selection.remove('woy')
        #for c in columns_selection:
        #    print(c)
        #sys.exit(0)

        columns_selection = remove_correlated_features(agadir_model, columns_selection)
        
        if 0:
            f = gzip.open("garbage_columns_agadir.gz","r")
            garbage_columns  = pickle.load(f)
            f.close()

            for col_results in garbage_columns: # 3.88
                if (1-col_results[1][1]/col_results[1][0]) > 0:
                    columns_selection.remove(col_results[0])
        
        
        if 0:
            print("Correlation plot")

            columns_selection_nocorr = remove_correlated_features(agadir_model, columns_selection)
                                                                  
            
            f, ax = plt.subplots(figsize=(10, 10))
            cmap = sns.blend_palette(["#00008B", "#6A5ACD", "#F0F8FF",
                          "#FFE6F8", "#C71585", "#8B0000"], as_cmap=True)
            
            sns.corrplot(agadir_model[columns_selection_nocorr ], annot=False, diag_names=False, cmap=cmap)
            ax.grid(False);
            
            plt.show()
            sys.exit(0)


        cv_list = fognet_utils.compute_cv_ranges(lin_model)
        
        if 1:
            work_model = agadir_model.copy()

            sub_form = lin_model[lin_model['is_train'] == True].copy()
            sub_form = sub_form[ (sub_form.index.day <= 4) & (sub_form.index.day >= 1)]
            
            evaluations_list = []
            if 0:
                regressor = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.005, n_estimators=10000,objective='reg:linear', subsample=0.65, colsample_bytree=0.8, seed=0, reg_lambda=0.97 , reg_alpha=0.2, gamma=1.0, missing = np.NaN )
                #regressor.fit(train_set_x_tot, train_set_y_tot ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
                #pred_agadir_valid = regressor.predict(valid_set_x)
                #f = gzip.open("pred_agadir_valid.gz","wb")
                #cPickle.dump( (pred_agadir_valid, valid_set_y, agadir_model_train, valid_indices) , f,cPickle.HIGHEST_PROTOCOL)
                (pred_model_agadir_valid, valid_set_agadir_y, agadir_model_train, agadir_valid_indices, agadir_pred , best_params) = process_model(agadir_model, columns_selection, regressor, need_eval_set = True, fit_params={ 'eval_metric':'rmse','early_stopping_rounds':50})
                
                print("rmse=", compute_rmse(valid_set_agadir_y, pred_model_agadir_valid))
            
            
                set_name = "agadirvanilla"
                agadir_model_train.loc[agadir_valid_indices,'yield_%s_predicted' % set_name ] = pred_model_agadir_valid
                evaluations_list.append((pred_model_agadir_valid, valid_set_agadir_y, agadir_model_train, agadir_valid_indices, set_name, agadir_model, agadir_pred, best_params))
            #sub_form = sub_form.join(agadir_model_train.loc[agadir_valid_indices,'yield_%s_predicted' % set_name],how='left')
            

            # sidi_wind_24 gagne contre tout le monde sauf guelmin_6
            # guelmin_6 gagne contre tout le monde sauf guelminwwind_6
            # guelminwwind_6 gagne contre toute le monde sauf sidi_wind_24
            # 759: sidi_wind_24 : 3.65
            # 571: sidi_wind_24 : 3.18
            # 562: sidi_wind_24 : 3.21
            # 558: guelmin_6 : 3.1099  ( meilleur que sidi_wind_24 )
            # 549: guelmin_6 : 3.135   
            # 515: sidi_wind_24 : 3.518
            # 478: sidi_wind_24 
            
            # micro2 meilleur plus general
            # micro2 battu par micro2_guelminwwind_2, micro2_sidiwind_4, micro4_sidiwindguelmin_4, micro_sidiguelmin_2,micro2_sidiwwind_2,micro2_sidiwind_2
            # micro2_sidiwwindguelminwwind_2 bat tout le monde
            # micro_sidiguelmin_2 bat tout le monde sauf micro2_sidiwwindguelminwwind_2
            # sidi_6_wind24 zone "571'
            #
            """
            'micro2_sidiwwindguelminwwind_2'
            'micro_sidiguelmin_2'
            'micro2_sidiwwind_2'
            'micro2_sidiwindguelmin_2'
            'micro2_guelminwwind_2'
            'micro2_sidiwind_2'
            'micro4_guelminwwind_4'
            'micro4_sidiwindguelmin_4'
            'micro2_sidiwind_4'
            'micro_2'
            'micro_4'
            'guelminwwind_2_sidiwwind2'
            'guelmin_6_wind24_sidiwpa6'
            'guelminsidiwwind_24'
            'guelminwwindsidiwwind_24'
            'guelminwwind_4_sidiwwind2'
            'guelmin_6_wind24'
            'guelminwwind_6_sidiwwind6'
            'guelmin_6'
            'guelmin_6_sidiwpa6'
            'guelmin_6_sidiwwind6'
            'guelmin_4_sidi4'
            'sidi_wwind_4'
            'sidi_wpa6_wind24'
            'sidi_6_wind24'
            'sidi_wwind_2'
            'guelmin_24'
            'guelminsidi_24'
            'guelminwwind_2'
            'guelminwwind_4'
            'guelmin_4'
            'guelmin_2'
            'guelmin_2_sidi2'
            'sidi_24'
            'sidi_2'
            'sidi_4'
            'sidi_pawindavgtg_4'
            'sidi_wind_246'
            'guelminw_4'
            'guelminw_2'
            'sidi_wind_24'
            'sidi_wind_6'
            'sidi_wind_4'
            'sidi_wind_2'
            'agadirvanilla']
            """
            if 1:
                if 0:
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_pa, 24, "sidi_pa_24") ], set_name = "sidi_pa_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_Tg, 24, "sidi_tg_24") ], set_name = "sidi_tg_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_avg_rain, 24, "sidi_avg_24") ], set_name = "sidi_avg_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_pa, 24, "sidi_pa_24"), (sidi_wind, 24, "sidi_wind_24")   ], set_name = "sidi_pawind_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_pa, 24, "sidi_pa_24"), (sidi_wind, 24, "sidi_wind_24"), (sidi_avg_rain, 24, "sidi_avg_24")   ], set_name = "sidi_pawindavg_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_pa, 24, "sidi_pa_24"), (sidi_wind, 24, "sidi_wind_24"), (sidi_avg_rain, 24, "sidi_avg_24"), (sidi_Tg, 24, "sidi_tg_24")   ], set_name = "sidi_pawindavgtg_24", evaluations_list =  evaluations_list, cv_list = cv_list )
    
                    f = gzip.open("fognet_compare_test_sidi24.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         
                else:
                    # best 759 results
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24") ], set_name = "sidi_wind_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24") ], set_name = "sidi_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 24, "guelmin_24") ], set_name = "guelmin_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24"),(guelmin, 24, "guelmin_24") ], set_name = "guelminsidi_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24"),(guelmin, 24, "guelmin_24"),(sidi_wind, 24, "sidi_wind_24") ], set_name = "guelminsidiwwind_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24"),(guelmin, 24, "guelmin_24"),(sidi_wind, 24, "sidi_wind_24"),(guelmin_wind, 24, "guelmin_wind_24") ], set_name = "guelminwwindsidiwwind_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 24, "guelmin_24"),(sidi, 24, "sidi_24") ,(sidi_wind, 24, "sidi_wind_24"),(guelmin_wind, 24, "guelmin_wind_24") , (sidi_pa, 24, "sidi_pa_24") ], set_name = "guelminwwindsidiwwindpa_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 24, "guelmin_24"),(sidi, 24, "sidi_24") ,(sidi_wind, 24, "sidi_wind_24"),(guelmin_wind, 24, "guelmin_wind_24") , (sidi_pa, 24, "sidi_pa_24"), (sidi_Tg, 24, "sidi_tg_24") ], set_name = "guelminwwindsidiwwindpatg_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 12, "guelmin_12"),(sidi, 12, "sidi_12") ,(sidi_wind, 12, "sidi_wind_12"),(guelmin_wind, 12, "guelmin_wind_12") , (sidi_pa, 12, "sidi_pa_12"), (sidi_Tg, 24, "sidi_tg_24") ], set_name = "guelminwwindsidiwwindpa12_tg24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 12, "guelmin_12"),(sidi, 12, "sidi_12") ,(sidi_wind, 12, "sidi_wind_12"),(guelmin_wind, 12, "guelmin_wind_12")  ], set_name = "guelminwwindsidiwwind12", evaluations_list =  evaluations_list, cv_list = cv_list )

                    f = gzip.open("fognet_compare_test_24d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         
                

                if 0:
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 6, "sidi_6") ], set_name = "sidi_6", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 6, "sidi_wind_6") ], set_name = "sidi_wind_6", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_pa, 6, "sidi_pa_6") ], set_name = "sidi_pa_6", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_avg_rain, 6, "sidi_avg_6") ], set_name = "sidi_avg_6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_Tg, 6, "sidi_tg_6") ], set_name = "sidi_tg_6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 6, "sidi_6"), (sidi_wind, 6, "sidi_wind_6") ], set_name = "sidi_wwind_6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 6, "sidi_6"), (sidi_pa, 6, "sidi_pa_6") ], set_name = "sidi_wpa_6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 6, "sidi_6"), (sidi_avg_rain, 6, "sidi_avg_6") ], set_name = "sidi_wavg_6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 6, "sidi_6"), (sidi_Tg, 6, "sidi_tg_6") ], set_name = "sidi_wtg_6", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 6, "sidi_6"),(sidi_pa, 6, "sidi_pa_6"),(sidi_wind, 6, "sidi_wind_6"), (sidi_avg_rain, 6, "sidi_avg_6"), (sidi_Tg, 6, "sidi_tg_6")   ], set_name = "sidi_pawindavgtg_6", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    f = gzip.open("fognet_compare_test_sidi6.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         
                else:                
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 6, "sidi_wind_6") ], set_name = "sidi_wind_6", evaluations_list =  evaluations_list, cv_list = cv_list )                                
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24"),(sidi_wind, 6, "sidi_wind_6") ], set_name = "sidi_wind_246", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24"),(sidi, 6, "sidi_6") ], set_name = "sidi_6_wind24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24"),(sidi, 6, "sidi_6") ,(sidi_pa, 6, "sidi_pa_6")], set_name = "sidi_wpa6_wind24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    f = gzip.open("fognet_compare_test_sidi62d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         
    
                
                if 0:
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6") ], set_name = "guelmin_6", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin_wind, 6, "guelminw_6") ], set_name = "guelminw_6", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"), (guelmin_wind, 6, "guelminw_6") ], set_name = "guelminwwind_6", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"), (sidi, 6, "sidi_6") ], set_name = "guelminsidi_6", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"), (sidi, 6, "sidi_6"), (sidi_wind, 6, "sidi_wind_6"), (guelmin_wind, 6, "guelminw_6") ], set_name = "guelminsidiwwind_6", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    f = gzip.open("fognet_compare_test_guelmin6.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         
                else:
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6") ], set_name = "guelmin_6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24"),(guelmin, 6, "guelmin_6") ], set_name = "guelmin_6_wind24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"),(sidi, 6, "sidi_6") ,(sidi_pa, 6, "sidi_pa_6") ], set_name = "guelmin_6_sidiwpa6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24"),(guelmin, 6, "guelmin_6"),(sidi, 6, "sidi_6") ,(sidi_pa, 6, "sidi_pa_6") ], set_name = "guelmin_6_wind24_sidiwpa6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"),(sidi, 6, "sidi_6") ,(sidi_wind, 6, "sidi_wind_6") ], set_name = "guelmin_6_sidiwwind6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"),(sidi, 6, "sidi_6") ,(sidi_wind, 6, "sidi_wind_6"),(guelmin_wind, 6, "guelmin_wind_6") ], set_name = "guelminwwind_6_sidiwwind6", evaluations_list =  evaluations_list, cv_list = cv_list )
    
                    f = gzip.open("fognet_compare_test_guelmin62d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         
    
    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2") ], set_name = "sidi_2", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 2, "sidi_wind_2") ], set_name = "sidi_wind_2", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi_pa, 2, "sidi_pa_2") ], set_name = "sidi_pa_2", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi_avg_rain, 2, "sidi_avg_2") ], set_name = "sidi_avg_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi_Tg, 2, "sidi_tg_2") ], set_name = "sidi_tg_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidi_wind_2") ], set_name = "sidi_wwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2"), (sidi_pa, 2, "sidi_pa_2") ], set_name = "sidi_wpa_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2"), (sidi_avg_rain, 2, "sidi_avg_2") ], set_name = "sidi_wavg_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2"), (sidi_Tg, 2, "sidi_tg_2") ], set_name = "sidi_wtg_2", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2"),(sidi_pa, 2, "sidi_pa_2"),(sidi_wind, 2, "sidi_wind_2"), (sidi_avg_rain, 2, "sidi_avg_2"), (sidi_Tg, 2, "sidi_tg_2")   ], set_name = "sidi_pawindavgtg_2", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    f = gzip.open("fognet_compare_test_sidi2d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         

    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 4, "sidi_4") ], set_name = "sidi_4", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 4, "sidi_wind_4") ], set_name = "sidi_wind_4", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi_pa, 2, "sidi_pa_2") ], set_name = "sidi_pa_2", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi_avg_rain, 2, "sidi_avg_2") ], set_name = "sidi_avg_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi_Tg, 2, "sidi_tg_2") ], set_name = "sidi_tg_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 4, "sidi_4"), (sidi_wind, 4, "sidi_wind_4") ], set_name = "sidi_wwind_4", evaluations_list =  evaluations_list, cv_list = cv_list )
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2"), (sidi_pa, 2, "sidi_pa_2") ], set_name = "sidi_wpa_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2"), (sidi_avg_rain, 2, "sidi_avg_2") ], set_name = "sidi_wavg_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    #add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2"), (sidi_Tg, 2, "sidi_tg_2") ], set_name = "sidi_wtg_2", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 4, "sidi_4"),(sidi_pa, 4, "sidi_pa_4"),(sidi_wind, 4, "sidi_wind_4"), (sidi_avg_rain, 4, "sidi_avg_4"), (sidi_Tg, 24, "sidi_tg_24")   ], set_name = "sidi_pawindavgtg_4", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    f = gzip.open("fognet_compare_test_sidi4d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         
    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 2, "guelmin_2") ], set_name = "guelmin_2", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin_wind, 2, "guelminw_2") ], set_name = "guelminw_2", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 2, "guelmin_2"), (guelmin_wind, 2, "guelminw_2") ], set_name = "guelminwwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 2, "guelmin_2"),(sidi, 2, "sidi_2")  ], set_name = "guelmin_2_sidi2", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    #add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"), (sidi, 6, "sidi_6") ], set_name = "guelminsidi_6", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    #add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"), (sidi, 6, "sidi_6"), (sidi_wind, 6, "sidi_wind_6"), (guelmin_wind, 6, "guelminw_6") ], set_name = "guelminsidiwwind_6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 2, "guelmin_2"),(sidi, 2, "sidi_2") ,(sidi_wind, 2, "sidi_wind_2"),(guelmin_wind, 2, "guelmin_wind_2") ], set_name = "guelminwwind_2_sidiwwind2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 2, "guelmin_2"),(sidi, 2, "sidi_2") ,(sidi_wind, 2, "sidi_wind_2"),(guelmin_wind, 2, "guelmin_wind_2") ,(sidi_pa, 4, "sidi_pa_4") ], set_name = "guelminwwind_2_sidiwwind2_sidipa4", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    f = gzip.open("fognet_compare_test_guelmin2d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()        
    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 4, "guelmin_4") ], set_name = "guelmin_4", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin_wind, 4, "guelminw_4") ], set_name = "guelminw_4", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 4, "guelmin_4"), (guelmin_wind, 4, "guelminw_4") ], set_name = "guelminwwind_4", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 4, "guelmin_4"),(sidi, 4, "sidi_4")  ], set_name = "guelmin_4_sidi4", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    #add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"), (sidi, 6, "sidi_6") ], set_name = "guelminsidi_6", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    #add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"), (sidi, 6, "sidi_6"), (sidi_wind, 6, "sidi_wind_6"), (guelmin_wind, 6, "guelminw_6") ], set_name = "guelminsidiwwind_6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 4, "guelmin_4"),(sidi, 4, "sidi_4") ,(sidi_wind, 4, "sidi_wind_4"),(guelmin_wind, 4, "guelmin_wind_4") ], set_name = "guelminwwind_4_sidiwwind2", evaluations_list =  evaluations_list, cv_list = cv_list )
    
        
                    f = gzip.open("fognet_compare_test_guelmin4d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()        
    
    
    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2") ], set_name = "micro_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2") ], set_name = "micro_sidiguelmin_2", evaluations_list =  evaluations_list, cv_list = cv_list )                                
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (guelmin_wind, 2, "guelminw_2") ], set_name = "micro2_guelminwwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (sidi_wind, 2, "sidiwind_2") ], set_name = "micro2_sidiwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2") ], set_name = "micro2_sidiwwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin, 2, "guelmin_2") ], set_name = "micro2_sidiwindguelmin_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin_wind, 2, "guelminwind_2") ], set_name = "micro2_sidiwwindguelminwwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin_wind, 2, "guelminwind_2"), (sidi_Tg, 24, 'siditg_24') ], set_name = "micro2_sidiwwindguelminwwind_2tg24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin_wind, 2, "guelminwind_2"), (sidi_Tg, 24, 'siditg_24'), (sidi_pa, 6, 'sidipa_6') ], set_name = "micro2_sidiwwindguelminwwind_2tg24pa6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin_wind, 2, "guelminwind_2"), (sidi_Tg, 24, 'siditg_24'), (sidi_pa, 6, 'sidipa_6'), (all_micro_2h_460, 2, 'micro460_2') ], set_name = "micro2_sidiwwindguelminwwind460_2tg24pa6", evaluations_list =  evaluations_list, cv_list = cv_list )
        
        
                    f = gzip.open("fognet_compare_test_micro2d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()        
    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 4, "micro_4") ], set_name = "micro_4", evaluations_list =  evaluations_list, cv_list = cv_list )                                
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 4, "micro_4"), (guelmin, 4, "guelmin_4"), (guelmin_wind, 4, "guelminw_4") ], set_name = "micro4_guelminwwind_4", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 4, "micro_4"), (sidi_wind, 4, "sidiwind_4") ], set_name = "micro2_sidiwind_4", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 4, "micro_4"), (sidi_wind, 4, "sidiwind_4"), (guelmin, 4, "guelmin_4") ], set_name = "micro4_sidiwindguelmin_4", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    f = gzip.open("fognet_compare_test_micro4d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()        
    
            
                if 0:
                    f = gzip.open("fognet_compare_test4.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         



                if 0:
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (agadir, 2, "agadir_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin_wind, 2, "guelminwind_2"), (sidi_Tg, 24, 'siditg_24'), (sidi_pa, 6, 'sidipa_6'), (sidi_avg_rain, 4, "sidi_avg_4"), (all_micro_2h_460, 2, 'micro460_2') ], set_name = "micro2_full_nulls", evaluations_list =  evaluations_list, cv_list = cv_list , do_dropna = False )
        
                    f = gzip.open("fognet_compare_test7.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         

            else:
                f = gzip.open("fognet_compare_test_micro4d.gz","r")
                (evaluations_list, sub_form) = pickle.load(f)
                f.close()         

            if 0:
                add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (agadir, 2, "agadir_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin_wind, 2, "guelminwind_2"), (sidi_Tg, 24, 'siditg_24'), (sidi_pa, 6, 'sidipa_6'), (sidi_avg_rain, 4, "sidi_avg_4"), (all_micro_2h_460, 2, 'micro460_2') ], set_name = "micro2_full_nulls", evaluations_list =  evaluations_list, cv_list = cv_list , do_dropna = False )

                f = gzip.open("fognet_compare_test_micro4d_nulls.gz","wb")
                cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                f.close()        

            if 0:
                # meilleure CV mais leaderboard baisse
                add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24"),(sidi, 2, "sidi_2"),(guelmin, 24, "guelmin_24"),(guelmin, 2, "guelmin_2"),(sidi_wind, 24, "sidi_wind_24"),(sidi_wind, 12, "sidi_wind_12"),(guelmin_wind, 24, "guelmin_wind_24"),(guelmin_wind, 12, "guelmin_wind_12") ], set_name = "guelminwwindsidiwwind_2_12_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24"),(sidi, 6, "sidi_6"),(guelmin, 24, "guelmin_24"),(guelmin, 6, "guelmin_6"),(sidi_wind, 6, "sidi_wind_6"),(guelmin_wind, 6, "guelmin_wind_6") ], set_name = "guelminwwindsidiwwind_6_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"),  (all_micro_2h_460, 2, 'micro460_2') ], set_name = "micro2_460_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 24, "guelmin_24"), (sidi, 24, "sidi_24"), (sidi_wind, 24, "sidiwind_24"), (guelmin_wind, 24, "guelminwind_24") ], set_name = "micro2_sidiwwindguelminwwind24", evaluations_list =  evaluations_list, cv_list = cv_list )
                add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"),  (all_micro_2h_460, 2, 'micro460_2'),(sidi, 2, "sidi_2"),(guelmin, 2, "guelmin_2") ], set_name = "micro2_460_sidiguelmin_2", evaluations_list =  evaluations_list, cv_list = cv_list )

                f = gzip.open("fognet_compare_test_micro5.gz","wb")
                cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                f.close()        


            if 0:                
                add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24"),(guelmin, 24, "guelmin_24"),(sidi_wind, 24, "sidi_wind_24"),(agadir_offset,2, 'agadir_offset_2') ], set_name = "guelminsidiwwind_24_agoff", evaluations_list =  evaluations_list, cv_list = cv_list )
                add_eval_sets(work_model = work_model, set_list = [ (sidi, 12, "sidi_12"),(guelmin, 12, "guelmin_12"),(sidi_wind, 24, "sidi_wind_24"),(agadir_offset,2, 'agadir_offset_2') ], set_name = "guelminsidiwwind_12_agoff", evaluations_list =  evaluations_list, cv_list = cv_list )
                add_eval_sets(work_model = work_model, set_list = [ (sidi, 12, "sidi_12"),(guelmin, 12, "guelmin_12"),(sidi_wind, 12, "sidi_wind_24"),(guelmin_wind, 12, "guelmin_wind_24"),(agadir_offset,2, 'agadir_offset_2') ], set_name = "guelminwwindsidiwwind_12_agoff", evaluations_list =  evaluations_list, cv_list = cv_list )
                
                add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (all_micro_offset, 2, 'micro_off_2') ], set_name = "micro_2_off", evaluations_list =  evaluations_list, cv_list = cv_list )
                add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (all_micro_offset, 2, 'micro_off_2')  ], set_name = "micro_sidiguelmin_2_off", evaluations_list =  evaluations_list, cv_list = cv_list )
                                    
                add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 24, "guelmin_24"), (sidi, 24, "sidi_24"), (sidi_wind, 24, "sidiwind_24"), (guelmin_wind, 24, "guelminwind_24"), (all_micro_offset, 2, 'micro_off_2') ], set_name = "micro2_sidiwwindguelminwwind24_off", evaluations_list =  evaluations_list, cv_list = cv_list )
                add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 12, "guelmin_12"), (sidi, 12, "sidi_12"), (sidi_wind, 12, "sidiwind_12"), (guelmin_wind, 12, "guelminwind_12"), (all_micro_offset, 2, 'micro_off_2') ], set_name = "micro2_sidiwwindguelminwwind12_off", evaluations_list =  evaluations_list, cv_list = cv_list )


                f = gzip.open("fognet_compare_test_micro6.gz","wb")
                cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                f.close()        

            if 0:
                #add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24"),(sidi,12, "sidi_12"), (guelmin, 24, "guelmin_24"),(guelmin, 12, "guelmin_12"), (sidi_wind, 24, "sidi_wind_24"),(sidi_wind, 12, "sidi_wind_12"),(guelmin_wind, 24, "guelmin_wind_24"),(guelmin_wind, 12, "guelmin_wind_12") ], set_name = "guelminwwindsidiwwind_12_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                add_eval_sets(work_model = work_model, set_list = [ (sidi, 4, "sidi_4"), (guelmin, 2, "guelmin_2"), (sidi_wind, 4, "sidi_wind_4"),(guelmin_wind, 2, "guelmin_wind_2") ], set_name = "guelminwwind2sidiwwind4", evaluations_list =  evaluations_list, cv_list = cv_list )
                add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 4, "sidi_4") ], set_name = "micro_sidi4guelmin2", evaluations_list =  evaluations_list, cv_list = cv_list )
                
                # la seule meilleure en cv mais moins bonne
                add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 4, "sidi_4"), (sidi_wind, 4, "sidiwind_4") ], set_name = "micro_sidiwwind4guelmin2", evaluations_list =  evaluations_list, cv_list = cv_list )
                
                add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (guelmin_wind, 2, "guelminwind_2"), (sidi, 4, "sidi_4"), (sidi_wind, 4, "sidiwind_4") ], set_name = "micro_sidiwwind4guelminwwind2", evaluations_list =  evaluations_list, cv_list = cv_list )

                f = gzip.open("fognet_compare_test_micro7.gz","wb")
                cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                f.close()        


            if 0:
                add_eval_sets(work_model = work_model, set_list = [  (agadir, 2, "agadir_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin_wind, 2, "guelminwind_2"),  (guelmin, 6, "guelmin_6"), (sidi, 6, "sidi_6"), (sidi_wind, 6, "sidiwind_6"), (guelmin_wind, 6, "guelminwind_6"),  (guelmin, 24, "guelmin_24"), (sidi, 24, "sidi_24"), (sidi_wind, 24, "sidiwind_24"), (guelmin_wind, 24, "guelminwind_24"), (sidi_Tg, 24, 'siditg_24'), (sidi_pa, 6, 'sidipa_6'), (sidi_avg_rain, 4, "sidi_avg_4") ], set_name = "nomicro_full_nulls", evaluations_list =  evaluations_list, cv_list = cv_list , do_dropna = False )

                f = gzip.open("fognet_compare_test_nomicro_nulls.gz","wb")
                cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                f.close()        

                            
            #comparison_list = compare_evaluations(evaluations_list, sub_form)
            comparison_list = fognet_utils.compare_evaluations2(evaluations_list, sub_form, cv_list)

            if 1:
                f = gzip.open("fognet_comparison_list10.gz","wb")
                cPickle.dump( comparison_list , f,cPickle.HIGHEST_PROTOCOL)
                f.close()
            else:
                f = gzip.open("fognet_comparison_list5.gz","r")
                comparison_list = pickle.load(f)
                f.close()
            
            fognet_utils.generate_valid_sub2(evaluations_list, lin_model.copy(), cv_list = cv_list, comparison_list = comparison_list)
                                            
            sys.exit(0)
                        
            
            if 1:
                group_list = GenerateGroupsBy(sidi_wind, "sidi_wind_24", hours_grouped = 24)
                for grouped_result in group_list:
                    work_model = work_model.join(grouped_result, how='left')
                print(work_model.isnull().sum(axis=0))
                print(work_model.count(axis=0))
                work_model.dropna(inplace=True)
                
                columns_selection = list(work_model.columns)
                columns_selection.remove('yield')
                columns_selection.remove('is_train')
                
                columns_selection_nocorr = remove_correlated_features(work_model, columns_selection)

                regressor = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.005, n_estimators=10000,objective='reg:linear', subsample=0.65, colsample_bytree=0.8, seed=0, reg_lambda=0.97 , reg_alpha=0.2, gamma=1.0, missing = np.NaN )
                
                (pred_model_work_valid, valid_set_work_y, work_model_train, work_valid_indices, work_pred) = process_model(work_model, columns_selection, regressor, need_eval_set = True, fit_params={ 'eval_metric':'rmse','early_stopping_rounds':50})
                print("rmse=", compute_rmse(valid_set_work_y, pred_model_work_valid))

                set_name = "agadirsidiw24"
                work_model_train.loc[work_valid_indices,'yield_%s_predicted' % set_name ] = pred_model_work_valid
                evaluations_list.append((pred_model_work_valid, valid_set_work_y, work_model_train, work_valid_indices, set_name))
                #sub_form = sub_form.join(work_model_train.loc[work_valid_indices,'yield_%s_predicted' % set_name],how='left')

                f = gzip.open("fognet_compare_test.gz","wb")
                cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                f.close()         

                compare_evaluations(evaluations_list, sub_form)
                
                sys.exit(0)
                

            if 1:
                group_list = GenerateGroupsBy(sidi_pa, "sidi_pa_24", hours_grouped = 24)
                for grouped_result in group_list:
                    agadir_model = agadir_model.join(grouped_result, how='left')
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))

            if 1:
                group_list = GenerateGroupsBy(sidi_Tg, "sidi_tg_24", hours_grouped = 24)
                for grouped_result in group_list:
                    agadir_model = agadir_model.join(grouped_result, how='left')
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))

            if 1:
                group_list = GenerateGroupsBy(sidi_avg_rain, "sidi_avg_24", hours_grouped = 24)
                for grouped_result in group_list:
                    agadir_model = agadir_model.join(grouped_result, how='left')
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))
        
            plot_errors(valid_set_agadir_y, pred_model_agadir_valid, agadir_model, agadir_valid_indices)

            sys.exit(0)

            

        
        X = agadir_model.loc[agadir_model['is_train'] == True, columns_selection ].as_matrix().astype(np.float32)
        X_pred = agadir_model.loc[agadir_model['is_train'] == False, columns_selection ].as_matrix().astype(np.float32)
        y = agadir_model.loc[agadir_model['is_train'] == True, "yield" ].as_matrix().astype(np.float32)
        
        agadir_model_train = agadir_model.loc[agadir_model['is_train'] == True]
        valid_indices = (agadir_model_train.index.day <= 4) & (agadir_model_train.index.day >= 1)
        cv_fold_1 = (agadir_model_train.index.day <= 12) & (agadir_model_train.index.day >= 5)
        cv_fold_2 = (agadir_model_train.index.day <= 20) & (agadir_model_train.index.day >= 13)
        cv_fold_3 = (agadir_model_train.index.day <= 31) & (agadir_model_train.index.day >= 21)
        custom_cv = [ (cv_fold_1,  cv_fold_2 | cv_fold_3), (cv_fold_2, cv_fold_1 | cv_fold_3), (cv_fold_3, cv_fold_1 | cv_fold_2) ]
        #print(len(columns_selection))
        #print(X.shape)
        #print(X)
        #sys.exit(0)
        if 0:
            print("PCA: Scale")
            pca_scaler  = StandardScaler()
            pca_scaler.fit(X)
            pca = PCA()
            print("PCA: Fit")
            pca.fit(pca_scaler.transform(X))
            variance  = pca.explained_variance_ratio_
            ratio_cible = 0.9999
            n_vars = 1
            for i in xrange(1,variance.shape[0]+1):
                if np.sum(variance[0:i]) >= ratio_cible:
                    n_vars = i
                    break
            print ("Retain",n_vars,"among",variance.shape[0],"which keep",np.sum(variance[0:n_vars])," of explained variance")
            X = pca.transform(pca_scaler.transform(X))
            X = X[:,0:n_vars].copy().astype(np.float32)
            
            X_pred = pca.transform(pca_scaler.transform(X_pred))
            X_pred = X_pred[:,0:n_vars].copy().astype(np.float32)

            
            

        #X = np.nan_to_num(X)        
        #X = np.nan_to_num(X)
        if 1:
            print "Split train in train / valid"
            #kf = cross_validation.KFold(len(y),n_folds=10,shuffle=True,random_state=0)
            arr_indices = np.arange(X.shape[0])
              
            #train_set_x_tot, valid_set_x, train_set_y_tot, valid_set_y, train_indices, valid_indices = train_test_split(X,y, arr_indices, test_size = 0.2, random_state = 0, stratify = None )
            #X_train, X_valid = train_test_split(X, test_size=0.1, random_state = 1302)
            train_set_x_tot = X[np.logical_not(valid_indices)].copy()
            train_set_y_tot = y[np.logical_not(valid_indices)].copy()
            valid_set_x = X[valid_indices].copy()
            valid_set_y = y[valid_indices].copy()
            
            eval_set=[(train_set_x_tot,train_set_y_tot),(valid_set_x, valid_set_y)]

        if 0:
            
            rmse_base = 0
            custom_cv.append(())
            if 1:
                for cv_set in custom_cv:
                    valid_indices = cv_set[0]
                    train_indices = cv_set[1]
                    
                    train_set_x_tot = X[np.logical_not(valid_indices)].copy()
                    train_set_y_tot = y[np.logical_not(valid_indices)].copy()
                    valid_set_x = X[valid_indices].copy()
                    valid_set_y = y[valid_indices].copy()
                    
                    eval_set=[(train_set_x_tot,train_set_y_tot),(valid_set_x, valid_set_y)]
                    
                    regressor = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.005, n_estimators=10000,objective='reg:linear', subsample=0.65, colsample_bytree=0.8, seed=0, reg_lambda=0.97 , reg_alpha=0.2, gamma=1.0, missing = np.NaN )
                    regressor.fit(train_set_x_tot, train_set_y_tot ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
                    pred_agadir_valid = regressor.predict(valid_set_x)
                    
                    rmse = compute_rmse(valid_set_y, regressor.predict(valid_set_x))                
                    print("rmse=", rmse )
                    rmse_base += rmse
                
                rmse_base = rmse_base / 3.0
            else:
                rmse_base = 2.7449169158935547
            print("rmse_base=", rmse_base)
            garbage_columns = []
            for c in columns_selection:
                print("Trying without", c)
                columns_sub_selection = list(columns_selection)
                columns_sub_selection.remove(c)
            
                X = agadir_model.loc[agadir_model['is_train'] == True, columns_sub_selection ].as_matrix().astype(np.float32)
                y = agadir_model.loc[agadir_model['is_train'] == True, "yield" ].as_matrix().astype(np.float32)

                rmse_sub = 0
                for cv_set in custom_cv:
                    valid_indices = cv_set[0]
                    train_indices = cv_set[1]
                    
                    train_set_x_tot = X[np.logical_not(valid_indices)].copy()
                    train_set_y_tot = y[np.logical_not(valid_indices)].copy()
                    valid_set_x = X[valid_indices].copy()
                    valid_set_y = y[valid_indices].copy()
                    
                    eval_set=[(train_set_x_tot,train_set_y_tot),(valid_set_x, valid_set_y)]
                    
                    regressor = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.005, n_estimators=10000,objective='reg:linear', subsample=0.65, colsample_bytree=0.8, seed=0, reg_lambda=0.97 , reg_alpha=0.2, gamma=1.0, missing = np.NaN )
                    regressor.fit(train_set_x_tot, train_set_y_tot ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
                    pred_agadir_valid = regressor.predict(valid_set_x)
                    
                    rmse = compute_rmse(valid_set_y, regressor.predict(valid_set_x))                
                    print("rmse=", rmse )
                    rmse_sub = rmse_sub + rmse
                
                rmse_sub = rmse_sub / 3.0
                print("Col",c,"rmse_sub=", rmse_sub,"vs",rmse_base)
                #if (rmse_sub < rmse_base):
                garbage_columns.append((c,[rmse_base, rmse_sub]))
                f = gzip.open("garbage_columns_agadir.gz","wb")
                cPickle.dump( garbage_columns , f,cPickle.HIGHEST_PROTOCOL)
                f.close()         
            sys.exit()
                

        
        if 0:
            regressor = XGBRegressor(max_depth=4, silent=True, learning_rate= 0.05, n_estimators=10,objective='reg:linear', subsample=0.9, colsample_bytree=0.9, seed=0, missing = np.NaN )
            # Utility function to report best scores
            param_dist = {"max_depth": [3,4,5],
                          "subsample" : sp_uniform(0.6,0.4),
                          "colsample_bytree" : sp_uniform(0.6,0.4),
                          "gamma" : sp_uniform(0.8,0.4),
                          "reg_lambda" : sp_uniform(0.5,1.0),
                          "reg_alpha" : sp_uniform(0.0,0.5),
                          #"learning_rate" : [0.01,0.001,0.0001,0.00001],
                          "learning_rate" : [0.01],
                          "n_estimators" : [10000]
                        }                  
            n_iter_search = 30

            #eval_set=[(valid_set_x, valid_set_y)]
            random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse','eval_set':eval_set,'early_stopping_rounds':50},cv=custom_cv, scoring = rmse_scoring, refit = False )
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse'},cv=5)
            
            random_search.fit(X, y)
            #random_search.fit(train_set_x_tot, train_set_y_tot) 
            report(random_search.grid_scores_, n_top = 10)
            sys.exit(0)
            # 3.115 / depth 3, gamma 0.168, 
        if 0:
            scaler = StandardScaler()
            #scaler.fit(train_set_x_tot)
            scaler.fit(X)
            regressor = SVR(C=1, kernel='rbf', gamma = 'auto' )
            # Utility function to report best scores
            param_dist = {
                          "C" : sp_uniform(0.5,30.0),
                        }                  
            n_iter_search = 100
            #eval_set=[(valid_set_x, valid_set_y)]
            random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=5,n_iter=n_iter_search,verbose=3,cv=custom_cv, scoring = rmse_scoring, refit = False )
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse'},cv=5)
            
            #random_search.fit(X, y)
            #random_search.fit(scaler.transform(train_set_x_tot), train_set_y_tot) 
            random_search.fit(scaler.transform(X), y)
            report(random_search.grid_scores_, n_top = 10)
            
            sys.exit(0)
            # 3.157 C = 9
            

        if 0:
            regressor = KNeighborsRegressor(weights='distance', algorithm='auto', leaf_size=10, p=2, n_jobs=1 )
            # Utility function to report best scores
            param_dist = {"weights": ['distance','uniform'],
                          #"algorithm": ['ball_tree','kd_tree'],
                          "p" : [1,2,3],
                          "leaf_size" : [1,5,10,20]
                        }                  
            #n_iter_search = 20

            #eval_set=[(valid_set_x, valid_set_y)]
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse','eval_set':eval_set,'early_stopping_rounds':50},cv=3, scoring = rmse_scoring)
            
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse'},cv=5)
            random_search = GridSearchCV(regressor, param_grid=param_dist,cv=custom_cv,scoring = rmse_scoring,n_jobs=-1,verbose=3)
            
            random_search.fit(X, y)
            #random_search.fit(train_set_x_tot, train_set_y_tot) 
            report(random_search.grid_scores_, n_top = 10)
            sys.exit(0)
             # -3.84

        if 0:
            scaler = StandardScaler()
            #scaler.fit(train_set_x_tot)
            scaler.fit(X)
            regressor = SVR(C=1, kernel='rbf', gamma = 'auto' )
            # Utility function to report best scores
            param_dist = {
                          "C" : sp_uniform(0.5,30.0),
                        }                  
            n_iter_search = 100
            #eval_set=[(valid_set_x, valid_set_y)]
            random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=5,n_iter=n_iter_search,verbose=3,cv=custom_cv, scoring = rmse_scoring, refit = False )
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse'},cv=5)
            
            #random_search.fit(X, y)
            #random_search.fit(scaler.transform(train_set_x_tot), train_set_y_tot) 
            random_search.fit(scaler.transform(X), y)
            report(random_search.grid_scores_, n_top = 10)
            
            sys.exit(0)

        if 0:
            scaler = StandardScaler()
            #scaler.fit(train_set_x_tot)
            scaler.fit(X)

            regressor = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, random_state=0)
            # Utility function to report best scores
            param_dist = {"loss": ['squared_loss', 'huber', 'epsilon_insensitive',  'squared_epsilon_insensitive'],
                          "penalty" : ['none','l2','l1','elasticnet'],
                          #"l1_ratio" : [ 0.01,0.15,0.30,0.50],
                          "l1_ratio" : [ 0.15],
                          #"n_iter" : [20,50,100],
                          "n_iter" : [20],
                          #"power_t" : [0.1,0.25,0.5,0.75],
                          "power_t" : [0.25,0.5,0.75],
                          #"epsilon" : [0.0001,0.001,0.01,0.1],
                          "epsilon" : [0.0001],
                          #"learning_rate" : ["constant","optimal","invscaling"]
                          "learning_rate" : ["invscaling"]
                        }                  
            #n_iter_search = 20

            #eval_set=[(valid_set_x, valid_set_y)]
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse','eval_set':eval_set,'early_stopping_rounds':50},cv=3, scoring = rmse_scoring)
            
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse'},cv=5)
            random_search = GridSearchCV(regressor, param_grid=param_dist,cv=custom_cv,scoring = rmse_scoring,n_jobs=-1,verbose=3, refit=False)
            
            random_search.fit(scaler.transform(X), y)
            #random_search.fit(train_set_x_tot, train_set_y_tot) 
            report(random_search.grid_scores_, n_top = 30)
            sys.exit(0)
            # 3.179


            """
Model with rank: 1
Mean validation score: -3.318 (std: 0.251)
Parameters: {'reg_alpha': 0.2071842941131844, 'colsample_bytree': 0.7483928553654988, 'learning_rate': 0.01, 'n_estimators': 10000, 'subsample': 0.8870530303395604, 'reg_lambda': 0.9746975022884129, 'max_depth': 3, 'gamma': 0.13062166509307968}

Model with rank: 2
Mean validation score: -3.320 (std: 0.240)
Parameters: {'reg_alpha': 0.1326947454697227, 'colsample_bytree': 0.7650466412882628, 'learning_rate': 0.01, 'n_estimators': 10000, 'subsample': 0.7281821532275324, 'reg_lambda': 1.0232480534666997, 'max_depth': 3, 'gamma': 0.11303777332097507}

Model with rank: 3
Mean validation score: -3.327 (std: 0.254)
Parameters: {'reg_alpha': 0.0690914756743069, 'colsample_bytree': 0.8014022844516675, 'learning_rate': 0.01, 'n_estimators': 10000, 'subsample': 0.8106175511982892, 'reg_lambda': 0.6965823616800535, 'max_depth': 3, 'gamma': 0.13495046445180414}

Model with rank: 4
Mean validation score: -3.328 (std: 0.251)
Parameters: {'reg_alpha': 0.48183138025051464, 'colsample_bytree': 0.8153145121878099, 'learning_rate': 0.01, 'n_estimators': 10000, 'subsample': 0.9375175114247993, 'reg_lambda': 0.8834415188257777, 'max_depth': 3, 'gamma': 0.059506921308894456}

Model with rank: 5
Mean validation score: -3.336 (std: 0.234)
Parameters: {'reg_alpha': 0.36963178969915084, 'colsample_bytree': 0.85288731301597, 'learning_rate': 0.01, 'n_estimators': 10000, 'subsample': 0.7848420887729228, 'reg_lambda': 0.5391877922543207, 'max_depth': 4, 'gamma': 0.011142938740321262}

Model with rank: 6
Mean validation score: -3.341 (std: 0.249)
Parameters: {'reg_alpha': 0.07483743359184158, 'colsample_bytree': 0.8705301846605945, 'learning_rate': 0.01, 'n_estimators': 10000, 'subsample': 0.8159466943377586, 'reg_lambda': 0.7223213882515876, 'max_depth': 4, 'gamma': 0.0037579600872710284}

Model with rank: 7
Mean validation score: -3.344 (std: 0.246)
Parameters: {'reg_alpha': 0.41803938176868877, 'colsample_bytree': 0.8586684759258713, 'learning_rate': 0.01, 'n_estimators': 10000, 'subsample': 0.8944515616153591, 'reg_lambda': 0.8373961604172684, 'max_depth': 4, 'gamma': 0.11360891221878647}

Model with rank: 8
Mean validation score: -3.358 (std: 0.238)
Parameters: {'reg_alpha': 0.4941869190296131, 'colsample_bytree': 0.7975141687025057, 'learning_rate': 0.01, 'n_estimators': 10000, 'subsample': 0.7626630268284503, 'reg_lambda': 0.6020448107480281, 'max_depth': 6, 'gamma': 0.007685085294546945}

Model with rank: 9
Mean validation score: -3.360 (std: 0.246)
Parameters: {'reg_alpha': 0.3603163273629584, 'colsample_bytree': 0.8384438086758795, 'learning_rate': 0.01, 'n_estimators': 10000, 'subsample': 0.8612119688347032, 'reg_lambda': 1.0820197920751071, 'max_depth': 5, 'gamma': 0.1561058352572911}

Model with rank: 10
Mean validation score: -3.365 (std: 0.246)
Parameters: {'reg_alpha': 0.3518686396449581, 'colsample_bytree': 0.701408642857764, 'learning_rate': 0.01, 'n_estimators': 10000, 'subsample': 0.8299864185946132, 'reg_lambda': 0.7884764370485287, 'max_depth': 6, 'gamma': 0.13556330735924602}

            """

        if 1:
            regressor = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.005, n_estimators=10000,objective='reg:linear', subsample=0.65, colsample_bytree=0.8, seed=0, reg_lambda=0.97 , reg_alpha=0.2, gamma=1.0, missing = np.NaN )
            regressor.fit(train_set_x_tot, train_set_y_tot ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
            pred_agadir_valid = regressor.predict(valid_set_x)
            f = gzip.open("pred_agadir_valid.gz","wb")
            cPickle.dump( (pred_agadir_valid, valid_set_y, agadir_model_train, valid_indices) , f,cPickle.HIGHEST_PROTOCOL)
            f.close()         
            
            if 0:
                f = gzip.open("agadir_xgb_8_01_165.gz","wb")
                cPickle.dump( regressor , f,cPickle.HIGHEST_PROTOCOL)
                f.close()         
            

        if 0:
            regressor = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.01, n_estimators=2900,objective='reg:linear', subsample=0.887, colsample_bytree=0.75, seed=0, reg_lambda=0.97 , reg_alpha=0.2, gamma=0.13, missing = np.NaN )
            regressor.fit(X, y ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
            #regressor.fit(train_set_x_tot, train_set_y_tot ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
            print("ok")
            if 0:
                f = gzip.open("agadir_xgb_3_403.gz","wb")
                cPickle.dump( regressor , f,cPickle.HIGHEST_PROTOCOL)
                f.close()         

        if 0:
            regressor = KNeighborsRegressor(weights='uniform', algorithm='auto', leaf_size=10, p=1, n_jobs=1 )            
            regressor.fit(train_set_x_tot, train_set_y_tot)
            #regressor.fit(X, y)
            #regressor.fit(train_set_x_tot, train_set_y_tot ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
            print("ok")
            if 0:
                f = gzip.open("agadir_clouds_knn_2.71.gz","wb")
                cPickle.dump( regressor , f,cPickle.HIGHEST_PROTOCOL)
                f.close()         
        
        if 0:
            scaler = StandardScaler()
            #scaler.fit(train_set_x_tot)
            scaler.fit(X)

            regressor = SGDRegressor(loss='squared_epsilon_insensitive', epsilon= 0.0001, learning_rate = 'invscaling', penalty='l2', alpha=0.0001, power_t = 0.5, n_iter=20, l1_ratio=0.15, random_state=0)

            train_set_x_tot = scaler.transform(train_set_x_tot)
            valid_set_x = scaler.transform(valid_set_x)
            #regressor.fit(scaler.transform(X), y)
            regressor.fit(train_set_x_tot, train_set_y_tot)

        if 1:
            scaler = StandardScaler()
            #scaler.fit(train_set_x_tot)
            scaler.fit(X)

        if 0:
            regressor = SVR(C=10, kernel='rbf', gamma = 'auto', )
            #regressor.fit(X, y ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
            train_set_x_tot = scaler.transform(train_set_x_tot)
            valid_set_x = scaler.transform(valid_set_x)
            #regressor.fit(scaler.transform(X), y)
            regressor.fit(train_set_x_tot, train_set_y_tot)
            # 3.81
            print("ok")
            if 0:
                f = gzip.open("agadir_SVC_3.8.gz","wb")
                cPickle.dump( regressor , f,cPickle.HIGHEST_PROTOCOL)
                f.close()         


        print("rmse=", compute_rmse(valid_set_y, regressor.predict(valid_set_x)))
            
        
        plot_errors(valid_set_y, regressor.predict(valid_set_x), agadir_model, valid_indices)

        sys.exit(0)
        if 0:
            #f = gzip.open("agadir_SVC_2.39.gz","r")
            f = gzip.open("agadir_SVC_3.8.gz","r")
            #f = gzip.open("agadir_clouds_knn_2.71.gz","r")
            #f = gzip.open("agadir_xgb_3_403.gz","r")
            regressor = pickle.load(f)
            f.close()         


        y_pred = regressor.predict(scaler.transform(X_pred))
        #y_pred = regressor.predict(X_pred)
        #print(y_pred)
        agadir_model.loc[agadir_model['is_train'] == False,'yield_agadir' ] = y_pred         
        agadir_model.loc[agadir_model['is_train'] == True,'yield_agadir' ] = agadir_model.loc[agadir_model['is_train'] == True, 'yield']
                
        pred_agadir_valid = regressor.predict(valid_set_x)
        f = gzip.open("pred_agadir_valid.gz","wb")
        cPickle.dump( (pred_agadir_valid, valid_set_y, agadir_model_train, valid_indices) , f,cPickle.HIGHEST_PROTOCOL)
        f.close()         


        
        if 0:
            sub_form = lin_model[lin_model['is_train'] == False]
            sub_form = sub_form.join(agadir_model.loc[agadir_model['is_train'] == False,'yield_agadir'])
            sub_form['yield'] = sub_form['yield_agadir']
            print(sub_form.isnull().sum(axis=0))
            print(sub_form.describe())
            print("Missing predicts:", np.sum(sub_form['yield_agadir'].isnull()))
            sub_form.loc[sub_form['yield_agadir'].isnull(), 'yield'] = sub_form['yield_lin']
            print("Negative predicts:", np.sum(sub_form['yield'] < 0 ))
            print(sub_form.loc[sub_form['yield'] < 0, 'yield'] )
            sub_form.loc[sub_form['yield']< 0.0, 'yield'] = 0.0
            sub_form['yield'].to_csv('fognet_sub_aga_clouds_svc.csv')
            sys.exit(0)
        

    if 1:
        # Model based only on agadir
        print("Agadir+ model")

        agadir_model2 = agadir_model.copy()

        if 1:
            group_list = GenerateGroupsBy(agadir, "agadir_2", hours_grouped = 2)            
            for grouped_result in group_list:
                agadir_model2 = agadir_model2.join(grouped_result)
                print(agadir_model2.isnull().sum(axis=0))
                print(agadir_model2.count(axis=0))
                
            if 1:
                group_list = GenerateGroupsBy(sidi, "sidi_2", hours_grouped = 2)
                for grouped_result in group_list:
                    agadir_model2 = agadir_model2.join(grouped_result, how='left')
                print(agadir_model2.isnull().sum(axis=0))
                print(agadir_model2.count(axis=0))                                                

            if 1:
                group_list = GenerateGroupsBy(guelmin, "guelmin_2", hours_grouped = 2)
                for grouped_result in group_list:
                    agadir_model2 = agadir_model2.join(grouped_result, how='left')
                print(agadir_model2.isnull().sum(axis=0))
                print(agadir_model2.count(axis=0))                                                

            print(agadir_model2.isnull().sum(axis=0))
            agadir_model2.dropna(inplace=True)
        
            if 1:
                group_list = GenerateGroupsBy(sidi_wind, "sidi_wind_2", hours_grouped = 2)
                for grouped_result in group_list:
                    agadir_model2 = agadir_model2.join(grouped_result, how='left')
                    
                #agadir_model2.fillna(-1, inplace=True)
                print(agadir_model2.isnull().sum(axis=0))
                print(agadir_model2.count(axis=0))                                                

            if 1:
                group_list = GenerateGroupsBy(guelmin_wind, "guelmin_wind_2", hours_grouped = 2)
                for grouped_result in group_list:
                    agadir_model2 = agadir_model2.join(grouped_result, how='left')
                print(agadir_model2.isnull().sum(axis=0))
                print(agadir_model2.count(axis=0))
                #agadir_model2.fillna(-1, inplace=True)                                                

            if 2:
                for c in agadir_model2.columns:
                    print("Processing column",c)
                    if c == "yield":
                        continue
                    if c == "is_train":
                        continue
                        
                    nb_nulls =  np.sum(agadir_model2[c].isnull())
                    print ("There are", nb_nulls, "nulls in this column")
                    if nb_nulls > 0:
                        #agadir_model2["%s_nulls" % c] = agadir_model2[c].isnull()
                        agadir_model2[c].fillna(-2, inplace=True)

        
        #print(agadir_model.isnull().sum(axis=0))
        print(agadir_model2.count(axis=0))
        print(agadir_model2.describe())
        #sys.exit(0)

        columns_selection = list(agadir_model2.columns)
        columns_selection.remove('yield')
        columns_selection.remove('is_train')
        columns_selection.remove('yield_agadir')
        
        #for c in columns_selection:
        #    print(c)
        #sys.exit(0)
        
        X = agadir_model2.loc[agadir_model2['is_train'] == True, columns_selection ].as_matrix().astype(np.float32)
        y = agadir_model2.loc[agadir_model2['is_train'] == True, "yield" ].as_matrix().astype(np.float32)
        X_pred = agadir_model2.loc[agadir_model2['is_train'] == False, columns_selection ].as_matrix().astype(np.float32)
        
        agadir_model2_train = agadir_model2.loc[agadir_model2['is_train'] == True]
        valid_indices = (agadir_model2_train.index.day <= 4) & (agadir_model2_train.index.day >= 1)
        cv_fold_1 = (agadir_model2_train.index.day <= 12) & (agadir_model2_train.index.day >= 5)
        cv_fold_2 = (agadir_model2_train.index.day <= 20) & (agadir_model2_train.index.day >= 13)
        cv_fold_3 = (agadir_model2_train.index.day <= 31) & (agadir_model2_train.index.day >= 21)
        custom_cv = [ (cv_fold_1,  cv_fold_2 | cv_fold_3), (cv_fold_2, cv_fold_1 | cv_fold_3), (cv_fold_3, cv_fold_1 | cv_fold_2) ]

        f = gzip.open("fognet_agadir2.gz","wb")
        cPickle.dump( (agadir_model2, columns_selection)  , f,cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        if 1:
            print "Split train in train / valid"
            #kf = cross_validation.KFold(len(y),n_folds=10,shuffle=True,random_state=0)
            arr_indices = np.arange(X.shape[0])
              
            #train_set_x_tot, valid_set_x, train_set_y_tot, valid_set_y, train_indices, valid_indices = train_test_split(X,y, arr_indices, test_size = 0.2, random_state = 0, stratify = None )
            #X_train, X_valid = train_test_split(X, test_size=0.1, random_state = 1302)
            train_set_x_tot = X[np.logical_not(valid_indices)].copy()
            train_set_y_tot = y[np.logical_not(valid_indices)].copy()
            valid_set_x = X[valid_indices].copy()
            valid_set_y = y[valid_indices].copy()
        
            eval_set=[(train_set_x_tot,train_set_y_tot),(valid_set_x, valid_set_y)]

        if 0:
            regressor = XGBRegressor(max_depth=4, silent=True, learning_rate= 0.05, n_estimators=10,objective='reg:linear', subsample=0.9, colsample_bytree=0.9, seed=0, missing = np.NaN )
            # Utility function to report best scores
            param_dist = {"max_depth": [3,4,5,6,7,8,9],
                          "subsample" : sp_uniform(0.7,0.3),
                          "colsample_bytree" : sp_uniform(0.7,0.3),
                          "gamma" : sp_uniform(0.0,0.2),
                          "reg_lambda" : sp_uniform(0.5,1.0),
                          "reg_alpha" : sp_uniform(0.0,0.5),
                          #"learning_rate" : [0.01,0.001,0.0001,0.00001],
                          "learning_rate" : [0.005],
                          "n_estimators" : [10000]
                        }                  
            n_iter_search = 30

            #eval_set=[(valid_set_x, valid_set_y)]
            random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse','eval_set':eval_set,'early_stopping_rounds':50},cv=custom_cv, scoring = rmse_scoring, refit = False)
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse'},cv=5)
            
            random_search.fit(X, y)
            #random_search.fit(train_set_x_tot, train_set_y_tot) 
            report(random_search.grid_scores_, n_top = 10)
            sys.exit(0)
            """
Model with rank: 1
Mean validation score: -2.664 (std: 0.534)
Parameters: {'reg_alpha': 0.4583614770097304, 'colsample_bytree': 0.8699804362619725, 'learning_rate': 0.005, 'n_estimators': 10000, 'subsample': 0.7249337477891806, 'reg_lambda': 1.4211576102371999, 'max_depth': 6, 'gamma': 0.053077898187889085}

Model with rank: 2
Mean validation score: -2.672 (std: 0.546)
Parameters: {'reg_alpha': 0.47478552672537105, 'colsample_bytree': 0.896898876839582, 'learning_rate': 0.005, 'n_estimators': 10000, 'subsample': 0.704071490683633, 'reg_lambda': 1.1625268669500444, 'max_depth': 4, 'gamma': 0.02763659026972276}

Model with rank: 3
Mean validation score: -2.674 (std: 0.539)
Parameters: {'reg_alpha': 0.3571206497745557, 'colsample_bytree': 0.8343778515612189, 'learning_rate': 0.005, 'n_estimators': 10000, 'subsample': 0.7448344913973981, 'reg_lambda': 1.4988470065678665, 'max_depth': 7, 'gamma': 0.019913817822162217}


            """            
        if 0:
            scaler = StandardScaler()
            #scaler.fit(train_set_x_tot)
            scaler.fit(X)
            regressor = SVR(C=1, kernel='rbf', gamma = 'auto' )
            # Utility function to report best scores
            param_dist = {
                          "C" : sp_uniform(0.5,30.0),
                        }                  
            n_iter_search = 100
            #eval_set=[(valid_set_x, valid_set_y)]
            random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=5,n_iter=n_iter_search,verbose=3,cv=custom_cv, scoring = rmse_scoring, refit = False )
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse'},cv=5)
            
            #random_search.fit(X, y)
            #random_search.fit(scaler.transform(train_set_x_tot), train_set_y_tot) 
            random_search.fit(scaler.transform(X), y)
            report(random_search.grid_scores_, n_top = 10)
            
            sys.exit(0)
            # 2.689 avec C=20
        if 1:
            regressor = XGBRegressor(max_depth=6, silent=True, learning_rate= 0.005, n_estimators=10000,objective='reg:linear', subsample=0.73, colsample_bytree=0.87, seed=0, reg_lambda=1.4 , reg_alpha=0.45, gamma=0.05, missing = np.NaN )
            regressor.fit(train_set_x_tot, train_set_y_tot ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)

            pred_agadir2_valid = regressor.predict(valid_set_x)
            f = gzip.open("pred_agadir2_valid.gz","wb")
            cPickle.dump( (pred_agadir2_valid, valid_set_y, agadir_model2_train, valid_indices) , f,cPickle.HIGHEST_PROTOCOL)
            f.close()         
            
            if 1:
                f = gzip.open("agadir2_xgb_8_01_081.gz","wb")
                cPickle.dump( regressor , f,cPickle.HIGHEST_PROTOCOL)
                f.close()         

        if 0:
            scaler = StandardScaler()
            #scaler.fit(train_set_x_tot)
            scaler.fit(X)

        if 0:
            regressor = SVR(C=8, kernel='rbf', gamma = 'auto', )
            #regressor.fit(X, y ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
            train_set_x_tot = scaler.transform(train_set_x_tot)
            valid_set_x = scaler.transform(valid_set_x)
            #regressor.fit(scaler.transform(X), y)
            regressor.fit(train_set_x_tot, train_set_y_tot)
            pred_agadir2_valid = regressor.predict(valid_set_x)
            f = gzip.open("pred_agadir2_valid.gz","wb")
            cPickle.dump( (pred_agadir2_valid, valid_set_y, agadir_model2_train, valid_indices) , f,cPickle.HIGHEST_PROTOCOL)
            f.close()         
            
            # 3.81
            print("ok")
            if 0:
                f = gzip.open("agadir_SVC_2.9.gz","wb")
                cPickle.dump( regressor , f,cPickle.HIGHEST_PROTOCOL)
                f.close()         


        if 1:        
            y_hat = regressor.predict(valid_set_x)
            rmse = np.sqrt(mean_squared_error(valid_set_y, y_hat))
            print("rmse=",rmse)
            #sys.exit(0)
            
        
        if 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.scatter(valid_set_y, y_hat)
            
            plt.xlabel('actual', fontsize=20)
            plt.ylabel('predicted', fontsize=20)
            plt.plot(np.linspace(0, 35), np.linspace(0, 35), label="$y=x$")
            
            plt.xlim(0, 35)
            plt.ylim(0, 35)
            plt.legend(loc='upper left', fontsize=20)
            plt.show()
            
            
            fig, ax = plt.subplots(figsize=(16, 4))
            err = valid_set_y - y_hat
            
            print(agadir_model[agadir_model['is_train'] == True].reset_index().columns)
            valid_dates = agadir_model[agadir_model['is_train'] == True].reset_index()['index'][valid_indices]
            ax.plot_date( valid_dates, err, c='r', ms=3)
            ax.set_title('residuals on test data (each)', fontsize=20)
            ax.set_ylabel('error')
            plt.show()
                        
            fig, ax = plt.subplots(figsize=(16, 4))
            plt.hist(err, bins=20, normed=True)
            plt.title('residuals on test data (distribution)', fontsize=20)
            plt.xlim(-20, 20)

            plt.show()
            sys.exit(0)
         
        if 0:
            f = gzip.open("agadir_SVC_2.9.gz","r")
            regressor = pickle.load(f)
            f.close()         

        #y_pred = regressor.predict(scaler.transform(X_pred))
        y_pred = regressor.predict(X_pred)
        #print(y_pred)
        agadir_model2.loc[agadir_model2['is_train'] == False,'yield_agadir2' ] = y_pred         
        agadir_model2.loc[agadir_model2['is_train'] == True,'yield_agadir2' ] = agadir_model2.loc[agadir_model2['is_train'] == True, 'yield']
                



        
        if 0:
            f = gzip.open("pred_agadir_valid.gz","r")
            (pred_agadir_valid, valid_set_y, agadir_model_train, valid_indices) = pickle.load(f)
            f.close()        
            pred_agadir_valid[pred_agadir_valid < 0] = 0.0 
            agadir_model_train.loc[valid_indices,'yield_agadir_predicted'] = pred_agadir_valid

            y_hat = pred_agadir_valid
            rmse = np.sqrt(mean_squared_error(valid_set_y, y_hat))
            print("rmse1=",rmse)
                        
            f = gzip.open("pred_agadir2_valid.gz","r")
            (pred_agadir2_valid, valid_set_y2, agadir_model2_train, valid_indices2) = pickle.load(f)
            f.close()         
            pred_agadir2_valid[pred_agadir2_valid < 0] = 0.0
            
            agadir_model2_train.loc[valid_indices2,'yield_agadir2_predicted'] = pred_agadir2_valid
            
            sub_form = lin_model[lin_model['is_train'] == True]
            sub_form = sub_form[ (sub_form.index.day <= 4) & (sub_form.index.day >= 1)]
            
            sub_form = sub_form.join(agadir_model_train.loc[valid_indices,'yield_agadir_predicted'],how='left')
            
            sub_form.loc[np.logical_not(sub_form['yield_agadir_predicted'].isnull()), 'yield_predicted'] = sub_form.loc[np.logical_not(sub_form['yield_agadir_predicted'].isnull()),'yield_agadir_predicted']
            
            y_hat = sub_form.loc[np.logical_not(sub_form['yield_agadir_predicted'].isnull()), 'yield_predicted'].as_matrix()
            y = sub_form.loc[np.logical_not(sub_form['yield_agadir_predicted'].isnull()),'yield'].as_matrix()

            rmse = np.sqrt(mean_squared_error(y, y_hat))
            print("rmse2=",rmse)
            
            #sub_form['yield_predicted'] = sub_form['yield_agadir_predicted']
            print("Missing predicts for agadir:", np.sum(sub_form['yield_agadir_predicted'].isnull()))
            #sub_form.loc[sub_form['yield_agadir_predicted'].isnull(), 'yield_predicted'] = 0.0 #sub_form['yield_lin']
            
            print("Join agadir2")
            sub_form = sub_form.join(agadir_model2_train.loc[valid_indices2,'yield_agadir2_predicted'], how='left')
            
            print("Predicts from agadir2:", np.sum(np.logical_not(sub_form['yield_agadir2_predicted'].isnull())))
            sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()), 'yield_predicted' ] = sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()),'yield_agadir2_predicted']
            
            
            y_hat = sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()),'yield_predicted'].as_matrix()
            y = sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()),'yield'].as_matrix()

            rmse = np.sqrt(mean_squared_error(y, y_hat))
            print("rmse3=",rmse)

            y_hat = sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()),'yield_agadir_predicted'].as_matrix()
            y = sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()),'yield'].as_matrix()

            rmse = np.sqrt(mean_squared_error(y, y_hat))
            print("rmse3b=",rmse)

            y_hat = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield_predicted'].as_matrix()
            y = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield'].as_matrix()

            rmse = np.sqrt(mean_squared_error(y, y_hat))
            print("rmse4=",rmse)

            f = gzip.open("subform_debug.gz","wb")
            cPickle.dump( sub_form  , f,cPickle.HIGHEST_PROTOCOL)
            f.close()         

            print(sub_form.isnull().sum(axis=0))
            print(sub_form.describe())

            sub_form.loc[sub_form['yield_predicted'].isnull(), 'yield_predicted'] = 0.0
            
            
            if 1:        
                y_hat = sub_form['yield_predicted'].as_matrix()
                rmse = np.sqrt(mean_squared_error(sub_form['yield'].as_matrix(), y_hat))
                print("rmse=",rmse)
                #sys.exit(0)
                
            
            if 1:
                fig, ax = plt.subplots(figsize=(8, 8))
                plt.scatter(sub_form['yield'].as_matrix(), y_hat)
                
                plt.xlabel('actual', fontsize=20)
                plt.ylabel('predicted', fontsize=20)
                plt.plot(np.linspace(0, 35), np.linspace(0, 35), label="$y=x$")
                
                plt.xlim(0, 35)
                plt.ylim(0, 35)
                plt.legend(loc='upper left', fontsize=20)
                plt.show()
                
                
                fig, ax = plt.subplots(figsize=(16, 4))
                err = sub_form['yield'].as_matrix() - y_hat
                
                valid_dates = sub_form.reset_index()['index']
                ax.plot_date( valid_dates, err, c='r', ms=3)
                ax.set_title('residuals on test data (each)', fontsize=20)
                ax.set_ylabel('error')
                plt.show()
                            
                fig, ax = plt.subplots(figsize=(16, 4))
                plt.hist(err, bins=20, normed=True)
                plt.title('residuals on test data (distribution)', fontsize=20)
                plt.xlim(-20, 20)
    
                plt.show()
                sys.exit(0)
            
            sys.exit(0)

        
        if 0:
            sub_form = lin_model[lin_model['is_train'] == False]
            print("There are", sub_form.shape, "values to submit")
            sub_form = sub_form.join(agadir_model.loc[agadir_model['is_train'] == False,'yield_agadir'])
            sub_form['yield'] = sub_form['yield_agadir']
            print(sub_form.isnull().sum(axis=0))
            print(sub_form.describe())
            print("Missing predicts for agadir:", np.sum(sub_form['yield_agadir'].isnull()))
            sub_form.loc[sub_form['yield_agadir'].isnull(), 'yield'] = sub_form['yield_lin']
            
            print("Join agadir2")
            sub_form = sub_form.join(agadir_model2.loc[agadir_model2['is_train'] == False, 'yield_agadir2'], how='left')
            print("Predicts from agadir:", np.sum(np.logical_not(sub_form['yield_agadir2'].isnull())))
            sub_form.loc[np.logical_not(sub_form['yield_agadir2'].isnull()), 'yield' ] = sub_form['yield_agadir2']                        
            
            print("Negative predicts:", np.sum(sub_form['yield'] < 0 ))
            print(sub_form.loc[sub_form['yield'] < 0, 'yield'] )
            sub_form.loc[sub_form['yield']< 0.0, 'yield'] = 0.0
            sub_form['yield'].to_csv('fognet_sub_aga2.csv')
            
            sys.exit(0)

    if 1:
        # Model based on 5mn data
        print("5mn  model")

        #all_micro_5mn = pd.concat([train_micro_5mn, test_micro_5mn]).sort_index()    
        #all_micro_2h = pd.concat([train_micro_2h, test_micro_2h]).sort_index()


        fivemn_model = agadir_model.copy()

        if 0:
            group_list = GenerateGroupsBy(sidi_wind, "sidi_wind_24", hours_grouped = 24)
            for grouped_result in group_list:
                agadir_model = agadir_model.join(grouped_result, how='left')
            print(agadir_model.isnull().sum(axis=0))
            print(agadir_model.count(axis=0))
                
        if 1:
            group_list = GenerateGroupsBy(all_micro_2h, "micro_2h",hours_grouped = 4)
            for grouped_result in group_list:
                fivemn_model = fivemn_model.join(grouped_result)
            print(fivemn_model.isnull().sum(axis=0))
            print(fivemn_model.count(axis=0))

        if 0:
            group_list = GenerateGroupsBy(agadir, "guelmin_6", hours_grouped = 6)            
            for grouped_result in group_list:
                agadir_model = agadir_model.join(grouped_result)
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))
            
        if 0:
            group_list = GenerateGroupsBy(sidi, "sidi_6", hours_grouped = 6)
            for grouped_result in group_list:
                agadir_model = agadir_model.join(grouped_result, how='left')
            print(agadir_model.isnull().sum(axis=0))
            print(agadir_model.count(axis=0))
                
        if 0:
            group_list = GenerateGroupsBy(sidi_wind, "sidi_wind_6", hours_grouped = 6)
            for grouped_result in group_list:
                agadir_model = agadir_model.join(grouped_result, how='left')
            print(agadir_model.isnull().sum(axis=0))
            print(agadir_model.count(axis=0))
        
                    
        print(fivemn_model.isnull().sum(axis=0))    
        fivemn_model.dropna(inplace=True)
        
        print(fivemn_model.shape)

        columns_selection = list(fivemn_model.columns)
        columns_selection.remove('yield')
        columns_selection.remove('is_train')
        columns_selection.remove('yield_agadir')
        print(columns_selection)
        
        X = fivemn_model.loc[fivemn_model['is_train'] == True, columns_selection ].as_matrix().astype(np.float32)
        y = fivemn_model.loc[fivemn_model['is_train'] == True, "yield" ].as_matrix().astype(np.float32)
        print(y.shape)
        
        X_pred = fivemn_model.loc[fivemn_model['is_train'] == False, columns_selection ].as_matrix().astype(np.float32)
        
        fivemn_model_train = fivemn_model.loc[fivemn_model['is_train'] == True]
        valid_indices = (fivemn_model_train.index.day <= 4) & (fivemn_model_train.index.day >= 1)
        cv_fold_1 = (fivemn_model_train.index.day <= 12) & (fivemn_model_train.index.day >= 5)
        cv_fold_2 = (fivemn_model_train.index.day <= 20) & (fivemn_model_train.index.day >= 13)
        cv_fold_3 = (fivemn_model_train.index.day <= 31) & (fivemn_model_train.index.day >= 21)
        custom_cv = [ (cv_fold_1,  cv_fold_2 | cv_fold_3), (cv_fold_2, cv_fold_1 | cv_fold_3), (cv_fold_3, cv_fold_1 | cv_fold_2) ]

        f = gzip.open("fognet_fivemn.gz","wb")
        cPickle.dump( (fivemn_model, columns_selection)  , f,cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        if 1:
            print "Split train in train / valid"
            #kf = cross_validation.KFold(len(y),n_folds=10,shuffle=True,random_state=0)
            arr_indices = np.arange(X.shape[0])
              
            #train_set_x_tot, valid_set_x, train_set_y_tot, valid_set_y, train_indices, valid_indices = train_test_split(X,y, arr_indices, test_size = 0.2, random_state = 0, stratify = None )
            #X_train, X_valid = train_test_split(X, test_size=0.1, random_state = 1302)
            train_set_x_tot = X[np.logical_not(valid_indices)].copy()
            train_set_y_tot = y[np.logical_not(valid_indices)].copy()
            valid_set_x = X[valid_indices].copy()
            valid_set_y = y[valid_indices].copy()
        
            eval_set=[(train_set_x_tot,train_set_y_tot),(valid_set_x, valid_set_y)]

        if 0:
            regressor = XGBRegressor(max_depth=4, silent=True, learning_rate= 0.05, n_estimators=10,objective='reg:linear', subsample=0.9, colsample_bytree=0.9, seed=0, missing = np.NaN )
            # Utility function to report best scores
            param_dist = {"max_depth": [2,3,4,5,6],
                          "subsample" : sp_uniform(0.7,0.3),
                          "colsample_bytree" : sp_uniform(0.7,0.3),
                          "gamma" : sp_uniform(0.0,0.2),
                          "reg_lambda" : sp_uniform(0.5,1.0),
                          "reg_alpha" : sp_uniform(0.0,0.5),
                          #"learning_rate" : [0.01,0.001,0.0001,0.00001],
                          "learning_rate" : [0.01,0.001],
                          "n_estimators" : [30000]
                        }                  
            n_iter_search = 100
            
            random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse','eval_set':eval_set,'early_stopping_rounds':50},cv=5)
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse'},cv=5)
            
            #random_search.fit(X, y)
            random_search.fit(train_set_x_tot, train_set_y_tot) 
            report(random_search.grid_scores_, n_top = 10)
            sys.exit(0)
        if 0:
            scaler = StandardScaler()
            scaler.fit(X)
            regressor = SVR(C=1, kernel='rbf', gamma = 'auto' )
            # Utility function to report best scores
            param_dist = {
                          "C" : sp_uniform(1,20.0),
                        }                  
            n_iter_search = 30
            #eval_set=[(valid_set_x, valid_set_y)]
            random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=5,n_iter=n_iter_search,verbose=3,cv=custom_cv)
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse'},cv=5)
            
            #random_search.fit(X, y)
            random_search.fit(scaler.transform(X), y) 
            report(random_search.grid_scores_, n_top = 10)
            sys.exit(0)

            """

            """
        if 0:
            regressor = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.001, n_estimators=30000,objective='reg:linear', subsample=0.89, colsample_bytree=0.86, seed=0, reg_lambda=0.92 , reg_alpha=0.27, gamma=0.143, missing = np.NaN )
            regressor.fit(train_set_x_tot, train_set_y_tot ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
            # 2.36

        if 0:
            regressor = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.001, n_estimators=13000,objective='reg:linear', subsample=0.89, colsample_bytree=0.86, seed=0, reg_lambda=0.92 , reg_alpha=0.27, gamma=0.143, missing = np.NaN )
            regressor.fit(X, y ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
            #regressor.fit(train_set_x_tot, train_set_y_tot ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
            print("ok")
            if 1:
                #f = gzip.open("fivemn_reg_2.49.gz","wb")
                f = gzip.open("fivemn_leafwet_reg_2.40.gz","wb")
                cPickle.dump( regressor , f,cPickle.HIGHEST_PROTOCOL)
                f.close()         

        if 0:
            regressor = KNeighborsRegressor(weights='uniform', algorithm='auto', leaf_size=10, p=1, n_jobs=1 )            
            regressor.fit(train_set_x_tot, train_set_y_tot)
            #regressor.fit(X, y)

        if 1:
            scaler = StandardScaler()
            #scaler.fit(train_set_x_tot)
            scaler.fit(X)

        if 1:
            regressor = SVR(C=8, kernel='rbf', gamma = 'auto', )
            #regressor.fit(X, y ,eval_metric='rmse',eval_set = eval_set,early_stopping_rounds=50)
            train_set_x_tot = scaler.transform(train_set_x_tot)
            valid_set_x = scaler.transform(valid_set_x)
            X_pred = scaler.transform(X_pred)
            #regressor.fit(scaler.transform(X), y)
            regressor.fit(train_set_x_tot, train_set_y_tot)
            pred_fivemn_valid = regressor.predict(valid_set_x)
            f = gzip.open("pred_fivemn_valid.gz","wb")
            cPickle.dump( (pred_fivemn_valid, valid_set_y, fivemn_model_train, valid_indices) , f,cPickle.HIGHEST_PROTOCOL)
            f.close()         
            
            # 3.81
            print("ok")
            if 1:
                f = gzip.open("fivemn_SVC_2.58.gz","wb")
                cPickle.dump( regressor , f,cPickle.HIGHEST_PROTOCOL)
                f.close()         


        if 1:        
            y_hat = regressor.predict(valid_set_x)
            rmse = np.sqrt(mean_squared_error(valid_set_y, y_hat))
            print("rmse=",rmse)
            #sys.exit(0)


        if 0:        
            y_hat = regressor.predict(valid_set_x)
            rmse = np.sqrt(mean_squared_error(valid_set_y, y_hat))
            print("rmse=",rmse)
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.scatter(valid_set_y, y_hat)
            
            plt.xlabel('actual', fontsize=20)
            plt.ylabel('predicted', fontsize=20)
            plt.plot(np.linspace(0, 35), np.linspace(0, 35), label="$y=x$")
            
            plt.xlim(0, 35)
            plt.ylim(0, 35)
            plt.legend(loc='upper left', fontsize=20)
            plt.show()
            
            
            fig, ax = plt.subplots(figsize=(16, 4))
            err = valid_set_y - y_hat
            
            print(fivemn_model[fivemn_model['is_train'] == True].reset_index().columns)
            valid_dates = fivemn_model[fivemn_model['is_train'] == True].reset_index()['index'][valid_indices]
            ax.plot_date( valid_dates, err, c='r', ms=3)
            ax.set_title('residuals on test data (each)', fontsize=20)
            ax.set_ylabel('error')
            plt.show()
                        
            fig, ax = plt.subplots(figsize=(16, 4))
            plt.hist(err, bins=20, normed=True)
            plt.title('residuals on test data (distribution)', fontsize=20)
            plt.xlim(-20, 20)

            plt.show()
            sys.exit(0)

        
        if 0:
            f = gzip.open("fivemn_leafwet_reg_2.40.gz","r")
            regressor = pickle.load(f)
            f.close()         

        print(fivemn_model.loc[fivemn_model['is_train'] == False, columns_selection ].isnull().sum(axis=0)) 
        #X_pred = fivemn_model.loc[fivemn_model['is_train'] == False, columns_selection ].as_matrix().astype(np.float32)
        y_pred = regressor.predict(X_pred)
        #print(y_pred)
        fivemn_model.loc[fivemn_model['is_train'] == False,'yield_fivemn' ] = y_pred         
            

        if 1:
            f = gzip.open("pred_agadir_valid.gz","r")
            (pred_agadir_valid, valid_set_y, agadir_model_train, valid_indices) = pickle.load(f)
            f.close()        
            pred_agadir_valid[pred_agadir_valid < 0] = 0.0 
            agadir_model_train.loc[valid_indices,'yield_agadir_predicted'] = pred_agadir_valid

            y_hat = pred_agadir_valid
            rmse = np.sqrt(mean_squared_error(valid_set_y, y_hat))
            print("rmse1=",rmse)
                        
            f = gzip.open("pred_agadir2_valid.gz","r")
            (pred_agadir2_valid, valid_set_y2, agadir_model2_train, valid_indices2) = pickle.load(f)
            f.close()         
            pred_agadir2_valid[pred_agadir2_valid < 0] = 0.0
            
            agadir_model2_train.loc[valid_indices2,'yield_agadir2_predicted'] = pred_agadir2_valid
            
            sub_form = lin_model[lin_model['is_train'] == True]
            sub_form = sub_form[ (sub_form.index.day <= 4) & (sub_form.index.day >= 1)]
            
            sub_form = sub_form.join(agadir_model_train.loc[valid_indices,'yield_agadir_predicted'],how='left')
            
            sub_form.loc[np.logical_not(sub_form['yield_agadir_predicted'].isnull()), 'yield_predicted'] = sub_form.loc[np.logical_not(sub_form['yield_agadir_predicted'].isnull()),'yield_agadir_predicted']
            
            y_hat = sub_form.loc[np.logical_not(sub_form['yield_agadir_predicted'].isnull()), 'yield_predicted'].as_matrix()
            y = sub_form.loc[np.logical_not(sub_form['yield_agadir_predicted'].isnull()),'yield'].as_matrix()

            rmse = np.sqrt(mean_squared_error(y, y_hat))
            print("rmse2=",rmse)
            
            #sub_form['yield_predicted'] = sub_form['yield_agadir_predicted']
            print("Missing predicts for agadir:", np.sum(sub_form['yield_agadir_predicted'].isnull()))
            #sub_form.loc[sub_form['yield_agadir_predicted'].isnull(), 'yield_predicted'] = 0.0 #sub_form['yield_lin']
            
            print("Join agadir2")
            sub_form = sub_form.join(agadir_model2_train.loc[valid_indices2,'yield_agadir2_predicted'], how='left')
            
            print("Predicts from agadir2:", np.sum(np.logical_not(sub_form['yield_agadir2_predicted'].isnull())))
            sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()), 'yield_predicted' ] = sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()),'yield_agadir2_predicted']
            
            
            y_hat = sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()),'yield_predicted'].as_matrix()
            y = sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()),'yield'].as_matrix()

            rmse = np.sqrt(mean_squared_error(y, y_hat))
            print("rmse3=",rmse)

            y_hat = sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()),'yield_agadir_predicted'].as_matrix()
            y = sub_form.loc[np.logical_not(sub_form['yield_agadir2_predicted'].isnull()),'yield'].as_matrix()

            rmse = np.sqrt(mean_squared_error(y, y_hat))
            print("rmse3b=",rmse)

            y_hat = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield_predicted'].as_matrix()
            y = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield'].as_matrix()

            rmse = np.sqrt(mean_squared_error(y, y_hat))
            print("rmse4=",rmse)


            print("Join fivemn")
            f = gzip.open("pred_fivemn_valid.gz","r")
            (pred_fivemn_valid, valid_set_y5, fivemn_model_train, valid_indices5) = pickle.load(f)
            f.close()         
            pred_fivemn_valid[pred_fivemn_valid < 0] = 0.0
            
            fivemn_model_train.loc[valid_indices5,'yield_fivemn_predicted'] = pred_fivemn_valid
            
            sub_form = sub_form.join(fivemn_model_train.loc[valid_indices5,'yield_fivemn_predicted'], how='left')
            
            print("Predicts from fivemn:", np.sum(np.logical_not(sub_form['yield_fivemn_predicted'].isnull())))
            sub_form.loc[np.logical_not(sub_form['yield_fivemn_predicted'].isnull()), 'yield_predicted' ] = sub_form.loc[np.logical_not(sub_form['yield_fivemn_predicted'].isnull()),'yield_fivemn_predicted']
            
            
            y_hat = sub_form.loc[np.logical_not(sub_form['yield_fivemn_predicted'].isnull()),'yield_predicted'].as_matrix()
            y = sub_form.loc[np.logical_not(sub_form['yield_fivemn_predicted'].isnull()),'yield'].as_matrix()

            rmse = np.sqrt(mean_squared_error(y, y_hat))
            print("rmse5=",rmse)

            y_hat = sub_form.loc[np.logical_not(sub_form['yield_fivemn_predicted'].isnull()),'yield_agadir_predicted'].as_matrix()
            y = sub_form.loc[np.logical_not(sub_form['yield_fivemn_predicted'].isnull()),'yield'].as_matrix()

            rmse = np.sqrt(mean_squared_error(y, y_hat))
            print("rmse5b=",rmse)

            y_hat = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield_predicted'].as_matrix()
            y = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield'].as_matrix()

            rmse = np.sqrt(mean_squared_error(y, y_hat))
            print("rmse6=",rmse)


            f = gzip.open("subform_debug.gz","wb")
            cPickle.dump( sub_form  , f,cPickle.HIGHEST_PROTOCOL)
            f.close()         

            print(sub_form.isnull().sum(axis=0))
            print(sub_form.describe())

            sub_form.loc[sub_form['yield_predicted'].isnull(), 'yield_predicted'] = 0.0
            
            
            if 1:        
                y_hat = sub_form['yield_predicted'].as_matrix()
                rmse = np.sqrt(mean_squared_error(sub_form['yield'].as_matrix(), y_hat))
                print("rmse=",rmse)
                #sys.exit(0)
                
            
            if 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                plt.scatter(sub_form['yield'].as_matrix(), y_hat)
                
                plt.xlabel('actual', fontsize=20)
                plt.ylabel('predicted', fontsize=20)
                plt.plot(np.linspace(0, 35), np.linspace(0, 35), label="$y=x$")
                
                plt.xlim(0, 35)
                plt.ylim(0, 35)
                plt.legend(loc='upper left', fontsize=20)
                plt.show()
                
                
                fig, ax = plt.subplots(figsize=(16, 4))
                err = sub_form['yield'].as_matrix() - y_hat
                
                valid_dates = sub_form.reset_index()['index']
                ax.plot_date( valid_dates, err, c='r', ms=3)
                ax.set_title('residuals on test data (each)', fontsize=20)
                ax.set_ylabel('error')
                plt.show()
                            
                fig, ax = plt.subplots(figsize=(16, 4))
                plt.hist(err, bins=20, normed=True)
                plt.title('residuals on test data (distribution)', fontsize=20)
                plt.xlim(-20, 20)
    
                plt.show()
                sys.exit(0)
            
            #sys.exit(0)

        
        if 1:
            sub_form = lin_model[lin_model['is_train'] == False]
            print("There are", sub_form.shape, "values to submit")
            sub_form = sub_form.join(agadir_model.loc[agadir_model['is_train'] == False,'yield_agadir'])

            print("Missing predicts for agadir:", np.sum(sub_form['yield_agadir'].isnull()))
            
            print("After agadir join", sub_form.shape, "values to submit")

            sub_form = sub_form.join(agadir_model2.loc[agadir_model2['is_train'] == False,'yield_agadir2'])

            print("Missing predicts for agadir2:", np.sum(sub_form['yield_agadir2'].isnull()))
            print("After agadir2 join", sub_form.shape, "values to submit")
            
            sub_form = sub_form.join(fivemn_model.loc[fivemn_model['is_train'] == False, 'yield_fivemn'], how='left')

            print("Missing predicts for fivemn:", np.sum(sub_form['yield_fivemn'].isnull()))
            print("After 5mn join", sub_form.shape, "values to submit")

            print(sub_form['yield_fivemn'].isnull().sum(axis=0))            
                                                 
            sub_form['yield'] = sub_form['yield_fivemn']
            
            sub_form.loc[sub_form['yield_fivemn'].isnull(), 'yield'] = sub_form['yield_agadir2']
            
            sub_form.loc[sub_form['yield_agadir2'].isnull() & sub_form['yield_fivemn'].isnull() , 'yield'] = sub_form['yield_agadir']
            
            sub_form.loc[sub_form['yield_agadir'].isnull() & sub_form['yield_fivemn'].isnull() & sub_form['yield_agadir2'].isnull() , 'yield'] = sub_form['yield_lin']

            print(( sub_form['yield_fivemn'].isnull() & sub_form['yield_agadir'].isnull() & sub_form['yield_agadir2'].isnull() ).sum(axis=0))
            
            sub_form.loc[sub_form['yield']< 0.0, 'yield'] = 0.0
            sub_form['yield'].to_csv('fognet_sub_5mn.csv')

    
    exit(0)    
    train = pd.read_csv("rossman_train.csv", parse_dates=[2], nrows=nrows2load)
    test = pd.read_csv("rossman_test.csv", parse_dates=[3], nrows=nrows2load)
    store = pd.read_csv("rossman_store.csv")
    store_states = pd.read_csv("rossman_store_states.csv")
    
    train['Id'] = 0
    test['Sales'] = -1
    test['Customers'] = -1
    train = train.append(test)
    
    
    print("Assume store open, if not provided")
    train.fillna(1, inplace=True)
    #test.fillna(1, inplace=True)
    
    print("Join with store")
    train = pd.merge(train, store, on='Store')
    #test = pd.merge(test, store, on='Store')
    print("Join with store states")
    train = pd.merge(train, store_states, on='Store')
    
    features = []
    
    print("augment features")
    train = build_features(features, train)
    #build_features([], test)
    print(features)
    print train.columns
    
    print("Consider only open stores for training. Closed stores wont count into the score.")
    print "Test set size is",train[ train["Id"] > 0 ].shape
    test = train[ train["Id"] > 0 ].copy()
    print  test.shape
    train = train[train["Open"] != 0]
    print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
    #train = train[train["Sales"] > 0]
    train = train[train["Sales"] != 0]
    

    
    #test = train[ train["Id"] > 0 ]
    train = train[ train["Id"] == 0 ]
    
    
    
    col_list = list(features)
    col_list.append('Sales')
    train = train[col_list]
    col_list = list(features)
    col_list.append('Id')
    test = test[col_list]

    print "save to hdf"
    print train.columns
    train.to_hdf('rossman_train4.hdf','train',mode='w',complib='blosc')            
    test.to_hdf('rossman_test4.hdf','test',mode='w',complib='blosc')

else:
    if 0:
        train  = pd.read_hdf('rossman_train4.hdf','train')
        test  = pd.read_hdf('rossman_test4.hdf','test')
        features = list(train.columns)
        features.remove("Sales")
    else:
        store1_open  = pd.read_hdf('store1_open.hdf','all')
        print store1_open.columns 
        store1_open['weekofyear'] = store1_open.Date.dt.weekofyear
           
        f = gzip.open("lrs/pca_scalers%i.pkl.gz" % 0,"rb")
        (pca,scaler,n_vars,l_params) = pickle.load(f)
        f.close()
        l_params.append('Predict_LogNorm2')
        test = store1_open[store1_open['Id'] > 0 ].copy()
        train = store1_open[store1_open['Open'] != 0]
        train = train[train['Sales'] != 0]
        train = train[train['Id'] == 0]
        features = list(l_params)
        
        col_list = list(l_params)
        col_list.append('Sales')
        col_list.append('MaxSales')
        col_list.append('MaxLogSales')
        col_list.append('LogSales_Norm')
        col_list.append('LogSales_Norm2')
        col_list.append('Store')
        col_list.append('weekofyear')
        train = train[col_list]
        col_list.append('Id')
        test = test[col_list]
        



print('training data processed')


if 1:
    depth = 10
    n_estimators = 10000
    #eta = 0.012
    eta = 0.01
    #eta = 0.005
    #eta = 0.0025
    subsample = 1
    colsample_bytree = 0.5
    #colsample_bytree = 0.3
    gamma = 0
    #xgbtree = xgb.XGBRegressor(max_depth=depth,n_estimators=n_estimators,silent=False,objective='reg:linear', learning_rate = eta, subsample = subsample, seed = rnd_seed, colsample_bytree = colsample_bytree, gamma=gamma)

params = {"objective": "reg:linear",
          "eta": eta,
          "max_depth": depth,
          "subsample": subsample,
          "colsample_bytree": colsample_bytree,
          "silent": 0,
          "seed": 1301
          }
num_boost_round = n_estimators + 1


print("Train a XGBoost model")

stores_test = np.unique(test['Store'])
stores_train = np.unique(train['Store'])
stores_only_train = list(set(stores_train) - set(stores_test))
print len(stores_only_train)

#X_train, X_valid = train_test_split(train, test_size=0.012, random_state = 1302)
train_filter = np.logical_not(train['Store'].isin(stores_only_train) & (train['weekofyear'] == 30) & ( train['year_2015'] == 1 ) )
valid_filter = (train['Store'].isin(stores_only_train) & (train['weekofyear'] == 30) & ( train['year_2015'] == 1 ) )
X_train = train[train_filter]
X_valid = train[valid_filter]
print(X_train.shape,X_valid.shape)
X_train_maxsales = X_train['MaxSales'].copy()
X_valid_maxsales = X_valid['MaxSales'].copy()
y_train = X_train.Sales / X_train_maxsales
y_valid = X_valid.Sales / X_valid_maxsales
y_train = np.log1p(y_train)
y_valid = np.log1p(y_valid)

X_train = X_train[features]
X_valid = X_valid[features]

    

margin_test = None
margin_valid = None
margin_train = None

if 1:
    dtrain = xgb.DMatrix(X_train[features].as_matrix(), label=y_train)
    dvalid = xgb.DMatrix(X_valid[features].as_matrix(), y_valid)
    dtest = xgb.DMatrix(test[features].as_matrix())
    
    if 1:
        if 0:
            f = gzip.open("xgbross_6_1.000000_0.500000_0.010000_0.194069_0.226666_10000.gz","rb")
            start_tree = pickle.load(f)
            f.close()
            margin_train = start_tree.predict(dtrain, output_margin = True)
            margin_valid = start_tree.predict(dvalid, output_margin = True)
            margin_test = start_tree.predict(dtest, output_margin = True)
            
            f = gzip.open("margins_6_1.000000_0.500000_0.010000_0.194069_0.226666_10000.gz","wb")
            cPickle.dump(( margin_train, margin_valid, margin_test ), f,cPickle.HIGHEST_PROTOCOL)
            f.close() 
        else:
            f = gzip.open("margins_6_1.000000_0.500000_0.010000_0.194069_0.226666_10000.gz","rb")
            ( margin_train, margin_valid, margin_test ) = pickle.load(f)
            f.close()
    
        dtrain.set_base_margin(margin_train)
        dvalid.set_base_margin(margin_valid)
        dtest.set_base_margin(margin_test)
    
        if 0:
            if 0:
                f = gzip.open("xgbross_7_1.000000_0.500000_0.010000_0.133245_0.175956_10000.gz","rb")
                start_tree = pickle.load(f)
                f.close()
                margin_train = start_tree.predict(dtrain, output_margin = True)
                margin_valid = start_tree.predict(dvalid, output_margin = True)
                margin_test = start_tree.predict(dtest, output_margin = True)
                
                f = gzip.open("margins_7_1.000000_0.500000_0.010000_0.133245_0.175956_10000.gz","wb")
                cPickle.dump(( margin_train, margin_valid, margin_test ), f,cPickle.HIGHEST_PROTOCOL)
                f.close() 
            else:
                f = gzip.open("margins_7_1.000000_0.500000_0.010000_0.133245_0.175956_10000.gz","rb")
                ( margin_train, margin_valid, margin_test ) = pickle.load(f)
                f.close()
        
            
            dtrain.set_base_margin(margin_train)
            dvalid.set_base_margin(margin_valid)
            dtest.set_base_margin(margin_test)

            if 0:
                if 0:
                    f = gzip.open("xgbross_9_1.000000_0.500000_0.010000_0.115335_2500.gz","rb")
                    start_tree = pickle.load(f)
                    f.close()
                    margin_train = start_tree.predict(dtrain, output_margin = True)
                    margin_valid = start_tree.predict(dvalid, output_margin = True)
                    margin_test = start_tree.predict(dtest, output_margin = True)
                    
                    f = gzip.open("margins_9_1.000000_0.500000_0.010000_0.115335_2500.gz","wb")
                    cPickle.dump(( margin_train, margin_valid, margin_test ), f,cPickle.HIGHEST_PROTOCOL)
                    f.close() 
                else:
                    f = gzip.open("margins_9_1.000000_0.500000_0.010000_0.115335_2500.gz","rb")
                    ( margin_train, margin_valid, margin_test ) = pickle.load(f)
                    f.close()
            
                
                dtrain.set_base_margin(margin_train)
                dvalid.set_base_margin(margin_valid)
                dtest.set_base_margin(margin_test)

                if 1:
                    if 0:
                        f = gzip.open("xgbross_10_1.000000_0.300000_0.005000_0.112649_2500.gz","rb")
                        start_tree = pickle.load(f)
                        f.close()
                        margin_train = start_tree.predict(dtrain, output_margin = True)
                        margin_valid = start_tree.predict(dvalid, output_margin = True)
                        margin_test = start_tree.predict(dtest, output_margin = True)
                        
                        f = gzip.open("margins_10_1.000000_0.300000_0.005000_0.112649_2500.gz","wb")
                        cPickle.dump(( margin_train, margin_valid, margin_test ), f,cPickle.HIGHEST_PROTOCOL)
                        f.close() 
                    else:
                        f = gzip.open("margins_10_1.000000_0.300000_0.005000_0.112649_2500.gz","rb")
                        ( margin_train, margin_valid, margin_test ) = pickle.load(f)
                        f.close()
                
                    
                    dtrain.set_base_margin(margin_train)
                    dvalid.set_base_margin(margin_valid)
                    dtest.set_base_margin(margin_test)

                    if 1:
                        if 1:
                            f = gzip.open("xgbross_12_1.000000_0.300000_0.005000_0.109324_2500.gz","rb")
                            start_tree = pickle.load(f)
                            f.close()
                            margin_train = start_tree.predict(dtrain, output_margin = True)
                            margin_valid = start_tree.predict(dvalid, output_margin = True)
                            margin_test = start_tree.predict(dtest, output_margin = True)
                            
                            f = gzip.open("margins_12_1.000000_0.300000_0.005000_0.109324_2500.gz","wb")
                            cPickle.dump(( margin_train, margin_valid, margin_test ), f,cPickle.HIGHEST_PROTOCOL)
                            f.close() 
                        else:
                            f = gzip.open("margins_12_1.000000_0.300000_0.005000_0.109324_2500.gz","rb")
                            ( margin_train, margin_valid, margin_test ) = pickle.load(f)
                            f.close()
                    
                        
                        dtrain.set_base_margin(margin_train)
                        dvalid.set_base_margin(margin_valid)
                        dtest.set_base_margin(margin_test)

                        if 0:
                            if 1:
                                f = gzip.open("xgbross_15_1.000000_0.500000_0.005000_0.105805_1500.gz","rb")
                                start_tree = pickle.load(f)
                                f.close()
                                margin_train = start_tree.predict(dtrain, output_margin = True)
                                margin_valid = start_tree.predict(dvalid, output_margin = True)
                                margin_test = start_tree.predict(dtest, output_margin = True)
                                
                                f = gzip.open("margins_15_1.000000_0.500000_0.005000_0.105805_1500.gz","wb")
                                cPickle.dump(( margin_train, margin_valid, margin_test ), f,cPickle.HIGHEST_PROTOCOL)
                                f.close() 
                            else:
                                f = gzip.open("margins_15_1.000000_0.500000_0.005000_0.105805_1500.gz","rb")
                                ( margin_train, margin_valid, margin_test ) = pickle.load(f)
                                f.close()
                        
                            
                            dtrain.set_base_margin(margin_train)
                            dvalid.set_base_margin(margin_valid)
                            dtest.set_base_margin(margin_test)
        

if 1:        
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    try:
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=500, \
                        feval=rmspe_xg, verbose_eval=True)
    except KeyboardInterrupt:
        print "Interrupted!"
        
    print("Validating")
    yhat = gbm.predict(dvalid) #xgb.DMatrix(X_valid[features].as_matrix()))
    yhat = np.expm1(yhat)
    yhat = yhat * X_valid_maxsales
    
    error = rmspe(train[valid_filter]['Sales'].as_matrix(), yhat)
    print('RMSPE: {:.6f}'.format(error))

    print("Validating train")
    yhat = gbm.predict(dtrain) #xgb.DMatrix(X_valid[features].as_matrix()))
    yhat = np.expm1(yhat)
    yhat = yhat * X_train_maxsales
    
    errort = rmspe(train[train_filter]['Sales'].values, yhat)
    print('RMSPE t: {:.6f}'.format(errort))
    
    
    print "Save model"
    f = gzip.open("xgbross_%i_%f_%f_%f_%f_%f_%i.gz" % (depth,subsample,colsample_bytree,eta,error,errort,n_estimators), 'wb')
    cPickle.dump(gbm,f, cPickle.HIGHEST_PROTOCOL)
    f.close()
        
else:
    f = gzip.open("xgbross_6_1.000000_0.500000_0.010000_0.124229_5000.gz","rb")
    gbm = pickle.load(f)
    f.close()
    



print("Make predictions on the test set")

testcsv = pd.read_csv("rossman_test.csv", parse_dates=[3], nrows=None)

test = pd.merge(test,testcsv[["Id","Date"]], on='Id')

    
    
#dtest = xgb.DMatrix(test[features].as_matrix())
test_probs = gbm.predict(dtest)

test_probs = np.expm1(test_probs)
test_probs = test_probs * test['MaxSales'].values

# Make Submission
#result = pd.DataFrame({"Id": test["Id"],  'Sales': np.expm1(test_probs)})
result = pd.DataFrame({"Id": test["Id"],  'Sales': (test_probs)})
result.to_csv("xgboost_10_submission.csv", index=False)

# XGB feature importances
# Based on https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code

ceate_feature_map(features)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png',bbox_inches='tight',pad_inches=1)
