## Pong Yin Hui (23054197), Cynthia Rojas (14101049), Leo Gussack (23976155), Joann Lin (16006991)
## CIS 9650 Final Project
################################################

import pandas as pd
import numpy as np
from datetime import datetime
import geoplotlib as gp
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt



#################### DATA CLEANSING

## Read csv files, convert object to datetime float

dateparse = lambda x: datetime.strptime(x, '%H:%M:%S').time()
NYPD = pd.read_csv('NYPD Complaint.csv', parse_dates=['CMPLNT_FR_TM'], date_parser=dateparse)
wifi = pd.read_csv("NYC_Wi-Fi_Hotspot_Locations_Final.csv")
streetlight = pd.read_csv("Streetlight.csv")

## Drop and rename columns

NYPD = NYPD[['CMPLNT_NUM','CMPLNT_FR_TM',  'RPT_DT', 'LAW_CAT_CD', 'BORO_NM', 'Latitude', 'Longitude']]
NYPD.columns = ['compliant_num', 'time', 'date', 'category', 'boro_c', 'lat_c', 'lon_c']
# print(NYPD.head(5))

wifi = wifi[['OBJECTID', 'Borough Name', 'Latitude', 'Longitude']]
wifi.columns = ['obj_id', 'boro_w', 'lat_w', 'lon_w']
# print(wifi.head(5))

streetlight = streetlight[['Id', 'Pole Class', 'Borough', 'Latitude', 'Longitude']]
streetlight.columns = ['pole_id', 'pole', 'boro_s', 'lat_s', 'lon_s']
# print(streetlight.head(5))

## Filtering Pole Class column out for City

SL_filter = streetlight[streetlight['pole'] == 'CITY']

# Filtering out 2019

NYPD['date'] =  pd.to_datetime(NYPD['date']) 
NYPD_filter = NYPD[NYPD['date'].dt.strftime('%Y') == '2019']

# Adding a new column called Day/Night and assigning value based on time

start_time = datetime.strptime('6:00:00', '%H:%M:%S').time()
end_time = datetime.strptime('18:00:00', '%H:%M:%S').time()
NYPD_filter.loc[(NYPD_filter['time'] > start_time) & (NYPD_filter['time'] < end_time),'Day/Night'] = 'day'
NYPD_filter.loc[(NYPD_filter['time'] <= start_time) | (NYPD_filter['time'] >= end_time),'Day/Night'] = 'night'

## Dropping all empty cells in column Latitude and Longitude

NYPD_filter['lat_c'].replace('', np.nan, inplace = True)
NYPD_filter.dropna(subset = ['lat_c'], inplace = True)
NYPD_filter['lat_c'].replace('', np.nan, inplace = True)
NYPD_filter.dropna(subset = ['lat_c'], inplace = True)

## Dropping all empty cell in borough column

NYPD_filter['boro_c'].replace('', np.nan, inplace = True)
NYPD_filter.dropna(subset = ['boro_c'], inplace = True)

## Filtering by Borough in NYPD Complaint file

NYPD_BRONX = NYPD_filter[NYPD_filter['boro_c'] == 'BRONX']
NYPD_BROOKLYN = NYPD_filter[NYPD_filter['boro_c'] == 'BROOKLYN']
NYPD_MANHATTAN = NYPD_filter[NYPD_filter['boro_c'] == 'MANHATTAN']
NYPD_QUEENS = NYPD_filter[NYPD_filter['boro_c'] == 'QUEENS']
NYPD_STATENISLAND = NYPD_filter[NYPD_filter['boro_c'] == 'STATEN ISLAND']

## Filtering to felonies in NYPD Complaint file

NYPD_BNX_FEL = NYPD_BRONX[NYPD_BRONX['category'] == 'FELONY']
NYPD_BKN_FEL = NYPD_BROOKLYN[NYPD_BROOKLYN['category'] == 'FELONY']
NYPD_MNH_FEL = NYPD_MANHATTAN[NYPD_MANHATTAN['category'] == 'FELONY']
NYPD_QNS_FEL = NYPD_QUEENS[NYPD_QUEENS['category'] == 'FELONY']
NYPD_STN_FEL = NYPD_STATENISLAND[NYPD_STATENISLAND['category'] == 'FELONY']

## Filtering by Borough in Streetlight file

SL_BRONX = SL_filter[SL_filter['boro_s'] == 'Bronx']
SL_BROOKLYN = SL_filter[SL_filter['boro_s'] == 'Brooklyn']
SL_MANHATTAN = SL_filter[SL_filter['boro_s'] == 'Manhattan']
SL_QUEENS = SL_filter[SL_filter['boro_s'] == 'Queens']
SL_STATENISLAND = SL_filter[SL_filter['boro_s'] == 'Staten_Island']

## Filtering by Borough in Wifi file

wifi_BRONX = wifi[wifi['boro_w'] == 'Bronx']
wifi_BROOKLYN = wifi[wifi['boro_w'] == 'Brooklyn']
wifi_MANHATTAN = wifi[wifi['boro_w'] == 'Manhattan']
wifi_QUEENS = wifi[wifi['boro_w'] == 'Queens']
wifi_STATENISLAND = wifi[wifi['boro_w'] == 'Staten Island']



#################### MAP PLOTS

#1. Streetlights = green, wifi = blue, and crime locations = red (general - everything on the map)

alphaCrime = 20

streetLightColor = [103,191,92]
wifiColor = [0,122,255]
crimeColor = [237,102,93, alphaCrime]
gp.clear()
plotPointSize = 2

gp.set_window_size(800,800)
gp.tiles_provider('darkmatter')

#Street light plot
geoPoltData = streetlight[["lat_s","lon_s"]].rename(columns = {'lat_s':'lat', 'lon_s' : 'lon'}, inplace = False)
gp.dot(geoPoltData,point_size=plotPointSize, color = streetLightColor )

#Wifi plot
geoPoltData = wifi[["lat_w","lon_w"]].rename(columns = {'lat_w':'lat', 'lon_w' : 'lon'}, inplace = False)
gp.dot(geoPoltData,point_size=plotPointSize, color = wifiColor)

#Crime plot
geoPoltData = NYPD_filter[["lat_c","lon_c"]].rename(columns = {'lat_c':'lat', 'lon_c' : 'lon'}, inplace = False)
gp.dot(geoPoltData,point_size=plotPointSize, color = crimeColor)

gp.show()

#2. Only streetlights, wifi, and violent crime locations (no differentiation between night/day) - felonies

gp.clear()
plotPointSize = 2

gp.set_window_size(800,800)
gp.tiles_provider('darkmatter')

#Street light plot
geoPoltData = streetlight[["lat_s","lon_s"]].rename(columns = {'lat_s':'lat', 'lon_s' : 'lon'}, inplace = False)
gp.dot(geoPoltData,point_size=plotPointSize, color =streetLightColor)

#Wifi plot
geoPoltData = wifi[["lat_w","lon_w"]].rename(columns = {'lat_w':'lat', 'lon_w' : 'lon'}, inplace = False)
gp.dot(geoPoltData,point_size=plotPointSize, color =wifiColor)

#Crime plot felony
geoPoltData = NYPD_filter[NYPD_filter.category == "FELONY"][["lat_c","lon_c"]].rename(columns = {'lat_c':'lat', 'lon_c' : 'lon'}, inplace = False)
gp.dot(geoPoltData,point_size=plotPointSize, color =crimeColor)

gp.show()
#gp.savefig('map2')

#3. Only streetlights and night time felonies locations

#Street lights
gp.clear()

gp.set_window_size(800,800)
gp.tiles_provider('darkmatter')
geoPoltData = streetlight[["lat_s","lon_s"]].rename(columns = {'lat_s':'lat', 'lon_s' : 'lon'}, inplace = False)
gp.dot(geoPoltData,point_size=plotPointSize, color =streetLightColor)

#Crime plot felony and night
geoPoltData = NYPD_filter[(NYPD_filter.category == "FELONY") & (NYPD_filter["Day/Night"] == "night")][["lat_c","lon_c"]].rename(columns = {'lat_c':'lat', 'lon_c' : 'lon'}, inplace = False)
gp.dot(geoPoltData,point_size=plotPointSize, color =crimeColor)

gp.show()

#4. Only wifi and day time felonies locations

gp.clear()

gp.set_window_size(800,800)
gp.tiles_provider('darkmatter')

#Wifi plot
geoPoltData = wifi[["lat_w","lon_w"]].rename(columns = {'lat_w':'lat', 'lon_w' : 'lon'}, inplace = False)
gp.dot(geoPoltData,point_size=plotPointSize, color =wifiColor)

#Crime plot felony and day
geoPoltData = NYPD_filter[(NYPD_filter.category == "FELONY") & (NYPD_filter["Day/Night"] == "day")][["lat_c","lon_c"]].rename(columns = {'lat_c':'lat', 'lon_c' : 'lon'}, inplace = False)
gp.dot(geoPoltData,point_size=plotPointSize, color =crimeColor)

gp.show()



#################### CALCULATIONS / ANALYSIS

# from haversine import haversine  ## didn't work - pip issue

## Create function - finds distance between two sets of longitutde and latitude

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956 # Radius of earth in kilometers. Use 3956 for miles.  6371 for km.
    return c * r


###### --------------------------------------------------------------------------- QUEENS

## Create list of list of streetlight locations

streetlights_qns = []
for index, row in SL_QUEENS.iterrows():
    streetlights_qns.append([row['lon_s'], row['lat_s']])
    
## Create function - finds closest streetlight to each crime and returns distance            

def min_dist_s_qns(lon1, lat1):
    min_d = 999999999999999
    for loc in streetlights_qns:
        dist = haversine(lon1, lat1, loc[0], loc[1])
        if dist < min_d:
            min_d = dist
    print(lon1, lat1, min_d)
    return min_d

## Create list of list of wifi locations

wifi_qns = []
for index, row in wifi_QUEENS.iterrows():
    wifi_qns.append([row['lon_w'], row['lat_w']])

## Create function - finds closest wifi to each crime and returns distance            

def min_dist_w_qns(lon1, lat1):
    min_d = 999999999999999
    for loc in wifi_qns:
        dist = haversine(lon1, lat1, loc[0], loc[1])
        if dist < min_d:
            min_d = dist
    print(lon1, lat1, min_d)
    return min_d

## Create additional column of closest streetlight and wifi to each crime

NYPD_QNS_FEL["min_streetlight"] = round(
    NYPD_QNS_FEL.apply(lambda x: min_dist_s_qns(x['lon_c'], x['lat_c']), axis=1),
    3)
NYPD_QNS_FEL["min_wifi"] = round(
    NYPD_QNS_FEL.apply(lambda x: min_dist_w_qns(x['lon_c'], x['lat_c']), axis=1),
    3)
print(NYPD_QNS_FEL)

## Min / max of min streetlight and wifi distances

qns_minSL = NYPD_QNS_FEL['min_streetlight'].min()
print("qns_minSL:", qns_minSL)
qns_maxSL = NYPD_QNS_FEL['min_streetlight'].max()
print("qns_maxSL:", qns_maxSL)
qns_minWifi = NYPD_QNS_FEL['min_wifi'].min()
print("qns_minWifi:", qns_minWifi)
qns_maxWifi = NYPD_QNS_FEL['min_wifi'].max()
print("qns_maxWifi:", qns_maxWifi)

## Graph - Queens -  Streetlights vs felonies (night), wifi vs felonies (day)

## Get crimes, filter to only 1 mile radius

qns_lst_sl_fel_night = []
for index, row in NYPD_QNS_FEL.iterrows():
    if row['Day/Night'] == 'night':
        qns_lst_sl_fel_night.append(row['min_streetlight'])
# print(qns_lst_sl_fel_night)

qns_flst_sl_fel_night = list(filter(lambda qns_lst_sl_fel_night:
                                    qns_lst_sl_fel_night <= 0.06, qns_lst_sl_fel_night))
print(len(qns_flst_sl_fel_night))

qns_lst_wifi_fel_day = []
for index, row in NYPD_QNS_FEL.iterrows():
    if row['Day/Night'] == 'day':
        qns_lst_wifi_fel_day.append(row['min_wifi'])
# print(qns_lst_wifi_fel_day)

qns_flst_wifi_fel_day = list(filter(lambda qns_lst_wifi_fel_day:
                                    qns_lst_wifi_fel_day <= 0.06, qns_lst_wifi_fel_day))
print(len(qns_flst_wifi_fel_day))

## Create x and y

x_qns_sl_fel = []
x_qns_sl_fel_use = []
y_qns_sl_fel = []
x_qns_wifi_fel = []
x_qns_wifi_fel_use = []
y_qns_wifi_fel = []

## Create segments

spacing = 0.01
for i in range(6):
    r1 = round(spacing * i, 3)
    r2 = round(r1 + spacing, 3)
    fdata = list(filter(lambda qns_flst_sl_fel_night:
                        r1 <= qns_flst_sl_fel_night < r2, qns_flst_sl_fel_night))
    x_qns_sl_fel.append(str(r1) + " - " + str(r2))
    x_qns_sl_fel_use.append(r2)
    y_qns_sl_fel.append(len(fdata))

spacing = 0.01
for i in range(6):
    r1 = round(spacing * i, 3)
    r2 = round(r1 + spacing, 3)
    fdata = list(filter(lambda qns_flst_wifi_fel_day:
                        r1 <= qns_flst_wifi_fel_day < r2, qns_flst_wifi_fel_day))
    x_qns_wifi_fel.append(str(r1) + " - " + str(r2))
    x_qns_wifi_fel_use.append(r2)
    y_qns_wifi_fel.append(len(fdata))

## Create graph

plt.suptitle('Queens', fontsize=20)

plt.plot(x_qns_sl_fel_use, y_qns_sl_fel, label = "felonies(night) vs. streetlight")
z = np.polyfit(x_qns_sl_fel_use, y_qns_sl_fel, 1)
p = np.poly1d(z)
print(x_qns_sl_fel_use)
print(p(x_qns_sl_fel_use))
plt.plot(x_qns_sl_fel_use, p(x_qns_sl_fel_use), c='#1f77b4', alpha=0.5, linestyle='dashed')

# slope1, intercept1 = np.polyfit(np.log(x_qns_sl_fel_use), np.log(p(x_qns_sl_fel_use)), 1)
slope1 = ((p(x_qns_sl_fel_use)[1] - p(x_qns_sl_fel_use)[0]) / (x_qns_sl_fel_use[1] - x_qns_sl_fel_use[0])) * x_qns_sl_fel_use[0]  ## manual calculation of slope since polyfit didn't work
print("Slope:", slope1)
plt.text(0.055, 640, round(slope1))

plt.plot(x_qns_wifi_fel_use, y_qns_wifi_fel, label = "felonies(day) vs. wifi")
z = np.polyfit(x_qns_wifi_fel_use, y_qns_wifi_fel, 1)
p = np.poly1d(z)
print(x_qns_wifi_fel_use)
print(p(x_qns_wifi_fel_use))
plt.plot(x_qns_wifi_fel_use, p(x_qns_wifi_fel_use), c='#FFA500', alpha=0.5, linestyle='dashed')

# slope2, intercept2 = np.polyfit(np.log(x_qns_wifi_fel_use), np.log(p(x_qns_wifi_fel_use)), 1)
slope2 = ((p(x_qns_sl_fel_use)[1] - p(x_qns_sl_fel_use)[0]) / (x_qns_sl_fel_use[1] - x_qns_sl_fel_use[0])) * x_qns_sl_fel_use[0]  ## manual calculation of slope since polyfit didn't work
print("Slope:", slope2)
plt.text(0.055, 330, round(slope2))

plt.xlabel('miles (0.01 = approx 53 ft)')
plt.ylabel('# of crimes')
plt.legend()
plt.show()

## Create dfs with values

df_qns_fel_sl = pd.DataFrame()
df_qns_fel_sl['min_streetlight_distance'] = x_qns_sl_fel
df_qns_fel_sl['n_crimes'] = y_qns_sl_fel
df_qns_fel_wifi = pd.DataFrame()
df_qns_fel_wifi['min_wifi_distance'] = x_qns_wifi_fel
df_qns_fel_wifi['n_crimes'] = y_qns_wifi_fel

print(df_qns_fel_sl)
print(df_qns_fel_wifi)

## Compute change - streetlights

df_qns_fel_sl['shifted'] = df_qns_fel_sl['n_crimes'].shift(1)
df_qns_fel_sl['percent_change'] = ((df_qns_fel_sl['n_crimes'] - df_qns_fel_sl['shifted']) /df_qns_fel_sl['shifted']) * 100
print(df_qns_fel_sl)
avg_df_qns_fel_sl = df_qns_fel_sl['percent_change'].mean()
# print("Queens - Felonies vs. Streetlights Average % Change:", avg_df_qns_fel_sl)

## Compute change - wifi

df_qns_fel_wifi['shifted'] = df_qns_fel_wifi['n_crimes'].shift(1)
df_qns_fel_wifi['percent_change'] = ((df_qns_fel_wifi['n_crimes'] - df_qns_fel_wifi['shifted']) /df_qns_fel_wifi['shifted']) * 100
print(df_qns_fel_wifi)
avg_df_qns_fel_wifi = df_qns_fel_wifi['percent_change'].mean()
# print("Queens - Felonies vs. Wifi Average % Change:", avg_df_qns_fel_wifi)


######  --------------------------------------------------------------------------- MANHATTAN

## Create list of list of streetlight locations

streetlights_mhn = []
for index, row in SL_MANHATTAN.iterrows():
    streetlights_mhn.append([row['lon_s'], row['lat_s']])
    
## Create function - finds closest streetlight to each crime and returns distance            

def min_dist_s_mhn(lon1, lat1):
    min_d = 999999999999999
    for loc in streetlights_mhn:
        dist = haversine(lon1, lat1, loc[0], loc[1])
        if dist < min_d:
            min_d = dist
    print(lon1, lat1, min_d)
    return min_d

## Create list of list of wifi locations

wifi_mhn = []
for index, row in wifi_MANHATTAN.iterrows():
    wifi_mhn.append([row['lon_w'], row['lat_w']])

## Create function - finds closest wifi to each crime and returns distance            

def min_dist_w_mhn(lon1, lat1):
    min_d = 999999999999999
    for loc in wifi_mhn:
        dist = haversine(lon1, lat1, loc[0], loc[1])
        if dist < min_d:
            min_d = dist
    print(lon1, lat1, min_d)
    return min_d

## Create additional column of closest streetlight and wifi to each crime

NYPD_MNH_FEL["min_streetlight"] = round(
    NYPD_MNH_FEL.apply(lambda x: min_dist_s_mhn(x['lon_c'], x['lat_c']), axis=1),
    3)
NYPD_MNH_FEL["min_wifi"] = round(
    NYPD_MNH_FEL.apply(lambda x: min_dist_w_mhn(x['lon_c'], x['lat_c']), axis=1),
    3)
print(NYPD_MNH_FEL)

## Min / max of min streetlight and wifi distances

mhn_minSL = NYPD_MNH_FEL['min_streetlight'].min()
print("mhn_minSL:", mhn_minSL)
mhn_maxSL = NYPD_MNH_FEL['min_streetlight'].max()
print("mhn_maxSL:", mhn_maxSL)
mhn_minWifi = NYPD_MNH_FEL['min_wifi'].min()
print("mhn_minWifi:", mhn_minWifi)
mhn_maxWifi = NYPD_MNH_FEL['min_wifi'].max()
print("mhn_maxWifi:", mhn_maxWifi)

## Graph - Manhattan -  Streetlights vs felonies (night), wifi vs felonies (day)

## Get crimes, filter to only 1 mile radius

mhn_lst_sl_fel_night = []
for index, row in NYPD_MNH_FEL.iterrows():
    if row['Day/Night'] == 'night':
        mhn_lst_sl_fel_night.append(row['min_streetlight'])
# print(mhn_lst_sl_fel_night)

mhn_flst_sl_fel_night = list(filter(lambda mhn_lst_sl_fel_night:
                                    mhn_lst_sl_fel_night <= 0.06, mhn_lst_sl_fel_night))
print(len(mhn_flst_sl_fel_night))

mhn_lst_wifi_fel_day = []
for index, row in NYPD_MNH_FEL.iterrows():
    if row['Day/Night'] == 'day':
        mhn_lst_wifi_fel_day.append(row['min_wifi'])
# print(mhn_lst_wifi_fel_day)

mhn_flst_wifi_fel_day = list(filter(lambda mhn_lst_wifi_fel_day:
                                    mhn_lst_wifi_fel_day <= 0.06, mhn_lst_wifi_fel_day))
print(len(mhn_flst_wifi_fel_day))

## Create x and y

x_mhn_sl_fel = []
x_mhn_sl_fel_use = []
y_mhn_sl_fel = []
x_mhn_wifi_fel = []
x_mhn_wifi_fel_use = []
y_mhn_wifi_fel = []

## Create segments

spacing = 0.01
for i in range(6):
    r1 = round(spacing * i, 3)
    r2 = round(r1 + spacing, 3)
    fdata = list(filter(lambda mhn_flst_sl_fel_night:
                        r1 <= mhn_flst_sl_fel_night < r2, mhn_flst_sl_fel_night))
    x_mhn_sl_fel.append(str(r1) + " - " + str(r2))
    x_mhn_sl_fel_use.append(r2)
    y_mhn_sl_fel.append(len(fdata))

spacing = 0.01
for i in range(6):
    r1 = round(spacing * i, 3)
    r2 = round(r1 + spacing, 3)
    fdata = list(filter(lambda mhn_flst_wifi_fel_day:
                        r1 <= mhn_flst_wifi_fel_day < r2, mhn_flst_wifi_fel_day))
    x_mhn_wifi_fel.append(str(r1) + " - " + str(r2))
    x_mhn_wifi_fel_use.append(r2)
    y_mhn_wifi_fel.append(len(fdata))

## Create graph

plt.suptitle('Manhattan', fontsize=20)

plt.plot(x_mhn_sl_fel_use, y_mhn_sl_fel, label = "felonies(night) vs. streetlight")
z = np.polyfit(x_mhn_sl_fel_use, y_mhn_sl_fel, 1)
p = np.poly1d(z)
print(x_mhn_sl_fel_use)
print(p(x_mhn_sl_fel_use))
plt.plot(x_mhn_sl_fel_use, p(x_mhn_sl_fel_use), c='#1f77b4', alpha=0.5, linestyle='dashed')

# slope1, intercept1 = np.polyfit(np.log(x_mhn_sl_fel_use), np.log(p(x_mhn_sl_fel_use)), 1)
slope1 = ((p(x_mhn_sl_fel_use)[1] - p(x_mhn_sl_fel_use)[0]) / (x_mhn_sl_fel_use[1] - x_mhn_sl_fel_use[0])) * x_mhn_sl_fel_use[0]  ## manual calculation of slope since polyfit didn't work
print("Slope:", slope1)
plt.text(0.055, 2170, round(slope1))

plt.plot(x_mhn_wifi_fel_use, y_mhn_wifi_fel, label = "felonies(day) vs. wifi")
z = np.polyfit(x_mhn_wifi_fel_use, y_mhn_wifi_fel, 1)
p = np.poly1d(z)
print(x_mhn_wifi_fel_use)
print(p(x_mhn_wifi_fel_use))
plt.plot(x_mhn_wifi_fel_use, p(x_mhn_wifi_fel_use), c='#FFA500', alpha=0.5, linestyle='dashed')

# slope2, intercept2 = np.polyfit(np.log(x_mhn_wifi_fel_use), np.log(p(x_mhn_wifi_fel_use)), 1)
slope2 = ((p(x_mhn_sl_fel_use)[1] - p(x_mhn_sl_fel_use)[0]) / (x_mhn_sl_fel_use[1] - x_mhn_sl_fel_use[0])) * x_mhn_sl_fel_use[0]  ## manual calculation of slope since polyfit didn't work
print("Slope:", slope2)
plt.text(0.055, 1610, round(slope2))

plt.xlabel('miles (0.01 = approx 53 ft)')
plt.ylabel('# of crimes')
plt.legend()
plt.show()

## Create dfs with values

df_mhn_fel_sl = pd.DataFrame()
df_mhn_fel_sl['min_streetlight_distance'] = x_mhn_sl_fel
df_mhn_fel_sl['n_crimes'] = y_mhn_sl_fel
df_mhn_fel_wifi = pd.DataFrame()
df_mhn_fel_wifi['min_wifi_distance'] = x_mhn_wifi_fel
df_mhn_fel_wifi['n_crimes'] = y_mhn_wifi_fel

print(df_mhn_fel_sl)
print(df_mhn_fel_wifi)

## Compute change - streetlights

df_mhn_fel_sl['shifted'] = df_mhn_fel_sl['n_crimes'].shift(1)
df_mhn_fel_sl['percent_change'] = ((df_mhn_fel_sl['n_crimes'] - df_mhn_fel_sl['shifted']) /df_mhn_fel_sl['shifted']) * 100
print(df_mhn_fel_sl)
avg_df_mhn_fel_sl = df_mhn_fel_sl['percent_change'].mean()
# print("Manhattan - Felonies vs. Streetlights Average % Change:", avg_df_mhn_fel_sl)

## Compute change - wifi

df_mhn_fel_wifi['shifted'] = df_mhn_fel_wifi['n_crimes'].shift(1)
df_mhn_fel_wifi['percent_change'] = ((df_mhn_fel_wifi['n_crimes'] - df_mhn_fel_wifi['shifted']) /df_mhn_fel_wifi['shifted']) * 100
print(df_mhn_fel_wifi)
avg_df_mhn_fel_wifi = df_mhn_fel_wifi['percent_change'].mean()
# print("Manhattan - Felonies vs. Wifi Average % Change:", avg_df_mhn_fel_wifi)


######  --------------------------------------------------------------------------- BROOKLYIN

## Create list of list of streetlight locations

streetlights_bkn = []
for index, row in SL_BROOKLYN.iterrows():
    streetlights_bkn.append([row['lon_s'], row['lat_s']])
    
## Create function - finds closest streetlight to each crime and returns distance            

def min_dist_s_bkn(lon1, lat1):
    min_d = 999999999999999
    for loc in streetlights_bkn:
        dist = haversine(lon1, lat1, loc[0], loc[1])
        if dist < min_d:
            min_d = dist
    print(lon1, lat1, min_d)
    return min_d

## Create list of list of wifi locations

wifi_bkn = []
for index, row in wifi_BROOKLYN.iterrows():
    wifi_bkn.append([row['lon_w'], row['lat_w']])

## Create function - finds closest wifi to each crime and returns distance            

def min_dist_w_bkn(lon1, lat1):
    min_d = 999999999999999
    for loc in wifi_bkn:
        dist = haversine(lon1, lat1, loc[0], loc[1])
        if dist < min_d:
            min_d = dist
    print(lon1, lat1, min_d)
    return min_d

## Create additional column of closest streetlight and wifi to each crime

NYPD_BKN_FEL["min_streetlight"] = round(
    NYPD_BKN_FEL.apply(lambda x: min_dist_s_bkn(x['lon_c'], x['lat_c']), axis=1),
    3)
NYPD_BKN_FEL["min_wifi"] = round(
    NYPD_BKN_FEL.apply(lambda x: min_dist_w_bkn(x['lon_c'], x['lat_c']), axis=1),
    3)
print(NYPD_BKN_FEL)

## Min / max of min streetlight and wifi distances

bkn_minSL = NYPD_BKN_FEL['min_streetlight'].min()
print("bkn_minSL:", bkn_minSL)
bkn_maxSL = NYPD_BKN_FEL['min_streetlight'].max()
print("bkn_maxSL:", bkn_maxSL)
bkn_minWifi = NYPD_BKN_FEL['min_wifi'].min()
print("bkn_minWifi:", bkn_minWifi)
bkn_maxWifi = NYPD_BKN_FEL['min_wifi'].max()
print("bkn_maxWifi:", bkn_maxWifi)

## Graph - Brooklyn -  Streetlights vs felonies (night), wifi vs felonies (day)

## Get crimes, filter to only 1 mile radius

bkn_lst_sl_fel_night = []
for index, row in NYPD_BKN_FEL.iterrows():
    if row['Day/Night'] == 'night':
        bkn_lst_sl_fel_night.append(row['min_streetlight'])
# print(bkn_lst_sl_fel_night)

bkn_flst_sl_fel_night = list(filter(lambda bkn_lst_sl_fel_night:
                                    bkn_lst_sl_fel_night <= 0.06, bkn_lst_sl_fel_night))
print(len(bkn_flst_sl_fel_night))

bkn_lst_wifi_fel_day = []
for index, row in NYPD_BKN_FEL.iterrows():
    if row['Day/Night'] == 'day':
        bkn_lst_wifi_fel_day.append(row['min_wifi'])
# print(bkn_lst_wifi_fel_day)

bkn_flst_wifi_fel_day = list(filter(lambda bkn_lst_wifi_fel_day:
                                    bkn_lst_wifi_fel_day <= 0.06, bkn_lst_wifi_fel_day))
print(len(bkn_flst_wifi_fel_day))

## Create x and y

x_bkn_sl_fel = []
x_bkn_sl_fel_use = []
y_bkn_sl_fel = []
x_bkn_wifi_fel = []
x_bkn_wifi_fel_use = []
y_bkn_wifi_fel = []

## Create segments

spacing = 0.01
for i in range(6):
    r1 = round(spacing * i, 3)
    r2 = round(r1 + spacing, 3)
    fdata = list(filter(lambda bkn_flst_sl_fel_night:
                        r1 <= bkn_flst_sl_fel_night < r2, bkn_flst_sl_fel_night))
    x_bkn_sl_fel.append(str(r1) + " - " + str(r2))
    x_bkn_sl_fel_use.append(r2)
    y_bkn_sl_fel.append(len(fdata))

spacing = 0.01
for i in range(6):
    r1 = round(spacing * i, 3)
    r2 = round(r1 + spacing, 3)
    fdata = list(filter(lambda bkn_flst_wifi_fel_day:
                        r1 <= bkn_flst_wifi_fel_day < r2, bkn_flst_wifi_fel_day))
    x_bkn_wifi_fel.append(str(r1) + " - " + str(r2))
    x_bkn_wifi_fel_use.append(r2)
    y_bkn_wifi_fel.append(len(fdata))

## Create graph

plt.suptitle('Brooklyn', fontsize=20)

plt.plot(x_bkn_sl_fel_use, y_bkn_sl_fel, label = "felonies(night) vs. streetlight")
z = np.polyfit(x_bkn_sl_fel_use, y_bkn_sl_fel, 1)
p = np.poly1d(z)
print(x_bkn_sl_fel_use)
print(p(x_bkn_sl_fel_use))
plt.plot(x_bkn_sl_fel_use, p(x_bkn_sl_fel_use), c='#1f77b4', alpha=0.5, linestyle='dashed')

# slope1, intercept1 = np.polyfit(np.log(x_bkn_sl_fel_use), np.log(p(x_bkn_sl_fel_use)), 1)
slope1 = ((p(x_bkn_sl_fel_use)[1] - p(x_bkn_sl_fel_use)[0]) / (x_bkn_sl_fel_use[1] - x_bkn_sl_fel_use[0])) * x_bkn_sl_fel_use[0]  ## manual calculation of slope since polyfit didn't work
print("Slope:", slope1)
plt.text(0.05, 1450, round(slope1))

plt.plot(x_bkn_wifi_fel_use, y_bkn_wifi_fel, label = "felonies(day) vs. wifi")
z = np.polyfit(x_bkn_wifi_fel_use, y_bkn_wifi_fel, 1)
p = np.poly1d(z)
print(x_bkn_wifi_fel_use)
print(p(x_bkn_wifi_fel_use))
plt.plot(x_bkn_wifi_fel_use, p(x_bkn_wifi_fel_use), c='#FFA500', alpha=0.5, linestyle='dashed')

# slope2, intercept2 = np.polyfit(np.log(x_bkn_wifi_fel_use), np.log(p(x_bkn_wifi_fel_use)), 1)
slope2 = ((p(x_bkn_sl_fel_use)[1] - p(x_bkn_sl_fel_use)[0]) / (x_bkn_sl_fel_use[1] - x_bkn_sl_fel_use[0])) * x_bkn_sl_fel_use[0]  ## manual calculation of slope since polyfit didn't work
print("Slope:", slope2)
plt.text(0.05, 650, round(slope2))

plt.xlabel('miles (0.01 = approx 53 ft)')
plt.ylabel('# of crimes')
plt.legend()
plt.show()

## Create dfs with values

df_bkn_fel_sl = pd.DataFrame()
df_bkn_fel_sl['min_streetlight_distance'] = x_bkn_sl_fel
df_bkn_fel_sl['n_crimes'] = y_bkn_sl_fel
df_bkn_fel_wifi = pd.DataFrame()
df_bkn_fel_wifi['min_wifi_distance'] = x_bkn_wifi_fel
df_bkn_fel_wifi['n_crimes'] = y_bkn_wifi_fel

print(df_bkn_fel_sl)
print(df_bkn_fel_wifi)

## Compute change - streetlights

df_bkn_fel_sl['shifted'] = df_bkn_fel_sl['n_crimes'].shift(1)
df_bkn_fel_sl['percent_change'] = ((df_bkn_fel_sl['n_crimes'] - df_bkn_fel_sl['shifted']) /df_bkn_fel_sl['shifted']) * 100
print(df_bkn_fel_sl)
avg_df_bkn_fel_sl = df_bkn_fel_sl['percent_change'].mean()
# print("Brooklyn - Felonies vs. Streetlights Average % Change:", avg_df_bkn_fel_sl)

## Compute change - wifi

df_bkn_fel_wifi['shifted'] = df_bkn_fel_wifi['n_crimes'].shift(1)
df_bkn_fel_wifi['percent_change'] = ((df_bkn_fel_wifi['n_crimes'] - df_bkn_fel_wifi['shifted']) /df_bkn_fel_wifi['shifted']) * 100
print(df_bkn_fel_wifi)
avg_df_bkn_fel_wifi = df_bkn_fel_wifi['percent_change'].mean()
# print("Brooklyn - Felonies vs. Wifi Average % Change:", avg_df_bkn_fel_wifi)


######  --------------------------------------------------------------------------- BRONX

## Create list of list of streetlight locations

streetlights_bnx = []
for index, row in SL_BRONX.iterrows():
    streetlights_bnx.append([row['lon_s'], row['lat_s']])
    
## Create function - finds closest streetlight to each crime and returns distance            

def min_dist_s_bnx(lon1, lat1):
    min_d = 999999999999999
    for loc in streetlights_bnx:
        dist = haversine(lon1, lat1, loc[0], loc[1])
        if dist < min_d:
            min_d = dist
    print(lon1, lat1, min_d)
    return min_d

## Create list of list of wifi locations

wifi_bnx = []
for index, row in wifi_BRONX.iterrows():
    wifi_bnx.append([row['lon_w'], row['lat_w']])

## Create function - finds closest wifi to each crime and returns distance            

def min_dist_w_bnx(lon1, lat1):
    min_d = 999999999999999
    for loc in wifi_bnx:
        dist = haversine(lon1, lat1, loc[0], loc[1])
        if dist < min_d:
            min_d = dist
    print(lon1, lat1, min_d)
    return min_d

## Create additional column of closest streetlight and wifi to each crime

NYPD_BNX_FEL["min_streetlight"] = round(
    NYPD_BNX_FEL.apply(lambda x: min_dist_s_bnx(x['lon_c'], x['lat_c']), axis=1),
    3)
NYPD_BNX_FEL["min_wifi"] = round(
    NYPD_BNX_FEL.apply(lambda x: min_dist_w_bnx(x['lon_c'], x['lat_c']), axis=1),
    3)
print(NYPD_BNX_FEL)

## Min / max of min streetlight and wifi distances

bnx_minSL = NYPD_BNX_FEL['min_streetlight'].min()
print("bnx_minSL:", bnx_minSL)
bnx_maxSL = NYPD_BNX_FEL['min_streetlight'].max()
print("bnx_maxSL:", bnx_maxSL)
bnx_minWifi = NYPD_BNX_FEL['min_wifi'].min()
print("bnx_minWifi:", bnx_minWifi)
bnx_maxWifi = NYPD_BNX_FEL['min_wifi'].max()
print("bnx_maxWifi:", bnx_maxWifi)

## Graph - Bronx -  Streetlights vs felonies (night), wifi vs felonies (day)

## Get crimes, filter to only 1 mile radius

bnx_lst_sl_fel_night = []
for index, row in NYPD_BNX_FEL.iterrows():
    if row['Day/Night'] == 'night':
        bnx_lst_sl_fel_night.append(row['min_streetlight'])
# print(bnx_lst_sl_fel_night)

bnx_flst_sl_fel_night = list(filter(lambda bnx_lst_sl_fel_night:
                                    bnx_lst_sl_fel_night <= 0.06, bnx_lst_sl_fel_night))
print(len(bnx_flst_sl_fel_night))

bnx_lst_wifi_fel_day = []
for index, row in NYPD_BNX_FEL.iterrows():
    if row['Day/Night'] == 'day':
        bnx_lst_wifi_fel_day.append(row['min_wifi'])
# print(bnx_lst_wifi_fel_day)

bnx_flst_wifi_fel_day = list(filter(lambda bnx_lst_wifi_fel_day:
                                    bnx_lst_wifi_fel_day <= 0.06, bnx_lst_wifi_fel_day))
print(len(bnx_flst_wifi_fel_day))

## Create x and y

x_bnx_sl_fel = []
x_bnx_sl_fel_use = []
y_bnx_sl_fel = []
x_bnx_wifi_fel = []
x_bnx_wifi_fel_use = []
y_bnx_wifi_fel = []

## Create segments

spacing = 0.01
for i in range(6):
    r1 = round(spacing * i, 3)
    r2 = round(r1 + spacing, 3)
    fdata = list(filter(lambda bnx_flst_sl_fel_night:
                        r1 <= bnx_flst_sl_fel_night < r2, bnx_flst_sl_fel_night))
    x_bnx_sl_fel.append(str(r1) + " - " + str(r2))
    x_bnx_sl_fel_use.append(r2)
    y_bnx_sl_fel.append(len(fdata))

spacing = 0.01
for i in range(6):
    r1 = round(spacing * i, 3)
    r2 = round(r1 + spacing, 3)
    fdata = list(filter(lambda bnx_flst_wifi_fel_day:
                        r1 <= bnx_flst_wifi_fel_day < r2, bnx_flst_wifi_fel_day))
    x_bnx_wifi_fel.append(str(r1) + " - " + str(r2))
    x_bnx_wifi_fel_use.append(r2)
    y_bnx_wifi_fel.append(len(fdata))

## Create graph

plt.suptitle('The Bronx', fontsize=20)

plt.plot(x_bnx_sl_fel_use, y_bnx_sl_fel, label = "felonies(night) vs. streetlight")
z = np.polyfit(x_bnx_sl_fel_use, y_bnx_sl_fel, 1)
p = np.poly1d(z)
print(x_bnx_sl_fel_use)
print(p(x_bnx_sl_fel_use))
plt.plot(x_bnx_sl_fel_use, p(x_bnx_sl_fel_use), c='#1f77b4', alpha=0.5, linestyle='dashed')

# slope1, intercept1 = np.polyfit(np.log(x_bnx_sl_fel_use), np.log(p(x_bnx_sl_fel_use)), 1)
slope1 = ((p(x_bnx_sl_fel_use)[1] - p(x_bnx_sl_fel_use)[0]) / (x_bnx_sl_fel_use[1] - x_bnx_sl_fel_use[0])) * x_bnx_sl_fel_use[0]  ## manual calculation of slope since polyfit didn't work
print("Slope:", slope1)
plt.text(0.055, 1420, round(slope1))

plt.plot(x_bnx_wifi_fel_use, y_bnx_wifi_fel, label = "felonies(day) vs. wifi")
z = np.polyfit(x_bnx_wifi_fel_use, y_bnx_wifi_fel, 1)
p = np.poly1d(z)
print(x_bnx_wifi_fel_use)
print(p(x_bnx_wifi_fel_use))
plt.plot(x_bnx_wifi_fel_use, p(x_bnx_wifi_fel_use), c='#FFA500', alpha=0.5, linestyle='dashed')

# slope2, intercept2 = np.polyfit(np.log(x_bnx_wifi_fel_use), np.log(p(x_bnx_wifi_fel_use)), 1)
slope2 = ((p(x_bnx_sl_fel_use)[1] - p(x_bnx_sl_fel_use)[0]) / (x_bnx_sl_fel_use[1] - x_bnx_sl_fel_use[0])) * x_bnx_sl_fel_use[0]  ## manual calculation of slope since polyfit didn't work
print("Slope:", slope2)
plt.text(0.055, 590, round(slope2))

plt.xlabel('miles (0.01 = approx 53 ft)')
plt.ylabel('# of crimes')
plt.legend()
plt.show()

## Create dfs with values

df_bnx_fel_sl = pd.DataFrame()
df_bnx_fel_sl['min_streetlight_distance'] = x_bnx_sl_fel
df_bnx_fel_sl['n_crimes'] = y_bnx_sl_fel
df_bnx_fel_wifi = pd.DataFrame()
df_bnx_fel_wifi['min_wifi_distance'] = x_bnx_wifi_fel
df_bnx_fel_wifi['n_crimes'] = y_bnx_wifi_fel

print(df_bnx_fel_sl)
print(df_bnx_fel_wifi)

## Compute change - streetlights

df_bnx_fel_sl['shifted'] = df_bnx_fel_sl['n_crimes'].shift(1)
df_bnx_fel_sl['percent_change'] = ((df_bnx_fel_sl['n_crimes'] - df_bnx_fel_sl['shifted']) /df_bnx_fel_sl['shifted']) * 100
print(df_bnx_fel_sl)
avg_df_bnx_fel_sl = df_bnx_fel_sl['percent_change'].mean()
# print("The Bronx - Felonies vs. Streetlights Average % Change:", avg_df_bnx_fel_sl)

## Compute change - wifi

df_bnx_fel_wifi['shifted'] = df_bnx_fel_wifi['n_crimes'].shift(1)
df_bnx_fel_wifi['percent_change'] = ((df_bnx_fel_wifi['n_crimes'] - df_bnx_fel_wifi['shifted']) /df_bnx_fel_wifi['shifted']) * 100
print(df_bnx_fel_wifi)
avg_df_bnx_fel_wifi = df_bnx_fel_wifi['percent_change'].mean()
# print("The Bronx - Felonies vs. Wifi Average % Change:", avg_df_bnx_fel_wifi)


######  --------------------------------------------------------------------------- STATEN ISLAND

## Create list of list of streetlight locations

streetlights_stn = []
for index, row in SL_STATENISLAND.iterrows():
    streetlights_stn.append([row['lon_s'], row['lat_s']])
    
## Create function - finds closest streetlight to each crime and returns distance            

def min_dist_s_stn(lon1, lat1):
    min_d = 999999999999999
    for loc in streetlights_stn:
        dist = haversine(lon1, lat1, loc[0], loc[1])
        if dist < min_d:
            min_d = dist
    print(lon1, lat1, min_d)
    return min_d

## Create list of list of wifi locations

wifi_stn = []
for index, row in wifi_STATENISLAND.iterrows():
    wifi_stn.append([row['lon_w'], row['lat_w']])

## Create function - finds closest wifi to each crime and returns distance            

def min_dist_w_stn(lon1, lat1):
    min_d = 999999999999999
    for loc in wifi_stn:
        dist = haversine(lon1, lat1, loc[0], loc[1])
        if dist < min_d:
            min_d = dist
    print(lon1, lat1, min_d)
    return min_d

## Create additional column of closest streetlight and wifi to each crime

NYPD_STN_FEL["min_streetlight"] = round(
    NYPD_STN_FEL.apply(lambda x: min_dist_s_stn(x['lon_c'], x['lat_c']), axis=1),
    3)
NYPD_STN_FEL["min_wifi"] = round(
    NYPD_STN_FEL.apply(lambda x: min_dist_w_stn(x['lon_c'], x['lat_c']), axis=1),
    3)
print(NYPD_STN_FEL)

## Min / max of min streetlight and wifi distances

stn_minSL = NYPD_STN_FEL['min_streetlight'].min()
print("stn_minSL:", stn_minSL)
stn_maxSL = NYPD_STN_FEL['min_streetlight'].max()
print("stn_maxSL:", stn_maxSL)
stn_minWifi = NYPD_STN_FEL['min_wifi'].min()
print("stn_minWifi:", stn_minWifi)
stn_maxWifi = NYPD_STN_FEL['min_wifi'].max()
print("stn_maxWifi:", stn_maxWifi)

## Graph - Staten Island -  Streetlights vs felonies (night), wifi vs felonies (day)

## Get crimes, filter to only 1 mile radius

stn_lst_sl_fel_night = []
for index, row in NYPD_STN_FEL.iterrows():
    if row['Day/Night'] == 'night':
        stn_lst_sl_fel_night.append(row['min_streetlight'])
# print(stn_lst_sl_fel_night)

stn_flst_sl_fel_night = list(filter(lambda stn_lst_sl_fel_night:
                                    stn_lst_sl_fel_night <= 0.06, stn_lst_sl_fel_night))
print(len(stn_flst_sl_fel_night))

stn_lst_wifi_fel_day = []
for index, row in NYPD_STN_FEL.iterrows():
    if row['Day/Night'] == 'day':
        stn_lst_wifi_fel_day.append(row['min_wifi'])
# print(stn_lst_wifi_fel_day)

stn_flst_wifi_fel_day = list(filter(lambda stn_lst_wifi_fel_day:
                                    stn_lst_wifi_fel_day <= 0.06, stn_lst_wifi_fel_day))
print(len(stn_flst_wifi_fel_day))

## Create x and y

x_stn_sl_fel = []
x_stn_sl_fel_use = []
y_stn_sl_fel = []
x_stn_wifi_fel = []
x_stn_wifi_fel_use = []
y_stn_wifi_fel = []

## Create segments

spacing = 0.01
for i in range(6):
    r1 = round(spacing * i, 3)
    r2 = round(r1 + spacing, 3)
    fdata = list(filter(lambda stn_flst_sl_fel_night:
                        r1 <= stn_flst_sl_fel_night < r2, stn_flst_sl_fel_night))
    x_stn_sl_fel.append(str(r1) + " - " + str(r2))
    x_stn_sl_fel_use.append(r2)
    y_stn_sl_fel.append(len(fdata))

spacing = 0.01
for i in range(6):
    r1 = round(spacing * i, 3)
    r2 = round(r1 + spacing, 3)
    fdata = list(filter(lambda stn_flst_wifi_fel_day:
                        r1 <= stn_flst_wifi_fel_day < r2, stn_flst_wifi_fel_day))
    x_stn_wifi_fel.append(str(r1) + " - " + str(r2))
    x_stn_wifi_fel_use.append(r2)
    y_stn_wifi_fel.append(len(fdata))

## Create graph

plt.suptitle('Staten Island', fontsize=20)

plt.plot(x_stn_sl_fel_use, y_stn_sl_fel, label = "felonies(night) vs. streetlight")
z = np.polyfit(x_stn_sl_fel_use, y_stn_sl_fel, 1)
p = np.poly1d(z)
print(x_stn_sl_fel_use)
print(p(x_stn_sl_fel_use))
plt.plot(x_stn_sl_fel_use, p(x_stn_sl_fel_use), c='#1f77b4', alpha=0.5, linestyle='dashed')

# slope1, intercept1 = np.polyfit(np.log(x_stn_sl_fel_use), np.log(p(x_stn_sl_fel_use)), 1)
slope1 = ((p(x_stn_sl_fel_use)[1] - p(x_stn_sl_fel_use)[0]) / (x_stn_sl_fel_use[1] - x_stn_sl_fel_use[0])) * x_stn_sl_fel_use[0]  ## manual calculation of slope since polyfit didn't work
print("Slope:", slope1)
plt.text(0.05, 28, round(slope1, 1))

plt.plot(x_stn_wifi_fel_use, y_stn_wifi_fel, label = "felonies(day) vs. wifi")
z = np.polyfit(x_stn_wifi_fel_use, y_stn_wifi_fel, 1)
p = np.poly1d(z)
print(x_stn_wifi_fel_use)
print(p(x_stn_wifi_fel_use))
plt.plot(x_stn_wifi_fel_use, p(x_stn_wifi_fel_use), c='#FFA500', alpha=0.5, linestyle='dashed')

# slope2, intercept2 = np.polyfit(np.log(x_stn_wifi_fel_use), np.log(p(x_stn_wifi_fel_use)), 1)
slope2 = ((p(x_stn_sl_fel_use)[1] - p(x_stn_sl_fel_use)[0]) / (x_stn_sl_fel_use[1] - x_stn_sl_fel_use[0])) * x_stn_sl_fel_use[0]  ## manual calculation of slope since polyfit didn't work
print("Slope:", slope2)
plt.text(0.05, 23, round(slope2, 1))

plt.xlabel('miles (0.01 = approx 53 ft)')
plt.ylabel('# of crimes')
plt.legend()
plt.show()

## Create dfs with values

df_stn_fel_sl = pd.DataFrame()
df_stn_fel_sl['min_streetlight_distance'] = x_stn_sl_fel
df_stn_fel_sl['n_crimes'] = y_stn_sl_fel
df_stn_fel_wifi = pd.DataFrame()
df_stn_fel_wifi['min_wifi_distance'] = x_stn_wifi_fel
df_stn_fel_wifi['n_crimes'] = y_stn_wifi_fel

print(df_stn_fel_sl)
print(df_stn_fel_wifi)

## Compute change - streetlights

df_stn_fel_sl['shifted'] = df_stn_fel_sl['n_crimes'].shift(1)
df_stn_fel_sl['percent_change'] = ((df_stn_fel_sl['n_crimes'] - df_stn_fel_sl['shifted']) /df_stn_fel_sl['shifted']) * 100
print(df_stn_fel_sl)
avg_df_stn_fel_sl = df_stn_fel_sl['percent_change'].mean()
# print("Staten Island - Felonies vs. Streetlights Average % Change:", avg_df_stn_fel_sl)

## Compute change - wifi

df_stn_fel_wifi['shifted'] = df_stn_fel_wifi['n_crimes'].shift(1)
df_stn_fel_wifi['percent_change'] = ((df_stn_fel_wifi['n_crimes'] - df_stn_fel_wifi['shifted']) /df_stn_fel_wifi['shifted']) * 100
print(df_stn_fel_wifi)
avg_df_stn_fel_wifi = df_stn_fel_wifi['percent_change'].mean()
# print("Staten Island - Felonies vs. Wifi Average % Change:", avg_df_stn_fel_wifi)

