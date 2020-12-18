
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as pyplot
import pandas as pandas
from datetime import timedelta
import numpy as np
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

def exponential_fit(x, a, b, c):
    return a*np.exp(-b*x) + c


# In[ ]:


#dataset = pandas.read_csv("~/Downloads/v3Nbr69N",sep=",",header=0,encoding="utf8",index_col=0,parse_dates=True,dayfirst=True)
#dataset_dl = pandas.read_csv("https://opendata.ecdc.europa.eu/covid19/casedistribution/csv",sep=",",header=0,encoding="utf8",index_col=0,parse_dates=True,dayfirst=True)
dataset_bonn_dl = pandas.read_csv("https://stadtplan.bonn.de/csv?OD=4379",sep=";",header=0,encoding="utf8",index_col=0,parse_dates=True)
dataset_muenster_dl = pandas.read_csv("https://raw.githubusercontent.com/od-ms/resources/master/coronavirus-fallzahlen-regierungsbezirk-muenster.csv",sep=",",header=0,encoding="utf8",parse_dates=True,dayfirst=True)

# Select either offical data with Meldeverzug or crowd sourced data
#dataset_states_dl = pandas.read_csv("https://raw.githubusercontent.com/jgehrcke/covid-19-germany-gae/master/cases-rki-by-state.csv",sep=",",header=0,encoding="utf8",index_col=0,parse_dates=True)
dataset_states_dl = pandas.read_csv("https://raw.githubusercontent.com/jgehrcke/covid-19-germany-gae/master/cases-rl-crowdsource-by-state.csv",sep=",",header=0,encoding="utf8",index_col=0,parse_dates=True)

states_ew_dl = pandas.read_csv("https://www.nieuwenhuisen.de/states_ew.csv",sep=",",header=0,encoding="utf8",index_col=0)


# In[ ]:


# Setup
#countries = ["DE","SE","US"]
countries = ["BN","DE","DE-NW","DE-SN"]
max_days=60
extrapolation_look_back = 10
extrapolation_future = 4

dataset = dataset_dl
dataset_states = dataset_states_dl
dataset_bonn = dataset_bonn_dl
dataset_regbez_muenster = dataset_muenster_dl
states_ew = states_ew_dl.to_dict(orient="index")


# In[ ]:


# Convert Bonn dataset to ECDC format and append
dataset_bonn_sorted = dataset_bonn.sort_index(ascending=True)
dataset_bonn_sorted.index = dataset_bonn_sorted.index.shift(-1, freq='D')
dataset_bonn_sorted = dataset_bonn_sorted.rename(columns={"positiv_getest": "BN"})
relevant_bonn = dataset_bonn_sorted[['BN']]

dataset_states = dataset_states.rename(columns={"sum_cases": "DE"})
dataset_states.index = dataset_states.index.date
merged_data = pandas.merge(relevant_bonn,dataset_states,how='left',left_index=True,right_index=True)
merged_data_diff = merged_data.diff()
merged_data_diff = merged_data_diff.sort_index(ascending=False)


# In[ ]:


for country in countries:
    merged_data_diff["cases_per_100k_"+country] = merged_data_diff[country] * 100000 / states_ew[country]['Ew']

    indexer = pandas.api.indexers.FixedForwardWindowIndexer(window_size=7)
for country in countries:
    merged_data_diff['cum_cases_7_days_'+country] = merged_data_diff['cases_per_100k_'+country].shift(1).rolling(window=indexer).sum()


# In[ ]:


if "MUENSTER" in countries:
    # Convert Muenster dataset to ECDC format and append
    dataset_muenster = dataset_regbez_muenster[dataset_regbez_muenster['Gebiet'].isin(['Stadt Münster'])]
    dataset_muenster_relevant = dataset_muenster[['Datum','Bestätigte Faelle']].set_index('Datum')

    dataset_muenster_relevant.index = pandas.DatetimeIndex(dataset_muenster_relevant.index,dayfirst=True)
    dataset_muenster_relevant = dataset_muenster_relevant.diff(periods=-1)

    idx = pandas.date_range(start=dataset_muenster_relevant.index[-1] - timedelta(days=0), end=dataset_muenster_relevant.index[0] - timedelta(days=0))[::-1]
    dataset_muenster_relevant = dataset_muenster_relevant.reindex(idx,fill_value=0)
    dataset_muenster_relevant['popData2019'] = 315293
    dataset_muenster_relevant['countriesAndTerritories'] = 'Münster'
    dataset_muenster_relevant['geoId'] = 'MUENSTER'
    dataset_muenster_relevant = dataset_muenster_relevant.rename(columns={"Bestätigte Faelle": "cases"})
    dataset = dataset.append(dataset_muenster_relevant)


# In[ ]:


# Get relevant columns
#relevant = dataset[['geoId','cases','popData2019','countriesAndTerritories']]
# Put requested countries in different tables, rename columns with country suffix for later merging

#countrydata = {}
#for country in countries:
#    countrydata[country] = relevant[relevant['geoId'].isin([country])]
#    countrydata[country] = countrydata[country].rename(columns={"cases": "cases_"+country, "popData2019": "popData2019_"+country})

# Merge country data into single table
#merged_data = countrydata[countries[0]]
#for country in countries[1:]:
#    merged_data = pandas.merge(merged_data,countrydata[country],how='left',left_index=True,right_index=True)


# In[ ]:


# Calculate 7-day incidence
#for country in countries:
#    merged_data["cases_per_100k_"+country] = merged_data["cases_"+country] * 100000 / merged_data["popData2019_"+country]

#indexer = pandas.api.indexers.FixedForwardWindowIndexer(window_size=7)
#for country in countries:
#    merged_data['cum_cases_7_days_'+country] = merged_data['cases_per_100k_'+country].shift(1).rolling(window=indexer).sum()

pyplot.rcParams.update({'font.size': 22})
# Plot data
fig = pyplot.figure(figsize=(16, 12))
ax1 = fig.add_subplot(1,1,1)

ax1.set_ylabel('Neuinfektionen pro 7 Tage pro 100.000 Einwohner')
ax1.tick_params(labelright=True)

for country in countries:
    color = next(ax1._get_lines.prop_cycler)['color']
    #country_name = dataset[dataset['geoId']==country].head(1)['countriesAndTerritories'].iloc[0]
    country_name = states_ew[country]['State']
    merged_data_diff[1:max_days]['cum_cases_7_days_'+country].plot(kind='line',ax=ax1,label=country_name,color=color,linewidth=3)
    # extrapolation
    try:
        if country != "MUENSTER":
            y = merged_data_diff[1:extrapolation_look_back]['cum_cases_7_days_' + country]
            i = y.index.to_pydatetime()
            xi = mdates.date2num(i)
            p0 = [1,1,y[-1]]
            fitting_parameters, covariance = curve_fit(exponential_fit, range(len(xi)), y, p0 = p0 )
            a, b, c = fitting_parameters
            new_data = []
            new_data_index = []
            for i in reversed(range(-extrapolation_future,len(y))):
                new_data.append(exponential_fit(i, a, b, c))
                new_data_index.append(y.index[0]+pandas.DateOffset(-i))
            df = pandas.DataFrame(index=new_data_index,data=new_data)
            pyplot.plot(df,color=color,linestyle='dashed')
    except:
        pass
    
ax1.legend()
#ax1.legend(prop={'size': 32})
ax1.set_xlabel('Datum')
pyplot.axvline(x='2020-11-02', ymin=0, ymax=160, linestyle='--',color='black',label="Lockdown Light")
pyplot.text('2020-11-03', # x-value from data
                 80, # wiggle the labels 2so they don't overlap
                 'Lockdown Light', # string label
                 color='black',rotation=90)
pyplot.axvline(x='2020-12-24', ymin=0, ymax=160, linestyle='--',color='black',label="Heilig Abend")
pyplot.text('2020-12-25', # x-value from data
                 80, # wiggle the labels 2so they don't overlap
                 'Heilig Abend', # string label
                 color='black',rotation=90)

pyplot.axvline(x='2020-12-02', ymin=0, ymax=160, linestyle='--',color='black',label="Heilig Abend")

pyplot.savefig("/home/nieuwenh/ownCloud/cases.png")
pyplot.show()

# Print last two days
print( merged_data_diff[0:1].index.date[0] )
for country in countries:
    #country_name = dataset[dataset['geoId']==country].head(1)['countriesAndTerritories'].iloc[0]
    country_name = states_ew[country]['State']
    country_incidence = merged_data_diff[1:2]['cum_cases_7_days_'+country].iloc[0]
    country_incidence_yesterday = merged_data_diff[2:3]['cum_cases_7_days_'+country].iloc[0]
    print( "{}: {:.1f} (Gestern: {:.1f})".format( country_name, country_incidence, country_incidence_yesterday ) )

