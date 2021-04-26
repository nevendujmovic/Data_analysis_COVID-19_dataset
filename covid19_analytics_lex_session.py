# -*- coding: utf-8 -*-
"""
Created on Apr 25 2021

@author: Neven Dujmovic
"""

# Import libraries
import pandas as pd
# import numpy as np

# Read Covid-19 dataset with daily statistics.
data_full = pd.read_csv('owid-covid-data.csv')

# #####################################################
# ## understanding the data is always the first step ##
# #####################################################

# # Check the number of records for each column.
# # Note that in the term "feature" is used in the Python/Pandas environment. 
# # A feature in case of a dataset simply means a column.    
number_of_records_all = data_full.count()\
                    .reset_index(name='count')\
                    .sort_values(['count'], ascending=False)

print ('')
print ('Missing values in the dataset')
print (data_full.isnull().sum(axis=0))

# To define our analysis strategy, we need to understand which 
# data are variable and which data are constant.
# Let's check out the unique values in all columns.
temp1 = data_full.copy() 
unique_data_full = temp1.groupby('location').nunique().add_prefix('num_').reset_index()


# Let's select the columns with data useful for the analysis.
# # Note that in the term "feature" is used in the Python/Pandas environment. 
# # A feature in case of a dataset simply means a column.
data = data_full[['location', 'continent', 'date', 'total_deaths', 'new_deaths', \
                  'stringency_index', 'population', 'population_density', \
                  'median_age', 'aged_65_older', 'aged_70_older', \
                  'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', \
                  'diabetes_prevalence', 'female_smokers', 'male_smokers', \
                  'hospital_beds_per_thousand', 'life_expectancy', \
                  'human_development_index']]


# We need to analyse daily death rate per 1 million people.
# First all null entries concernig those values must be removed 
# from 'new_deaths' and 'population'.
covid_ds = data.dropna(subset = ['new_deaths'])
covid_ds = covid_ds.dropna(subset = ['population'])
print ('')
print ('Missing values in the dataset')
print (covid_ds.isnull().sum(axis=0))

# There is a strange inconsistency total_deaths
total_deaths_nan = covid_ds[covid_ds['total_deaths'].isnull()]
# We have to clean null entries for 'total_deaths' column.
covid_ds = covid_ds.dropna(subset = ['total_deaths'])
print ('')
print ('Missing values in the dataset')
print (covid_ds.isnull().sum(axis=0))
    

# We will add an additional calculated column ('day_deaths_per_million')   
covid_ds['day_deaths_per_million'] = covid_ds['new_deaths'] / covid_ds['population'] * 1000000

# Let's introduce time dimension in our analysis.
covid_ds_time_filter = covid_ds\
   .loc[(covid_ds['date'] >= '2020-07-01') & (covid_ds['date'] <= '2021-04-25')]


# data can be gruped and conclusion can be made from average score
avg_day_deaths_per_million =\
         covid_ds_time_filter.groupby(['location'])['day_deaths_per_million']\
         .mean()\
         .reset_index()\
         .sort_values(['day_deaths_per_million'], ascending=False)


# import libraries that will enable plotting of graphs
import matplotlib.pyplot as plt

covid_ds_pivot = pd.pivot_table(covid_ds_time_filter, index=['date'], values=['day_deaths_per_million'], columns=['location'])
covid_ds_pivot.columns = covid_ds_pivot.columns.droplevel(0) #remove amount
covid_ds_pivot.columns.name = None #remove categories
# covid_ds_pivot = covid_ds_pivot.reset_index() #index to columns
# # covid_ds_pivot_sel = covid_ds_pivot[['Greece', 'Italy', 'Mexico', 'Australia']]
covid_ds_pivot_sel = covid_ds_pivot[['Croatia', 'Hungary', 'Slovenia']]
covid_ds_pivot_sel = covid_ds_pivot_sel.dropna()
covid_ds_pivot_sel.plot(y=['Croatia', 'Hungary', 'Slovenia'], kind = 'line')
plt.show()
    


    
# ###############################
# ## fatality factors analysis ##
# ###############################

# GOAL: how much the total death rate per country is influenced by various factors.
# Let's select variable and constant columns.
temp2 = covid_ds.copy() 

unique_covid_ds = temp2.groupby('location')['location', 'continent', 'date', 'total_deaths', 'new_deaths', \
                  'stringency_index', 'population', 'population_density', \
                  'median_age', 'aged_65_older', 'aged_70_older', \
                  'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', \
                  'diabetes_prevalence', 'female_smokers', 'male_smokers', \
                  'hospital_beds_per_thousand', 'life_expectancy', \
                  'human_development_index'].nunique().add_prefix('num_').reset_index()


# We can conclude that 'total_deaths' column values vary with cumulative 
# behavior from day to day tracking.
# Daily cahanging values are not in the focus here, so we can remove 'date' and 'new_deaths'.
# We should additiaonaly consider 'stringency_index' where data is not always constant, but 
# since data is avalibale at level of 85% it could be interestnig to consider.
# The proposed aproach is to create calculated column with avarage 'stringency_index'.
# We will add an additional calculated column ('stringency_index_avg').
covid_ds['stringency_index_avg'] = covid_ds.groupby(['location'])['stringency_index'].transform('mean')

# All other values from the selected columns are constant if available
# Also, there are NULL values in the remaining columns, but it is not a stopper for further analysis.


# We identified the following columns that are needed for our analysis.
covid_ds_selected = covid_ds[['location', 'continent', 'total_deaths', \
                  'stringency_index_avg', 'population', 'population_density', \
                  'median_age', 'aged_65_older', 'aged_70_older', \
                  'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', \
                  'diabetes_prevalence', 'female_smokers', 'male_smokers', \
                  'hospital_beds_per_thousand', 'life_expectancy', \
                  'human_development_index']]



# # For the purpose of our analysis only the final figure of 'total_deaths' is required.
# # With usage of 'loc' and and 'idmax' final dataset will be created with data grouped by country.
# # A prerequisite for grouping is to remove NaN or NULL values from a column 
# # that is being tested by idxmax(). For now, NaN values in other columns will remain.
covid_ds_selected = covid_ds_selected.dropna(subset = ["total_deaths"])

covid_ds_grouped = covid_ds_selected.loc[covid_ds_selected.groupby(['location'])["total_deaths"].idxmax()]

# Data cleaning is good and neccesery practice before being engaged in further data analysis.
list_of_countries = data.drop_duplicates('location')[['location']].reset_index()

# # We must remove entries that are not countries. Also those entries as 
# # probably and outliers in certain categories and they will distort the 
# # results if being left in the dataset. Let's sort entries by "population" and 
# # find out what are the suspicious entries.
covid_ds_grouped_sorted = covid_ds_grouped.sort_values(['population'], ascending=False)

# # Continents or non-country data entries must be removed:
# # 'World', 'Asia' 'Africa', 'Europe', 'European Union', 'North America', 
# # 'South America', 'International'
covid_ds_grouped.drop(covid_ds_grouped[covid_ds_grouped['location'] == 'World'].index, inplace = True)
covid_ds_grouped.drop(covid_ds_grouped[covid_ds_grouped['location'] == 'Asia'].index, inplace = True)
covid_ds_grouped.drop(covid_ds_grouped[covid_ds_grouped['location'] == 'Africa'].index, inplace = True)
covid_ds_grouped.drop(covid_ds_grouped[covid_ds_grouped['location'] == 'Europe'].index, inplace = True)
covid_ds_grouped.drop(covid_ds_grouped[covid_ds_grouped['location'] == 'European Union'].index, inplace = True)
covid_ds_grouped.drop(covid_ds_grouped[covid_ds_grouped['location'] == 'North America'].index, inplace = True)
covid_ds_grouped.drop(covid_ds_grouped[covid_ds_grouped['location'] == 'South America'].index, inplace = True)
covid_ds_grouped.drop(covid_ds_grouped[covid_ds_grouped['location'] == 'International'].index, inplace = True)
covid_ds_grouped.drop(covid_ds_grouped[covid_ds_grouped['location'] == 'Oceania'].index, inplace = True)


# We will add an additional calculated column ("deaths_per_100k") 
# where the percentage of the ratio between Covid-19 test cases and 
# tests that are done in the specific day will be stored.    
covid_ds_grouped['deaths_per_100k'] = \
    covid_ds_grouped['total_deaths'] / covid_ds_grouped['population'] * 100000

# The final selecton of columnes for data analysis
covid_ds_grouped_final = covid_ds_grouped[['location', 'continent', 'deaths_per_100k', \
                  'stringency_index_avg', 'population_density', \
                  'median_age', 'aged_65_older', 'aged_70_older', \
                  'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', \
                  'diabetes_prevalence', 'female_smokers', 'male_smokers', \
                  'hospital_beds_per_thousand', 'life_expectancy', \
                  'human_development_index']] \
                  .sort_values(['deaths_per_100k'], ascending=False)

# # Let's see the correlation between all data in the dataset
covid_ds_grouped_final_corr_pearson = covid_ds_grouped_final.corr(method ='pearson')
covid_ds_grouped_final_corr_kendall = covid_ds_grouped_final.corr(method ='kendall')
covid_ds_grouped_final_corr_spearman = covid_ds_grouped_final.corr(method ='spearman')

# # #############################################################################
# # Rule of Thumb for Interpreting the Size of a Correlation Coefficient
# #
# #    Size of Correlation	       Interpretation
# #    .90 to 1.00 (−.90 to −1.00)   Very high positive (negative) correlation
# #    .70 to .90 (−.70 to −.90)	   High positive (negative) correlation
# #    .50 to .70 (−.50 to −.70)	   Moderate positive (negative) correlation
# #    .30 to .50 (−.30 to −.50)	   Low positive (negative) correlation
# #    .00 to .30 (.00 to −.30)	   negligible correlation
# # #############################################################################

# # # Let's clean our dataset from missing values to see effect on correlaton.
covid_ds_grouped_final_clean = covid_ds_grouped_final.dropna()
print ('')
print ('Missing values in the dataset after we dropped NA')
print (covid_ds_grouped_final_clean.isnull().sum(axis=0))


# # Let's repaet the correlation between all data in the dataset
covid_ds_grouped_final_clean_corr_pearson_clean = covid_ds_grouped_final_clean.corr(method ='pearson')
covid_ds_grouped_final_clean_kendall_clean = covid_ds_grouped_final_clean.corr(method ='kendall')
covid_ds_grouped_final_clean_corr_spearman_clean = covid_ds_grouped_final_clean.corr(method ='spearman')

# There are some adjustments, but data similar correlations remain.

# ##########################
# ## Adding new DataSets ##
# ##########################

# Read obesity rates dataset.
# https://worldpopulationreview.com/country-rankings/obesity-rates-by-country
obesity_rates_by_country = pd.read_csv('obesity-rates-by-country-2021_data.csv')

covid_ds_obesity = covid_ds_grouped_final.join(obesity_rates_by_country.set_index('location'), on='location')


# Read average yearly temperature in celsius dataset with daily statistics.
# https://en.wikipedia.org/wiki/List_of_countries_by_average_yearly_temperature
average_temperature_by_country = pd.read_csv('average-yearly-temperature-celsius.csv', \
                                              sep=';', decimal=",", dtype={0:str, 1:float})

covid_ds_obesity_temperature = covid_ds_obesity.join(average_temperature_by_country.set_index('location'), on='location')

# Let's repaet the correlation between all data in the dataset
covid_ds_obesity_temperature_corr_pearson = covid_ds_obesity_temperature.corr(method ='pearson')
covid_ds_obesity_temperature_corr_kendall = covid_ds_obesity_temperature.corr(method ='kendall')
covid_ds_obesity_temperature_corr_spearman = covid_ds_obesity_temperature.corr(method ='spearman')




# #############################
# ## Adding regional filters ##
# #############################

covid_ds_region = covid_ds_obesity_temperature\
                    .loc[(covid_ds_obesity_temperature['continent'] == 'Europe')\
                           | (covid_ds_obesity_temperature['continent'] == 'Asia')]

# Let's repaet the correlation between all data in the dataset
covid_ds_region_corr_pearson = covid_ds_region.corr(method ='pearson')
covid_ds_region_corr_kendall = covid_ds_region.corr(method ='kendall')
covid_ds_region_corr_spearman = covid_ds_region.corr(method ='spearman')



covid_ds_europe = covid_ds_obesity_temperature\
                    .loc[(covid_ds_obesity_temperature['continent'] == 'Europe')]

# define x and y axes from the results dataset and select "bar" type graph
ax = covid_ds_europe.plot(x ='location', y='deaths_per_100k', kind = 'bar')
plt.show()

# define x and y axes from the results dataset and select "bar" type graph
ax = covid_ds_europe.plot(x ='location', y='deaths_per_100k', kind = 'bar')

# to avoid crowded labels on x-axes, we will show every third country and 
# rotate labels by 90 degrees
for i, t in enumerate(ax.get_xticklabels()):
   if (i % 3) != 0:
       t.set_visible(False)

plt.xticks(rotation=90)

# additionally we can set chart title, legend and axes labels
plt.title('COVID-19 data results Europe')
plt.xlabel('Countries')
plt.ylabel('deaths per 100k')

plt.show()






