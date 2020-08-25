# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import pandas
import pandas as pd


##### STEP 2: COLLECT DATA #####

# Download data in .CSV file from the web:
# https://ourworldindata.org/coronavirus-testing
# Load the data from the comma-separated values (CSV) text file 
# to the "covid_data" variable.
covid_data = pd.read_csv('owid-covid-data_new.csv')


##### STEP 3: PREPARE DATA #####

# Make a copy of dataset to use it for data preparation.
ds = covid_data.copy()

# We identified the following three columns that are needed for our analysis.
covid_ds = ds[['location', 'date', 'new_tests', 'new_cases']]
# The "location" column contains information about the Country,
# the "date" column is important to have a time dimension,
# the "new_tests" counts the number of tests done on a specific date,
# the "new_cases" counts a number of reported Covid-19 positive cases.

# Let's do a simple check of the quality of data concerning number of records 
# and the number of missing values.

# Check the number of records for each column.
# Note that in the term "feature" is used in the Python/Pandas environment. 
# A feature in case of a dataset simply means a column.
number_of_records1 = covid_ds.count()\
                    .reset_index(name='count')\
                    .sort_values(['count'], ascending=False)

# We noticed a different number of records which implies the presence of 
# a significant number of NULL values or as Python calls it "nan" values.
# This has to be intreprated further with nice Python/Pandas line of code.
# Let's identify the columns with missing values along with the count and 
# print it on the console.
print ('')
print ('Missing values in the dataset')
print (covid_ds.isnull().sum(axis=0))

# The good news is that the "location" and "date" have no missing values.

# The bad news is that the "new_tests" column only 11606 records 
# out of 37194 records have value entry.
# DECISION: it is not possible to perform analysis without "new_tests" value.
# We will remove all rows where the "new_tests" value is missing.
covid_ds = covid_ds.dropna(subset = ["new_tests"])
print ('')
print ('Missing values in the dataset after we dropped \"new_tests\"')
print (covid_ds.isnull().sum(axis=0))


# Also, it is not good that the "new_cases" column has still missing values.
# Let's select records where the "new_cases" column has NaN.
new_cases_nan = covid_ds[covid_ds['new_cases'].isnull()]
# Well, there are some discoveries:
# 1) Data where "new_tests" are done and "new_cases" are missing are all 
#    in the period from 2020-02-06 to 2020-03-19. This was at the beginning 
#    of the pandemic and this could be a test accuracy or reporting problem.
# 2) Also, we can see that the United Arab Emirates conducted 33555 tests in 
#    that period and data of the "new_cases" are missing. We can say that those 
#    records from the United Arab Emirates are clear outliers that will 
#    significantly influence results for that country.
#    Note: Outliers are unusual values in the dataset that significantly vary 
#    from other data. Outliers are very problematic for many analyses because 
#    they can distort results and cause tests to either miss significant 
#    findings.
#  DECISION: We will remove all rows where the "new_cases" value is missing.
covid_ds = covid_ds.dropna(subset = ["new_cases"])
print ('')
print ('Missing values in the dataset after we dropped \"new_cases\"')
print (covid_ds.isnull().sum(axis=0))

# Again, we will check the number of records for each column.
number_of_records2 = covid_ds.count()\
                    .reset_index(name='count')\
                    .sort_values(['count'], ascending=False)
# Great! There are now 11457 records without missing values.


# Let’s look at possible records with cases where there are 
# more new positive cases for Covid-19 than tests in a single day. 
# This will not make sense for our analysis.             
temp1_before = covid_ds\
       .loc[covid_ds['new_cases'] > covid_ds['new_tests']]
# Unfortunately, we have found such cases. However, this is only 
# for the 65 records. It is also worrying that we have seen negative 
# values. This situation should be an alarm to the analyst. 
# Perhaps, the data collection was not adequate, and it may be best 
# to do a new collection or reach for some other data sources.
# DECISION: We will remove all rows where the "new_cases" > "new_tests"
covid_ds.drop(covid_ds[covid_ds['new_cases'] > covid_ds['new_tests']]\
              .index, inplace = True)
temp1_after = covid_ds.loc[covid_ds['new_cases'] > covid_ds['new_tests']]


# Also, the same number of tests and positive cases could be unrealistic 
# since 100% accuracy of the test is not expected. This scenario is 
# most likely an error. 
temp2_before = covid_ds\
       .loc[covid_ds['new_cases'] == covid_ds['new_tests']]
# There are six records found! The inspection of data shows that it 
# is probably an error.
# DECISION: We will remove all rows where the "new_cases" == "new_tests"       
covid_ds.drop(covid_ds[covid_ds['new_cases'] == covid_ds['new_tests']]\
              .index, inplace = True)  
temp2_after = covid_ds\
       .loc[covid_ds['new_cases'] == covid_ds['new_tests']]
       
# Again, we will check the number of records for each column.
number_of_records3 = covid_ds.count()\
                    .reset_index(name='count')\
                    .sort_values(['count'], ascending=False)


# Now, we will also check for all negative values for "new_tests" 
# and "new_cases." Those values must be removed for our dataset 
# since it is evident that we are dealing with error entries.

temp3 = covid_ds.loc[covid_ds['new_tests'] < 0 ]
# No action required since there are no records where "new_tests" 
# have a negative value number.

temp4_before = covid_ds.loc[covid_ds['new_cases'] < 0 ]
# There are 7 records where "new_cases" has negative value number. 
covid_ds.drop(covid_ds[covid_ds['new_cases'] < 0]\
              .index, inplace = True)
temp4_after = covid_ds.loc[covid_ds['new_cases'] < 0 ]


# Again, we will check the number of records for each column.
number_of_records4 = covid_ds.count()\
                    .reset_index(name='count')\
                    .sort_values(['count'], ascending=False)


# Let's see a list of countries to verify entries.
list_of_countries = covid_ds.drop_duplicates('location')[['location']]\
                    .reset_index()
# We are left with 86 countries for the analysis after data cleaning activity.

# Note: The reset_index() is used to add new index column starting with 0 
# for the resulting dataset. Usage is optional and if not used, 
# original index entries remain, and in some cases that could be useful 
# to link record back to original data.

# This ends the "data preparation" step, which is most usually 
# the most time consuming and complicated part of the analytical process.
# Now "covid_ds"   dataset is after adjustments and data cleaning prepared 
# for the analyses. It is not the perfect data source, and as mentioned above, 
# its use would probably be questionable regarding data quality, but for this
# training, it will be just fine.


##### STEP 4: ANALYZE DATA #####

# We will add an additional calculated column ("test_percentage") 
# where the percentage of the ratio between Covid-19 test cases and tests 
# that are done in the specific day will be stored.
covid_ds['test_percentage'] = covid_ds['new_cases'] / covid_ds['new_tests'] * 100

# Now it is time approach to the goal of our analysis and list 
# average percentages grouped by countries.
avg_percentage_test_cases =\
          covid_ds.groupby(['location'])['test_percentage']\
          .mean()\
          .reset_index()\
          .sort_values(['test_percentage'], ascending=False)
          

# We can also give a time dimension to our query by providing date values to 
# analyze data only in the specific period.
# Execute the following Python code get results for the specific time period:

covid_ds_time_filter = covid_ds\
       .loc[(covid_ds['date'] >= '2020-07-25') & (covid_ds['date'] <= '2020-08-25')]
       
avg_percentage_time_filter =\
          covid_ds_time_filter.groupby(['location'])['test_percentage']\
          .mean()\
          .reset_index()\
          .sort_values(['test_percentage'], ascending=False)
                
       
       
       
##### STEP 5: INTERPRET RESULTS #####

# import libraries that will enable plotting of graphs
import matplotlib.pyplot as plt

# define x and y axes from the results dataset and select "bar" type graph
ax = avg_percentage_time_filter.plot(x ='location', y='test_percentage', kind = 'bar')

# to avoid crowded labels on x-axes, we will show every third country and 
# rotate labels by 90 degrees
for i, t in enumerate(ax.get_xticklabels()):
    if (i % 3) != 0:
        t.set_visible(False)
        
plt.xticks(rotation=90)

# additionally we can set chart title, legend and axes labels
plt.title('COVID-19 data results')
plt.legend(['from 2020-07-25 to 2020-08-25'])
plt.xlabel('Countries')
plt.ylabel('cases vs. test ratio in %')

plt.show()




