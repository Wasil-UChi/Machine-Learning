# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import geopandas
import os
import seaborn as sns

path = '/Users/wasilengel/Desktop/School/Harris/AP2/PSet 2'
labels_path = os.path.join(path, 'IAEPv2_0_2015labels.csv')
numbers_path = os.path.join(path, 'IAEPv2_0_2015numeric.csv')
wb_path = os.path.join(path, 'API_NE.CON.GOVT.ZS_DS2_en_csv_v2_2055671.csv')



## 3

df_1 = pd.read_csv(labels_path)

pd.set_option('max_columns', None) # displays all columns
df_1.head()

# df_2 = pd.read_csv(numbers_path)

# pd.set_option('max_columns', None) # displays all columns
# df_2.head()
# Since I don't want to go through the code notebook, I'll use the classified version of the df, which I named df_1



# a 

# Find unique values for lelecsystem, which classifies electoral systems

df_1["lelecsystem"].unique()

# Renaming column values according to instructions 

df_1["lelecsystem"] = df_1['lelecsystem'].replace({'Plurality (FPP)': 'plurality', 'Majority': 'majority', 'Proportional representation': 'proportional representation', 'Mixed systems': 'mixed', 'N/A - no legislature': 'missing', 'N/A - no elected legislature': 'missing', 'Missing information': 'missing'})

df_1["lelecsystem"].unique()

# Reducing df_1 to essential columns needed for this assignment: df_1_red

df_1_red = df_1[["cabr", "cname", "year", "lelecsystem", "parties"]]
df_1_red.head()

df_1_red_2011 = df_1_red[df_1_red["year"] == 2011]
# df_1_red_2011["year"].unique() # check that it works -> Yes, 2011 is the unique value
df_1_red_2011.head() # 163 rows that means we have data for 163 countries here

# Plot world map showing the varying electoral systems across the world

## Prepare

# First, merge geopandas data with our data: 'naturalearth_lowres' is geopandas datasets so I can use it directly
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# Reduce size to essential columns
world = world[["iso_a3", "name", "geometry"]]
world # 177 rows
# Rename the columns in world dataframe so that I can merge
world.columns=['iso_a3', 'cname', 'geometry']
merged = pd.merge(world, df_1_red_2011, on='cname')
merged.head() # 147 rows: yields more matches than when merging on cabr

## Plot 

fig, ax = plt.subplots(figsize=(20,20))
merged.plot(ax=ax, color='white', edgecolor='black')
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)
ax = merged.plot(ax=ax, column='lelecsystem', legend=True, cax=cax) 
ax.axis('off')
ax.set_title('Electoral Systems');



# b 

# Refer back to df_1_red_2011
# Focus on and identify unique values of "parties" column that captures 
# the number of parties with more than 5% of seats in the legislature

df_1_red_2011

df_1_red_2011["parties"].unique()

# Rename in order to have shorter legend
df_1_red_2011["parties"] = df_1['parties'].replace({'Two': "2", 'More than two': "2<", 'One': "1", 'Missing information': "N/A"})
df_1_red_2011

# Count occurences of parties per electoral system
stacked = df_1_red_2011.groupby(['lelecsystem', "parties"])["cabr"].count().reset_index(name="count")
stacked

# Draw a figure showing the relationship between parties and lelecsystem
# I want to display the number of parties (Y) per lelecsystem (X) using a stacked bar
# For lelecsystem, I excluded "Mixed" and "Missing" to prove Duverger's Law that focuses on majority/ plurality v. proportional
# I also summed majority and plurality displaying it in one column as to make a direct comparison with proportional
# For parties, I excluded "N/A" and "1" to prove Duverger's Law that focuses on two vs. more than two parties 

labels = ['Majority/ Plurality', 'Proportional Representation']
two = [7+4, 8]
twoplus = [19+9, 48]
# Drawing the numbers from the previous table 

fig, ax = plt.subplots()

plt.bar(labels, two, label = "2")
plt.bar(labels, twoplus, bottom = two, label = "2<")

ax.set_ylabel('Number of Parties')
ax.set_xlabel('Type of Electoral System')
ax.set_title('Worldwide: Number of Parties Per Electoral System (in numbers)')
ax.legend()

plt.show()

labels = ['Majority/ Plurality', 'Proportional Representation']
two = [11/39, 8/56]
twoplus = [28/39, 48/56]
# Standardizing numbers to percentages

fig, ax = plt.subplots()

plt.bar(labels, two, label = "2")
plt.bar(labels, twoplus, bottom = two, label = "2<")

ax.set_ylabel('Number of Parties')
ax.set_xlabel('Type of Electoral System')
ax.set_title('Worldwide: Number of Parties Per Electoral System (in per cent)')
ax.legend()

plt.show()



# c 

df_2 = pd.read_csv(wb_path,
                    header=2)
# Download in Safari because Chrome not working

df_2.head()

df_2.columns

df_2_red = df_2[['Country Name', 'Country Code', '2000', '2001', '2002', '2003', '2004',
       '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012']] # 264 rows
df_2_red = df_2_red.dropna() # 200 rows
df_2_red.head()

# Calculate average of government spending for each country between 2000 and 2012 

gdp = df_2_red.loc[: , '2000':'2012'] # select all columns with years: 2000 to 2012
df_2_red['mean'] = gdp.mean(axis=1) # calculate mean and create new column "mean"
df_2_red.head()

# df_2_red.columns

# Merge (first renaming merging column: Country Name)

df_2_red.columns=['cname', 'Country Code', '2000', '2001', '2002', '2003', '2004',
       '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', 'mean']
merged_2 = pd.merge(df_2_red, df_1_red_2011, on='cname')
merged_2.head() # 123 rows: yields more matches than when merging on cabr, still sufficiently large

# Show relationship using seaborn to show boxplot: lelecsystem v. mean

merged_2 = merged_2[merged_2["lelecsystem"] != "missing"]
# Filtering out missing values in lelecsystem: 119 rows

merged_2["lelecsystem"] = merged_2['lelecsystem'].replace({'proportional representation': 'proportional'})
# Renaming just to have prettier plot

ax = sns.boxplot("lelecsystem", "mean", data=merged_2)
ax.set_ylabel('Average Spending')
ax.set_xlabel('Type of Electoral System')
ax.set_title('Worldwide: Government Spending as a Percentage of GDP between 2000 and 2012')

