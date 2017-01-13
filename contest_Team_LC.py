#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:02:33 2017

project for Harvard IACS computefest 2017

@author: chenyuelu
"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import math
import re

pd.set_option("display.max_rows",10)


#%%
df = pd.DataFrame()
cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
        'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points','Wilderness_Area',
        'Soil_Type','Cover_Type']
df = pd.DataFrame(columns = cols)

elevation = []
aspect = []
slope = []
horizontal_Distance_To_Hydrology = []
vertical_Distance_To_Hydrology = []
horizontal_Distance_To_Roadways = []
hillshade_9am = []
hillshade_Noon = []
hillshade_3pm = []
horizontal_Distance_To_Fire_Points = []
wilderness_Area = []
soil_Type = []
cover_Type = []

#%%
with open('/Users/chenyuelu/Desktop/computefest/Python/contest/covtype.data', "rt") as f:   
    for line in f:
        elevation.append(int(line.split(',')[0]))
        aspect.append(int(line.split(',')[1]))
        slope.append(int(line.split(',')[2]))
        horizontal_Distance_To_Hydrology.append(int(line.split(',')[3]))
        vertical_Distance_To_Hydrology.append(int(line.split(',')[4]))
        horizontal_Distance_To_Roadways.append(int(line.split(',')[5]))
        hillshade_9am.append(int(line.split(',')[6]))
        hillshade_Noon.append(int(line.split(',')[7]))
        hillshade_3pm.append(int(line.split(',')[8]))
        horizontal_Distance_To_Fire_Points.append(int(line.split(',')[9]))
        # The following three categories are categorical, and I assigned string type to them
        # wilderness_Area returns the index of the area
        wilderness_Area.append(str(line.split(',')[10:14].index('1')+ 1))
        # soil_Type returns the index of the soil type
        soil_Type.append(str(line.split(',')[14:54].index('1')+1))
        cover_Type.append(re.findall('\d',line.split(',')[-1])[0])

df.Elevation = elevation
df.Aspect = aspect 
df.Slope = slope
df.Horizontal_Distance_To_Hydrology = horizontal_Distance_To_Hydrology
df.Vertical_Distance_To_Hydrology = vertical_Distance_To_Hydrology
df.Horizontal_Distance_To_Roadways = horizontal_Distance_To_Roadways
df.Hillshade_9am = hillshade_9am
df.Hillshade_Noon = hillshade_Noon
df.Hillshade_3pm = hillshade_3pm
df.Horizontal_Distance_To_Fire_Points = horizontal_Distance_To_Fire_Points
df.Wilderness_Area = wilderness_Area
df.Soil_Type = soil_Type
df.Cover_Type = cover_Type

df.to_csv('/Users/chenyuelu/Desktop/computefest/Python/contest/dataframe.csv')   


#%%
# violinplot
sns.set_style("whitegrid")

oFig1 = plt.figure()
oFig1.set_size_inches(10.5, 18.5)

oFig1.add_subplot(521)
sns.violinplot(x='Cover_Type', y ='Elevation',scale = 'area', data = df)
plt.title('Elevation vs Cover_Type')
plt.xlabel('Cover_Type')
plt.ylabel('Elevation (m)')

oFig1.add_subplot(522)
sns.violinplot(x='Cover_Type', y ='Slope',scale = 'area', data = df)
plt.title('Slope vs Cover_Type')
plt.xlabel('Cover_Type')
plt.ylabel('Slope (degree)')

oFig1.add_subplot(523)
sns.violinplot(x='Cover_Type', y ='Aspect',scale = 'area',data = df)
plt.title('Aspect vs Cover_Type')
plt.xlabel('Cover_Type')
plt.ylabel('Aspect (degree)')

oFig1.add_subplot(524)
sns.violinplot(x='Cover_Type', y ='Horizontal_Distance_To_Hydrology',scale = 'area',data = df)
plt.title('Horizontal_Distance_To_Hydrology vs Cover_Type')
plt.xlabel('Cover_Type')
plt.ylabel('Distance (m)')

oFig1.add_subplot(525)
sns.violinplot(x='Cover_Type', y ='Vertical_Distance_To_Hydrology',scale = 'area',data = df)
plt.title('Vertical_Distance_To_Hydrology vs Cover_Type')
plt.xlabel('Cover_Type')
plt.ylabel('Distance (m)')

oFig1.add_subplot(526)
sns.violinplot(x='Cover_Type', y ='Horizontal_Distance_To_Roadways',scale = 'area',data = df)
plt.title('Horizontal_Distance_To_Roadways vs Cover_Type')
plt.xlabel('Cover_Type')
plt.ylabel('Distance (m)')

oFig1.add_subplot(527)
sns.violinplot(x='Cover_Type', y ='Hillshade_9am',scale = 'area',data = df)
plt.title('Hillshade_9am vs Cover_Type')
plt.xlabel('Cover_Type')
plt.ylabel('Hillshade (degree)')

oFig1.add_subplot(528)
sns.violinplot(x='Cover_Type', y ='Hillshade_Noon',scale = 'area',data = df)
plt.title('Hillshade_Noon vs Cover_Type')
plt.xlabel('Cover_Type')
plt.ylabel('Hillshade (degree)')

oFig1.add_subplot(529)
sns.violinplot(x='Cover_Type', y ='Hillshade_3pm',scale = 'area',data = df)
plt.title('Hillshade_3pm vs Cover_Type')
plt.xlabel('Cover_Type')
plt.ylabel('Hillshade (degree)')

oFig1.add_subplot(5,2,10)
sns.violinplot(x='Cover_Type', y ='Horizontal_Distance_To_Fire_Points',scale = 'area',data = df)
plt.title('Horizontal_Distance_To_Fire_Points vs Cover_Type')
plt.xlabel('Cover_Type')
plt.ylabel('Distance (m)')

plt.tight_layout()
oFig1.savefig('/Users/chenyuelu/Desktop/computefest/Python/contest/violinPlot')

#%%
# stacked bar graph 1
N1 = 7
ind1 = np.arange(N1)    
Area = []
labels1 = list(sorted(df.Cover_Type.unique()))
# count the number of occurences of cover type by wilderness area
for area_code in sorted(df.Wilderness_Area.unique()):
    area = []
    for i in range(len(df.Cover_Type.unique())):
        i += 1
        area.append(df[(df.Wilderness_Area==area_code) & (df.Cover_Type == str(i))].count()[0])
    Area.append(area)

f, ax1 = plt.subplots(1, figsize=(10,5))

bar_width = 0.75
tick_pos = [n+(bar_width/2) for n in ind1] 

# Create a bar plot, in position bar_1
ax1.bar(ind1, 
        # using the Area[0] data
        Area[0], 
        # set the width
        width=bar_width,
        # with the label AREA_1
        label='AREA_1', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='red')

ax1.bar(ind1, Area[1], width=bar_width, bottom=Area[0], label='AREA_2', alpha=0.5, color='yellow')

ax1.bar(ind1, Area[2],width=bar_width, 
        bottom=[j+k for j,k in zip(Area[0],Area[1])], 
        label='AREA_3', alpha=0.5, color='green')

ax1.bar(ind1,Area[3],width=bar_width,
        bottom=[j+k+l for j,k,l in zip(Area[0],Area[1],Area[2])], 
        label='AREA_4', alpha=0.5, color='blue')

# set the x ticks with names
plt.xticks(tick_pos, labels1)

# Set the label and legends
ax1.set_ylabel("Count")
ax1.set_xlabel("Cover_Type")
ax1.set_title("Differential Cover Type Distribution by Wilderness Area",fontsize = 18,y=1.01)
plt.legend(loc='upper right',fontsize = 13, frameon = True)
# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
fig1 = ax1.get_figure()
fig1.savefig("/Users/chenyuelu/Desktop/computefest/Python/contest/stackedBars1")

#%%

# stacked bar graph 2
N2 = 4
ind2 = np.arange(N2)    
Cover = []
labels2 = list(sorted(df.Wilderness_Area.unique()))

for cover_number in sorted(df.Cover_Type.unique()):
    cover = []
    for i in range(len(df.Wilderness_Area.unique())):
        i += 1
        cover.append(df[(df.Wilderness_Area==str(i)) & (df.Cover_Type == cover_number)].count()[0])
    Cover.append(cover)

f, ax2 = plt.subplots(1, figsize=(8,5))

bar_width = 0.5

tick_pos = [n+(bar_width/2) for n in ind2] 

ax2.bar(ind2, Cover[0], width=bar_width, label='COVER_TYPE_1', alpha=0.5,color='red')

ax2.bar(ind2, Cover[1], width=bar_width,bottom=Cover[0], label='COVER_TYPE_2', alpha=0.5, color='orange')

ax2.bar(ind2, Cover[2], width=bar_width,
        bottom=[j+k for j,k in zip(Cover[0],Cover[1])], 
        label='COVER_TYPE_3', alpha=0.5, color='yellow')

ax2.bar(ind2, Cover[3],width=bar_width,
        bottom=[j+k+l for j,k,l in zip(Cover[0],Cover[1],Cover[2])], 
        label='COVER_TYPE_4', alpha=0.5, color='green')

ax2.bar(ind2, Cover[4],width=bar_width,
        bottom=[j+k+l+m for j,k,l,m in zip(Cover[0],Cover[1],Cover[2],Cover[3])], 
        label='COVER_TYPE_5', alpha=0.5, color='blue')

ax2.bar(ind2, Cover[5], width=bar_width,
        bottom=[j+k+l+m+n for j,k,l,m,n in zip(Cover[0],Cover[1],Cover[2],Cover[3],Cover[4])], 
        label='COVER_TYPE_6', alpha=0.5, color='purple')

ax2.bar(ind2, Cover[6],width=bar_width,
        bottom=[j+k+l+m+n+o for j,k,l,m,n,o in zip(Cover[0],Cover[1],Cover[2],Cover[3],Cover[4],Cover[5])], 
        label='COVER_TYPE_7', alpha=0.5, color='black')

plt.xticks(tick_pos, labels2)

ax2.set_ylabel("Cover_Type")
ax2.set_xlabel("Count")
ax2.set_title("Differential Cover Type Distribution in Wilderness Areas",fontsize = 18,y=1.01)
plt.legend(loc='upper right',frameon = True)
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
fig2 = ax2.get_figure()
fig2.savefig("/Users/chenyuelu/Desktop/computefest/Python/contest/stackedBars2")


#%%
# heatmap to show the correlation between soil type and cover type 

SoilLog = []
for soil_type in sorted(df.Soil_Type.unique()):
    soilLog = []
    for i in range(len(df.Cover_Type.unique())):
        i += 1
        if df[(df.Soil_Type==soil_type) & (df.Cover_Type == str(i))].count()[0] == 0:
            soilLog.append(0)
        else:
            soilLog.append(math.log(df[(df.Soil_Type==soil_type) & (df.Cover_Type == str(i))].count()[0],10))
    SoilLog.append(soilLog)
snsheatmap = sns.heatmap(SoilLog, cmap=plt.cm.Blues, linewidths=.1)
plt.xlabel("Cover Type",fontsize = 15,y=1.01)
plt.ylabel("Soil Type",fontsize = 15,x=1.01)
plt.title("Differential Cover Type Distribution by Soil Type",fontsize = 20,y=1.01)

figHeat= snsheatmap.get_figure()
figHeat.set_size_inches(10, 15)
figHeat.savefig('/Users/chenyuelu/Desktop/computefest/Python/contest/heatmap')

#%%