#!/usr/bin/env python
# coding: utf-8

# 

# 

# In[48]:


import numpy as np
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from zipfile import ZipFile
import seaborn as sns
rows = 10
columns = 15

x,y = np.meshgrid(np.linspace(-1,1, columns), np.linspace(-1,1,rows))
d = np.sqrt(x*x*y*y)
sigma = 0.5
myGaussianDisc = (8*np.exp(-((d)**2/(2.0 * sigma**2)))).astype('uint8')
display(myGaussianDisc)


# 2. Then extract the value of the element at x=3 and y=5.

# In[49]:


x = 3
y = 5
display(myGaussianDisc[x,y])


# 

# 

# 

# 

# In[50]:


f,axes = plt.subplots(1,2)
(aGrayscale, aHeatmap)= axes.flatten()

aGrayscale.imshow(myGaussianDisc, cmap= "gray")
aGrayscale.set_axis_off()
aGrayscale.set_title("Grayscale Image")

sns.heatmap(myGaussianDisc, cmap = "viridis", square=True, annot=True, ax=aHeatmap)
aHeatmap.set_ylim(0,10)
aHeatmap.invert_yaxis()
aHeatmap.set_ylabel('x')
aHeatmap.set_ylabel('x')
aHeatmap.set_title("Heatmap Image")


# - For color images, we have to create a 3-channel Gaussian disc's array with values somewhere in the range of 0 and 8 with 10 rows and 15 columns.

# Create gray, gray-scaled, Viridis, Hot, diverging colormap, and HSV not perceptually uniform plot accordingly.

# In[51]:


rows =10
columns = 15
x,y = np.meshgrid(np.linspace(-1,1, columns), np.linspace(-1,1,rows))
d = np.sqrt(x*x+y*y)
sigma = 0.5
my12BitArray = ((2**12-1)*np.exp(-((d)**2/(2.0*sigma**2)))).astype('uint16')

f,axes = plt.subplots(2,3)
(aG, aS, aV, aH, aRB,aHSV) = axes.flatten()

aG.imshow(my12BitArray, cmap="gray", vmin=0, vmax=(2**16)-1)
aG.set_axis_off()
aG.set_title("Gray")

aG.imshow(my12BitArray, cmap="gray", vmin=0, vmax=(2**16)-1)
aG.set_axis_off()
aG.set_title("Gray")

aS.imshow(my12BitArray, cmap="gray")
aS.set_axis_off()
aS.set_title("Gray-scaled")

aV.imshow(my12BitArray, cmap="viridis")
aV.set_axis_off()
aV.set_title("Viridis")

aH.imshow(my12BitArray, cmap="hot")
aH.set_axis_off()
aH.set_title("Hot-squential-colormap")

aRB.imshow(my12BitArray, cmap="RdBu")
aRB.set_axis_off()
aRB.set_title("diverging colormap")

aHSV.imshow(my12BitArray, cmap="hsv")
aHSV.set_axis_off()
aHSV.set_title("HSV (not perceptually uniform)")

plt.show()


# In[52]:


import requests
images = requests.get("http://www.fil.ion.ucl.ac.uk/spm/download/data/attention/attention.zip")



# In[53]:


import zipfile
from io import BytesIO

zipstream = BytesIO(images.content)
zf = zipfile.ZipFile(zipstream)


# In[54]:


from nibabel import FileHolder
from nibabel.analyze import AnalyzeImage

header = BytesIO(zf.open('attention/structural/nsM00587_0002.hdr').read())
image = BytesIO(zf.open('attention/structural/nsM00587_0002.img').read())
img = AnalyzeImage.from_file_map({'header': FileHolder(fileobj=header), 'image': FileHolder(fileobj=image)})
arr = img.get_fdata()
arr.shape


# In[55]:


plt.imshow(arr[:,:,5])
plt.colorbar()
plt.plot()


# In[56]:


import geoplot as gplt
import geopandas  as gpd
import geoplot.crs as gcrs
import imageio
import pandas as pd
import pathlib
import matplotlib.animation as animation
import mapclassify as mc
import  pycountry
import plotly.express as px


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


rows = 10
columns = 15

x,y = np.meshgrid(np.linspace(-1,1, columns), np.linspace(-1,1,rows))
d = np.sqrt(x*x*y*y)
sigma = 0.5
disc = (8*np.exp(-((d)**2/(2.0 * sigma**2)))).astype('uint')
myRGBColorArray = np.stack([disc,np.roll(disc,2, axis=0), np.roll(disc,2,axis=1)], axis=2)
print("Red:")
display(myRGBColorArray[:,:,1])


# In[ ]:





# In[58]:


usa = gpd.read_file("us_state/us_state.shp")
print(usa.head())


# In[ ]:


state_pop = pd.read_csv("maps/us_state_est_population.csv")
print(state_pop.head)


# In[ ]:


pop_states = usa.merge(state_pop, left_on="NAME", right_on="NAME")
pop_states.head()


# In[ ]:


pop_states[pop_states.NAME=="California"].plot()


# In[ ]:


path = gplt.datasets.get_path("contiguous_usa")
contiguous_usa = gpd.read_file(path)
contiguous_usa.head()


# In[ ]:


gplt.polyplot(contiguous_usa)


# In[ ]:


path = gplt.datasets.get_path("usa_cities")
usa_cities = gpd.read_file(path)
usa_cities.head()


# In[ ]:


continental_usa_cities = usa_cities.query('STATE not in ["HI", "AK", "PR"]')
gplt.pointplot(continental_usa_cities)


# In[ ]:


ax = gplt.polyplot(contiguous_usa)
gplt.pointplot(continental_usa_cities, ax = ax)


# In[ ]:


ax =  gplt.polyplot(contiguous_usa, projection = gcrs.AlbersEqualArea())
gplt.pointplot(continental_usa_cities, ax =  ax)


# In[ ]:


ax = gplt.polyplot(contiguous_usa, projection = gcrs.AlbersEqualArea())
gplt.pointplot(
    continental_usa_cities,
    ax=ax,
    hue="ELEV_IN_FT",
    legend=True
)


# In[ ]:


ax = gplt.polyplot(
    contiguous_usa,
    edgecolor="white",
    facecolor = "lightgray",
    figsize= (12,8),
    projection=gcrs.AlbersEqualArea()
)

gplt.pointplot(
    continental_usa_cities,
    ax=ax,
    hue = "ELEV_IN_FT",
    cmap = "Blues",
    scheme= "quantiles",
    scale =  "ELEV_IN_FT",
    limits=(1,10),
    legend=True,
    legend_var="scale",
    legend_kwargs={"frameon": False},
    legend_values=[-110, 1750, 3600, 5500, 7400],
    legend_labels=["-1100 feet", "1750 feet", "3600 feet", "5500 feet", "7400 feet"]
)

ax.set_title("Cities in the continental US, by elevation", fontsize = 16)








# 

# In[ ]:


ax = gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea() )
gplt.choropleth(
    contiguous_usa,
    hue="population",
    edgecolor ="white",
    linewidth = 1,
    cmap = "Greens",
    legend=True,
    scheme = "FisherJenks",
    legend_labels=[
        "<3 million", "3-6.7 million", "6.7-12.8 million",
        "12.8-25 million", "25-37 million"
    ],
    projection=gcrs.AlbersEqualArea(),
    ax=ax

)


# In[ ]:


obesity = pd.read_csv(gplt.datasets.get_path("obesity_by_state"), sep ="\t")
obesity.head()



# In[ ]:


geo_obesity =  contiguous_usa.set_index("state").join(obesity.set_index("State"))
geo_obesity.head()


# In[ ]:


gplt.cartogram(
    geo_obesity,
    scale="Percent",
    projection = gcrs.AlbersEqualArea()
)


# In[ ]:


melbourne = gpd.read_file(gplt.datasets.get_path("melbourne"))
df = gpd.read_file(gplt.datasets.get_path("melbourne_schools"))
melbourne_primary_schools = df.query('School_Type == "Primary"')



ax = gplt.voronoi(
    melbourne_primary_schools,
    clip = melbourne,
    linewidth = 0.5,
    edgecolor="white",
    projection = gcrs.Mercator()
)


gplt.polyplot(
    melbourne,
    edgecolor ="None",
    facecolor="lightgray",
    ax=ax

)

gplt.pointplot(
    melbourne_primary_schools,
    color="black",
    ax=ax,
    s=1,
    extent = melbourne.total_bounds
)
plt.title("Primary Schools in Greater Melbourne, 2018")


# In[ ]:


#load the datafram
df_confirmedGlobal= pd.read_csv('time_series_covid19_confirmed_global.csv')
print(df_confirmedGlobal.head())


# In[ ]:


#Clean the Dataset

df_confirmedGlobal= df_confirmedGlobal.drop(columns={'Province/State', 'Lat', 'Long'})
df_confirmedGlobal = df_confirmedGlobal.groupby('Country/Region').agg('sum')
date_list= list(df_confirmedGlobal.columns)

#Get the three-letter country code for each country
def get_country_code(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

df_confirmedGlobal['Country'] = df_confirmedGlobal.index
df_confirmedGlobal['iso_alpha_3'] = df_confirmedGlobal['Country'].apply(get_country_code)

#Transform the dataset in a long format
df_long =pd.melt(df_confirmedGlobal, id_vars=['Country', 'iso_alpha_3'], value_vars= date_list)
print(df_long)



# In[61]:


# create a map animation with Plotly Express

fig= px.choropleth(df_long,
                  locations='iso_alpha_3',
                  color='value',
                  hover_name='Country',
                  animation_frame='variable',
                  projection='natural earth',
                  color_continuous_scale='Peach',
                  range_color=[0, 50000]
                  )

fig.show()
fig.write_html('Covid19_map.html') #Write it to html file


# ## Questions

# In[ ]:


rows = 10
columns = 15

x,y = np.meshgrid(np.linspace(-1,1, columns), np.linspace(-1,1,rows))
d = np.sqrt(x*x*y*y)
sigma = 0.5
disc = (8*np.exp(-((d)**2/(2.0 * sigma**2)))).astype('uint')
myRGBColorArray = np.stack([disc,np.roll(disc,2, axis=0), np.roll(disc,2,axis=1)], axis=2)
print("Red:")
print(disc)


f,axes = plt.subplots(1,4)
(red, blue, green, composite) = axes.flatten()

red.imshow(myRGBColorArray[:,:,0], cmap="Reds_r")
red.set_axis_off()
red.set_title("Reds")

green.imshow(myRGBColorArray[:,:,0], cmap="Greens")
green.set_axis_off()
green.set_title("Green")

blue.imshow(myRGBColorArray[:,:,0], cmap="Blues")
blue.set_axis_off()
blue.set_title("Blue")

composite.imshow(myRGBColorArray[:,:,0])
composite.set_axis_off()
composite.set_title('Composite')


# Quantile are cut-off points that partition a set number of observations into nearly equal sized subsets. 
# it basically split the data into groups that contain the same number of data points.
# So for example in this exercise were we used geoplot to show population of states within different brackers. The more population, the darker the color of the state ("<3 million", "3-6.7 million", "6.7-12.8 million",
#         "12.8-25 million", "25-37 million"). This data have 5 quantiles and show 5 different colors based on their population. If we would change the quantile to 2 for example, we could split the data into two different dataset. "<18 million and >18 million". THis means that all states with less than 18 million would be one color, while above would have another. More or less classifying the different states.
# in short, its a tool that enables different visualtions and makes you able to choose the amount of groups you want to divide the dataset into

# In[ ]:


ax = gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea(),figsize= (12,8), )
quantiles = mc.Quantiles(continental_usa_cities["POP_2010"], k=10)


gplt.pointplot(
    continental_usa_cities,
    hue="POP_2010",
    edgecolor ="white",
    linewidth = 1,
    cmap = "Reds",
    legend=True,
    scheme = quantiles,
    legend_labels=[
        "10008.00 - 11412.00 peoples", "11412 - 13227 peoples", "13227 - 15435 peoples",
        "15435 - 18383 peoples", " 18383 - 22308 peoples","  22308 - 27537  peoples"," 27537 - 35814 peoples","35814 - 51878 peoples","51878 - 85050 peoples","85050 - 8175133 peoples"
    ],
    projection=gcrs.AlbersEqualArea(),
    ax=ax
)


# In[ ]:


ax = gplt.voronoi(
    contiguous_usa,
    edgecolor="white",
    facecolor = "lightgray",
    figsize= (12,8),
    projection=gcrs.AlbersEqualArea()
)

gplt.voronoi(
    continental_usa_cities,
    ax=ax,
    hue = "ELEV_IN_FT",
    cmap = "Blues",
    scheme= "quantiles",
    scale =  "ELEV_IN_FT",
    limits=(1,10),
    legend=True,
    legend_var="scale",
    legend_kwargs={"frameon": False},
    legend_values=[-110, 1750, 3600, 5500, 7400],
    legend_labels=["-1100 feet", "1750 feet", "3600 feet", "5500 feet", "7400 feet"]
)

ax.set_title("Cities in the continental US, by elevation", fontsize = 16)


# In[ ]:


#load the datafram
df_recoveredGlobal= pd.read_csv('time_series_covid19_recovered_global.csv')
print(df_recoveredGlobal.head())

#Clean the Dataset

df_recoveredGlobal= df_recoveredGlobal.drop(columns={'Province/State', 'Lat', 'Long'})
df_recoveredGlobal = df_recoveredGlobal.groupby('Country/Region').agg('sum')
date_list= list(df_recoveredGlobal.columns)

#Get the three-letter country code for each country
def get_country_code(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

df_recoveredGlobal['Country'] = df_recoveredGlobal.index
df_recoveredGlobal['iso_alpha_3'] = df_recoveredGlobal['Country'].apply(get_country_code)

#Transform the dataset in a long format
df_long =pd.melt(df_recoveredGlobal, id_vars=['Country', 'iso_alpha_3'], value_vars= date_list)
print(df_long)

# create a map animation with Plotly Express

fig= px.choropleth(df_long,
                  locations='iso_alpha_3',
                  color='value',
                  hover_name='Country',
                  animation_frame='variable',
                  projection='natural earth',
                  color_continuous_scale='Peach',
                  range_color=[0, 50000]
                  )

fig.show()
fig.write_html('Covid19_map.html') #Write it to html file


# In[ ]:


#load the datafram
df_deathGlobal= pd.read_csv('time_series_covid19_deaths_global.csv')
print(df_deathGlobal.head())


#Clean the Dataset

df_deathGlobal= df_deathGlobal.drop(columns={'Province/State', 'Lat', 'Long'})
df_deathGlobal = df_deathGlobal.groupby('Country/Region').agg('sum')
date_list= list(df_deathGlobal.columns)

#Get the three-letter country code for each country
def get_country_code(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

df_deathGlobal['Country'] = df_deathGlobal.index
df_deathGlobal['iso_alpha_3'] = df_deathGlobal['Country'].apply(get_country_code)

#Transform the dataset in a long format
df_long =pd.melt(df_deathGlobal, id_vars=['Country', 'iso_alpha_3'], value_vars= date_list)
print(df_long)

# create a map animation with Plotly Express

fig= px.choropleth(df_long,
                  locations='iso_alpha_3',
                  color='value',
                  hover_name='Country',
                  animation_frame='variable',
                  projection='natural earth',
                  color_continuous_scale='Peach',
                  range_color=[0, 50000]
                  )

fig.show()
fig.write_html('Covid19_map.html') #Write it to html file


# In[ ]:


rows =10
columns = 15
x,y = np.meshgrid(np.linspace(-1,1, columns), np.linspace(-1,1,rows))
d = np.sqrt(x*x+y*y)
sigma = 0.5
my12BitArray = ((2**12-1)*np.exp(-((d)**2/(2.0*sigma**2)))).astype('uint16')
print(my12BitArray)
f,axes = plt.subplots(2,3)
(aG, aS, aV, aH, aRB,aHSV) = axes.flatten()

aG.imshow(my12BitArray, cmap="gray", vmin=0, vmax=(2**16)-1)
aG.set_axis_off()
aG.set_title("Gray")

aG.imshow(my12BitArray, cmap="gray", vmin=0, vmax=(2**16)-1)
aG.set_axis_off()
aG.set_title("Gray")

aS.imshow(my12BitArray, cmap="gray")
aS.set_axis_off()
aS.set_title("Gray-scaled")

aV.imshow(my12BitArray, cmap="viridis")
aV.set_axis_off()
aV.set_title("Viridis")

aH.imshow(my12BitArray, cmap="hot")
aH.set_axis_off()
aH.set_title("Hot-squential-colormap")

aRB.imshow(my12BitArray, cmap="RdBu")
aRB.set_axis_off()
aRB.set_title("diverging colormap")

aHSV.imshow(my12BitArray, cmap="hsv")
aHSV.set_axis_off()
aHSV.set_title("HSV (not perceptually uniform)")

plt.show()


# In[ ]:





# In[ ]:




