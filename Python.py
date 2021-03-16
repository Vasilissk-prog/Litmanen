# As the code is produced during a training programm I would suggest that it should be carefully structured and commented
# For this reason I have applied the following changes

#Import Libraries
import pandas as pd
# Check version
print(pd.__version__)

# Create lists with Data
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

# Combine two lists and create a pandas dataframe
cities_population_dataframe=pd.DataFrame({ 'City name': city_names, 'Population': population })

# See basic statistics for this dataframe and store them into another
cities_population_dataframe_desc = cities_population_dataframe.describe()

# Read comma seperated file from web and store it as dataframe
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

# Combine two lists and create a pandas dataframe (same as cities_population_dataframe)
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })

# Add new column to Dataframe
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])

# Create new variables based on column filtering
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
cities['Is san francisco'] = (cities['City name'] == 'San Francisco')

# Print dataframe
print(cities)
# Reindex based on 3rd, 1st, 2nd column and print dataframe
print(cities.reindex([2, 0, 1]))
