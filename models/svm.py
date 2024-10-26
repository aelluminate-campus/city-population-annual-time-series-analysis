import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

ds_url = 'https://datahub.io/core/population-city/r/unsd-citypopulation-year-fm.csv'

pop = pd.read_csv(ds_url)

print(pop.head())

# Deleting the Unneccessary
pop=pop.drop('Value Footnotes',axis=1)
pop=pop.drop('Reliability',axis=1)
pop=pop.drop('Record Type',axis=1)
pop=pop.drop('Area',axis=1)
pop=pop.drop('Source Year',axis=1)
pop = pop.dropna(axis=0)

# Renaming Columns
pop = pop.rename(columns={'Country or Area':'Country_or_Area', 'City type' : 'City_Type', 'Source Year' : 'Source_Year', 'Value' : 'Population'})

# Formatting
pop['Country_or_Area'] = pop['Country_or_Area'].str.capitalize()
pop['City_Type'] = pop['City_Type'].str.capitalize()
pop['City'] = pop['City'].str.capitalize()
pop['Population'] = pd.to_numeric(pop['Population'], errors='coerce').fillna(0).astype(int)
pop['Year'] = pd.to_numeric(pop['Year'], errors='coerce').fillna(0).astype(int)

print(pop.head())

#print(pop['Country_or_Area'].unique())

# Removing bad values
pop = pop[pop['Country_or_Area'] != '13']

#print(pop['Country_or_Area'].unique())

# Formatting Country Names
def clean_country_name(country):
    country = re.sub(r'\s*\(.*?\)\s*', '', country)
    country = country.replace('Å', 'A') 
    return country.strip()

pop['Country_or_Area'] = pop['Country_or_Area'].apply(clean_country_name)

#print(pop['Country_or_Area'].unique())

#print(pop['City'].unique())

# Cleaning City Names
special_chars_mapping = {
    'Å': 'A', 'Ä': 'A', 'Ö': 'O', 'Ü': 'U',
    'à': 'a', 'á': 'a', 'â': 'a', 'ä': 'a', 'æ': 'ae',
    'ç': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
    'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
    'ñ': 'n', 'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
    'ù': 'u', 'ú': 'u', 'û': 'u', 'ý': 'y', 'ÿ': 'y'
}

# Function to clean city names
def clean_city_name(city):
   # Remove text within parentheses
    city = re.sub(r'\s*\(.*?\)\s*', '', city)
    for special_char, replacement in special_chars_mapping.items():
        city = city.replace(special_char, replacement)
    return city.strip()

# Apply the cleaning function to the 'City' column
pop['City'] = pop['City'].apply(clean_city_name)

print(pop['City'].unique())

#print(pop['City_Type'].unique())

### Data Visualization ###

### Population Distribution by Years ###
pop_filtered = pop[(pop['Year'] >= 1972) & (pop['Year'] <= 2014)]
population_trend = pop_filtered.groupby('Year')['Population'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=population_trend, x='Year', y='Population')
plt.title('Population Distribution from 1972 to 2014')
plt.xlabel('Year')
plt.ylabel('Total Population')
plt.grid()
plt.xticks(rotation=45)
plt.show()

### Population Distribution by City Type ###
plt.figure(figsize=(10, 5))
sns.countplot(data=pop, x='City_Type', order=pop['City_Type'].value_counts().index)
plt.title('Number of Cities by City Type')
plt.xlabel('City Type')
plt.ylabel('Number of Cities')
plt.xticks(rotation=45)
plt.grid()
plt.show()

### Population Distribution by Sex ###
plt.figure(figsize=(8, 6))
sex_population_data = pop.groupby('Sex')['Population'].sum().reset_index()
sns.barplot(x='Sex', y='Population', data=sex_population_data)
plt.title('Total Population by Sex')
plt.xlabel('Sex')
plt.ylabel('Total Population')
plt.xticks(rotation=0)
plt.show()

### Population Distribution by Country (Top 20) ###
country_population = pop.groupby('Country_or_Area')['Population'].sum().reset_index()
country_population = country_population.sort_values(by='Population', ascending=False).head(20)  # Top 20 countries
plt.figure(figsize=(14, 8))
sns.barplot(data=country_population, x='Population', y='Country_or_Area', palette='viridis')
plt.title('Population Distribution for Each Country (Top 20)')
plt.xlabel('Total Population')
plt.ylabel('Country')
plt.grid(axis='x')
plt.show()

### Population Distribution by City (Top 25) ###
city_population = pop.groupby('City')['Population'].sum().reset_index()
top_cities_population = city_population.sort_values(by='Population', ascending=False).head(25)
plt.figure(figsize=(14, 8))
sns.barplot(data=top_cities_population, x='Population', y='City', palette='crest')
plt.title('Population Distribution for the Top 25 Cities')
plt.xlabel('Total Population')
plt.ylabel('City')
plt.grid(axis='x')
plt.show()

### Model Training ###

### Getting the Top 100 countries based on Population ###
top_cities = pop.groupby('City').agg({'Population': 'sum'}).reset_index()
top_100_cities = top_cities.nlargest(100, 'Population')
top_100_city_names = top_100_cities['City'].tolist()

pop_100 = pop[pop['City'].isin(top_100_city_names)]

### Model Traing ###
X = pop_100[['City_Type', 'Year', 'Country_or_Area', 'City']]  # Add any other relevant features
y = pop_100['Population']

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "XGBoost Regressor": xgb.XGBRegressor(random_state=42),
}

# Store results
results = {}

# Calculate the average population in the test set
avg_population = y_test.mean()

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    mae_percentage = (mae / avg_population) * 100  # Express MAE as a percentage of the average population
    r2 = r2_score(y_test, predictions) * 100  # R² Score as a percentage

    # Calculate accuracy percentage (100% - MAE percentage)
    accuracy = 100 - mae_percentage

    # Store results
    results[model_name] = {
        'Mean Absolute Error (%)': mae_percentage,
        'R^2 Score (%)': r2,
        'Accuracy (%)': accuracy  # Adding accuracy to results
    }

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results).T

# Display the results
print(results_df)