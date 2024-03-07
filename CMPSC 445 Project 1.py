# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import category_encoders as ce


# %%
df = pd.read_csv('data.csv') #import the data set

# %%
df.shape

# %%
df.head() # Preprocess the data set

# %%
df.columns



# %%
percentmissing = df.isnull().mean() * 100 # taking the inital data set finding all the columns that have a missing percent greater than 20%
missingcolumns = percentmissing[percentmissing > 20].index.tolist()
missingcolumns


# %%
missing = df.isnull().sum() # the graph shows how the data is spread, however the attributes are overlapping and cumbersome, making it harder to read. To fix this, we will remove a large portion of these columns
missing = missing[missing >= 100]
missing.sort_values(inplace=True)
missing.plot.barh()
plt.title("Features with Missing Attributes > 100")
plt.ylabel("Features")
plt.xlabel("Amount of Missing Attributes")


# %%
DROP_DATA = """
basements
central_air
cross_reference
date_exterior_condition
fuel
garage_type
house_extension
mailing_address_1
mailing_address_2
mailing_care_of
market_value_date
number_of_rooms
other_building
owner_2
separate_utilities
sewer
site_type
street_direction
suffix
type_heater
unfinished
unit
utility
year_built_estimate
"""

def dataclean(df):   
    df = df.drop(columns=DROP_DATA.split()) # gets rid of the columns with more than 20% missing attributes
    return df

df_dropped_missing = dataclean(df.copy())
df_dropped_missing.head()

# %%
# Get all unique values of (category_code, category_code_description)
df_dropped_missing[['category_code', 'category_code_description']].drop_duplicates().sort_values(by='category_code')

# %% [markdown]
# Remove non-residential homes
# 
# We can see that there are several types of buildings in this dataset. Since we are really only interested in the housing market we can really afford to drop all but the single and multi family.

# %%
def dataclean(df_dropped_missing):
    # Filter rows based on column: 'category_code' This keeps single family and multi-family
    df_dropped_missing = df_dropped_missing[df_dropped_missing['category_code'] <= 2]
    return df_dropped_missing
df_homes = dataclean(df_dropped_missing.copy())
df_homes.head()

# %%

def dataclean(df_homes): # drop unrelated data and columns

    df_homes = df_homes.drop(columns=['the_geom', 'the_geom_webmercator', 'assessment_date', 'beginning_point', 'book_and_page', 'building_code', 'building_code_description', 'category_code_description', 'general_construction', 'house_number', 'location', 'mailing_city_state', 'mailing_street', 'mailing_zip', 'owner_1', 'parcel_number', 'quality_grade', 'recording_date', 'registry_number', 'sale_date', 'state_code', 'street_designation', 'street_name', 'pin', 'building_code_description_new', 'building_code_new', 'objectid'])
    return df_homes

df_homes_clean = dataclean(df_homes.copy())
df_homes_clean.head()

# %%
df_homes_clean.dtypes

# %%
missing = df_homes_clean.isnull().sum()
missing = missing[missing >= 1000]
missing.sort_values(inplace=True)
missing.plot.barh()
plt.title("Features with Missing Attributes > 100")
plt.ylabel("Features")
plt.xlabel("Amount of Missing Attributes")



# %%
def dataclean(df_homes_clean): #drop additional missing attributes from each row since they are less than 6%
    
    df_homes_clean = df_homes_clean.dropna()
    
    df_homes_clean = df_homes_clean.drop_duplicates()
    return df_homes_clean

df_homes_clean_2 = dataclean(df_homes_clean.copy())
df_homes_clean_2.head()



# %%
df_homes_clean_2.loc[:, "lng"] = df_homes_clean_2['lng'].abs() # calculate the ABS longitude



# %%
df_homes_clean_2.dtypes.value_counts()

# %%

ONE_HOT = [ # setting up One-Hot Encoding labels
	'category_code',
	'parcel_shape',
	'topography',
	'view_type',

]
BINARY_ENCODING = [ #converting the labels into binary
	'zoning'
]

MAKE_BINARY = [ #taking the binary and making it into categories
	'homestead_exemption',
	'exempt_building',
	'exempt_land'
]



# %%
def dataclean(df_homes_clean_2): # converting the columns into boolean encoding

    df_homes_clean_2 = df_homes_clean_2.astype({'exempt_building': 'bool'})

    df_homes_clean_2 = df_homes_clean_2.astype({'exempt_land': 'bool'})

    df_homes_clean_2 = df_homes_clean_2.astype({'homestead_exemption': 'bool'})
    return df_homes_clean_2

df_encode = dataclean(df_homes_clean_2.copy())
df_encode.head()



# %%

df_encode_2 = pd.get_dummies(df_encode, columns=ONE_HOT) # encoding the columns for One-Hot Encoding
df_encode_2.head()



# %%

encoder = ce.BinaryEncoder(cols=['zoning']) # initializing the Binary Encoder


df_encoded = encoder.fit_transform(df_encode_2['zoning']) # transform the Binary Encoder into columns


df_binencode = pd.concat([df_encode_2.drop('zoning', axis=1), df_encoded], axis=1) # concat with the dataframe

df_binencode.head()



# %%
def remove_outliers(df): # remove the outliers using InterQuartile Range
    for column in df.select_dtypes(include=np.number).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lowbound = Q1 - 1.5 * IQR
        upbound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lowbound) & (df[column] <= upbound)]
    return df


df_outliers = remove_outliers(df_binencode)

df_outliers.head()




# %%
from sklearn.preprocessing import MinMaxScaler # scale the data using a MinMax Scaler

scaler = MinMaxScaler()


num_columns = df_outliers.select_dtypes(include=['int64', 'float64']).columns


df_outliers[num_columns] = scaler.fit_transform(df_outliers[num_columns])

df_scaled = df_outliers
df_scaled.head()



# %%

correlation = df_scaled.corr()

correlation['market_value'].sort_values(ascending=False)



# %%

X = df_scaled.drop(columns=['market_value', # setting up the data for the final prep
'parcel_shape_A',
'category_code_2',
'view_type_C',
'parcel_shape_B',
'view_type_0',
'topography_C',
'view_type_H',
'topography_B',
'view_type_B',
'topography_E',
'parcel_shape_C',
'parcel_shape_D',
'topography_D',
'view_type_E',
'street_code',
'view_type_D',
'category_code_1',
'lat',
'parcel_shape_E',
'view_type_I',
'topography_F'], axis=1)
y = df_scaled.market_value



# %%
from sklearn.decomposition import PCA # create an instance for PCA and transform the  data for it
PCA_VARIANCE = 0.8

pca = PCA(PCA_VARIANCE)
pca.fit(X)

X_pca = pca.transform(X)
print(X.shape)
X_pca.shape





# %%

from sklearn.model_selection import train_test_split # split the data for regression testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# %%

from sklearn.model_selection import train_test_split # split the data for PCA testing
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# %%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error



# %%

model = LinearRegression() # develop a Linear Regression model
model.fit(X_train, y_train)
score = model.score(X_train, y_train)
y_pred = model.predict(X_test)
rms = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"R^2 : {score}, RMS: {rms}, MAE: {mae}")



# %%

pca_model = LinearRegression() # develop a PCA Linear Regression model
pca_model.fit(X_pca_train, y_pca_train)
score_pca = pca_model.score(X_pca_test, y_pca_test)

y_pred = pca_model.predict(X_pca_test)
rms = np.sqrt(mean_squared_error(y_pca_test, y_pred))
mae = mean_absolute_error(y_pca_test, y_pred)
print(f"R^2 : {score_pca}, RMS: {rms}, MAE: {mae}")



# %%

tree_model = DecisionTreeRegressor() # develop a Decision Tree Regression model
tree_model.fit(X_train, y_train)
score_tree = tree_model.score(X_test, y_test)

y_pred = tree_model.predict(X_test)
rms = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"R^2 : {score_tree}, RMS: {rms}, MAE: {mae}")



# %%

pca_tree_model = DecisionTreeRegressor()  # develop a Decision Tree PCA Regression model
pca_tree_model.fit(X_pca_train, y_pca_train)
pca_score_tree = pca_tree_model.score(X_pca_test, y_pca_test)

y_pred = pca_tree_model.predict(X_pca_test)
rms = np.sqrt(mean_squared_error(y_pca_test, y_pred))
mae = mean_absolute_error(y_pca_test, y_pred)
print(f"R^2 : {pca_score_tree}, RMS: {rms}, MAE: {mae}")



# %%

forest_model = RandomForestRegressor() # develop a Random Forest Regression Model
forest_model.fit(X_train, y_train)
score_forest = forest_model.score(X_test, y_test)

y_pred = forest_model.predict(X_test)
rms = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"R^2 : {score_forest}, RMS: {rms}, MAE: {mae}")



# %%

pca_forest_model = RandomForestRegressor() # develop a Random Forest PCA Regression Model
pca_forest_model.fit(X_pca_train, y_pca_train)
pca_score_forest = pca_forest_model.score(X_pca_test, y_pca_test)

y_pred = pca_forest_model.predict(X_pca_test)
rms = np.sqrt(mean_squared_error(y_pca_test, y_pred))
mae = mean_absolute_error(y_pca_test, y_pred)
print(f"R^2 : {pca_score_forest}, RMS: {rms}, MAE: {mae}")



# %%

grad_model = GradientBoostingRegressor() # develop a Gradient Boosting Regression model
grad_model.fit(X_train, y_train)
score_grad = grad_model.score(X_test, y_test)

y_pred = grad_model.predict(X_test)
rms = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"R^2 : {score_grad}, RMS: {rms}, MAE: {mae}")



# %%

pca_grad_model = GradientBoostingRegressor() # develop a Gradient Boosting PCA Regression model
pca_grad_model.fit(X_pca_train, y_pca_train)
pca_score_grad = pca_grad_model.score(X_pca_test, y_pca_test)

y_pred = pca_grad_model.predict(X_pca_test)
rms = np.sqrt(mean_squared_error(y_pca_test, y_pred))
mae = mean_absolute_error(y_pca_test, y_pred)
print(f"R^2 : {pca_score_grad}, RMS: {rms}, MAE: {mae}")


