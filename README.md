# Philadelphia Housing Market Research

# Requirements
In order to use the data and get accurate results without error, download the CSV file from https://www.phila.gov/property/data/

# Data Collection
For this research project, the data was provided by the city of Philadelphia to analyze the housing market. Using the data.csv file, the dataset contains 81 columns and over 500,000 rows at a glance. The best way that I was able to convert this file was to use a pandas frame to analyze it. The various properties include attributes such as inspections, locations, sale information, and many others.

# Preprocessing
Considering the large amount of data and inconsistencies in the dataset, the preprocessing steps required having to drop columns and handling missing values. Converting data types to encode for One-Hot Encoding and Binary Encoding was useful as well since it removed outliers and created a consistent way to view the data. The first step required removing columns with substantial missing data and preserving columns with more than 80% of their data intact. Next, the focus was on the types of properties, and since the dataset included non-residential land, those attributes were removed in order to centralize the information properly. 

The data was then converted to account for each model required, which in this case was One-Hot Encoding and Binary Encoding. The categorical columns stayed the same, while some were converted to boolean to adjust for the encoding methods. One-Hot Encoding was useful for columns with a limited number of unique attributes, while Binary Encoding was better for multiple unique attributes since it reduced the dimensionality of the dataset. All outliers were identified in both encoders, and using the InterQuartile Range, the outliers were removed. The remaining dataset still had over 100,000 rows intact, which was then scaled using a MinMax Scaler from the sklearn module. All the attributes were placed in a consistent range for more accurate modeling.

# Feature Selection
The choices for feature selection were vast due to the large quantity of data provided from the preprocessed dataset. THrough various types of testing, the csv file retained the columns suitable for creating the regression models. Optimizing performance of the models required using a subset of the suitable columns based on correlation analysis with the purpose of utilizing all attributes present. Removing certain attributes created a marginal improvement in each overall performance of the models. Principal Component Analysis was used as an alternate approach for feature selection, since it selected a subset of the total features which could improve or diminish accuracy when compared to the original set. The overall feature selection involved experiment evaluation and correlation analysis to identify the relevant and informative attributes, however the manual selection was more effective than the PCA selection in enhancing the accuracy of the regression models for the dataset.

# Model Creation
After preprocessing the data and selecting the relevant attributes, I trained and evaluated various models on the preprocessed dataset. The models included Linear Regression, Decision Tree, Random Forest, and Gradient Boosting. For each model, I compared the performance of the models trained on the non-PCA features and the PCA features. The results are as follows:

Linear Regression:

Non-PCA: R^2: 0.9787, RMSE: 0.0216, MAE: 0.0077
PCA: R^2: 0.6921, RMSE: 0.0828, MAE: 0.0634
Decision Tree:

Non-PCA: R^2: 0.9960, RMSE: 0.0094, MAE: 0.0008
PCA: R^2: 0.9703, RMSE: 0.0257, MAE: 0.0110
Random Forest:

Non-PCA: R^2: 0.9977, RMSE: 0.0071, MAE: 0.0007
PCA: R^2: 0.9861, RMSE: 0.0176, MAE: 0.0071
Gradient Boosting:

Non-PCA: R^2: 0.9931, RMSE: 0.0124, MAE: 0.0041
PCA: R^2: 0.9330, RMSE: 0.0386, MAE: 0.0252
Across all models, the non-PCA features consistently outperformed the PCA features in terms of higher R-squared scores and lower RMSE and MAE values. The Random Forest model exhibited the best overall performance, with an R-squared score of 0.9977 and the lowest RMSE and MAE values.

It's important to note that if the dataset is obtained and cleaned in the same manner at a later time, the results may vary due to potential differences in the data or the specific preprocessing steps applied.

The evaluation metrics provide insights into the accuracy and predictive power of each model. A higher R-squared score closer to 1 (but not exactly 1) indicates a better fit between the model and the data. Additionally, lower RMSE and MAE values suggest that the model's predictions are closer to the actual values, with smaller errors.

# Conclusion
The project provided a nice learning experience in processing data and learning the steps in a more organized manner with hands-on learning. The most difficult part was selecting the models required for this and going through the preprocessing due to the large dataset. Removing a large quantity of data was required (some of which was converted), but this really showed me how to apply multiple data cleaning methods within the project. The feature selection phase went by faster since I had understood how to remove attributes and clean data from the above dataset. This along with the model selection phase was more relaxing than the preprocessing section, and allowed for a faster completion. 

Overall, this project increased my knowledge and skills on machine learning and regression models, along with using PCA as an alternative method for model selection. What made this more interesting was the large dataset of the city of Philadelphia, since it showed how to work with a real life scenario and developing something that could help in a given situation. In future projects, the process of machine learning will be easier to grasp due to this project's large levels of work, and it will create familiarity for me with more regression modeling and making better predictive models.
