# Health Insurance Cost Prediction Using machine Learning

## Introduction
This repository contains a health insurance prediction model that utilizes the Gradient Boost Regression algorithm to predict health insurance charges. The model was selected after comparing the performance of several other regression algorithms, including Linear Regression, SVR, and RandomForestRegressor.

## Features
The algorithm takes into account several features that are known to impact health insurance costs. Some of the common features used in this prediction model include:
Age: The age of the individual seeking insurance coverage.
BMI: The body mass index, which provides an indication of body fat based on height and weight.
Smoking: A binary variable indicating whether the individual is a smoker or not.
Region: The geographical region where the individual resides.
Number of children: The number of children the individual has.
These features serve as inputs to the machine learning model, which then generates a prediction for the health insurance costs.

## Dataset
The dataset used for training and testing the health insurance prediction model is  included in this repository. The dataset  includes features such as age, sex, BMI, number of children, smoker status, and region, along with the corresponding health insurance charges.

## Model Selection
To determine the most suitable regression algorithm for this prediction task, we experimented with several popular algorithms. The following algorithms were considered:
Linear Regression: A basic linear model that assumes a linear relationship between the features and the target variable.
Support Vector Regression (SVR): A regression model that uses support vector machines to find a hyperplane that best fits the data.
Random Forest Regressor: An ensemble model that combines multiple decision trees to make predictions.
After evaluating the performance of these models based on metrics such as mean squared error (MSE), mean absolute error (MAE), and R-squared (R²) score, we determined that the Gradient Boosting Regressor (GradientBoostingRegressor) provided the best fit for our health insurance prediction task. This algorithm combines multiple weak learners (decision trees) in a boosting framework to make accurate predictions.

## Usage
To use this health insurance prediction model, follow these steps:
- Prepare the dataset: Gather a dataset containing information about individuals, including their age, BMI, smoking status, region, and number of children. Ensure that the dataset includes the target variable, i.e., the actual health insurance costs.
- Data preprocessing: Perform necessary preprocessing steps, such as handling missing values, encoding categorical variables (like region and smoking), and scaling numerical features (like age and BMI).
- Train the model: Split the preprocessed dataset into training and testing subsets. Feed the training data into the GradientBoostingRegressor, which will learn the relationships between the input features and insurance costs.
- Evaluate the model: Use the testing data to evaluate the performance of the trained model. Calculate metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE) to assess the prediction accuracy.
- Predict insurance costs: Once the model is trained and evaluated, you can use it to predict insurance costs for new individuals. Provide the necessary feature values for an individual, and the model will output an estimated insurance cost.

## Requirements
run this project, you need the following dependencies:
Python (version 3.6 or later)
NumPy
Pandas
scikit-learn
Ensure that these dependencies are installed in your environment before running the code.

## Summary
This project provides a machine learning-based solution for predicting health insurance costs. By utilizing the GradientBoostingRegressor algorithm and considering relevant features such as age, BMI, smoking status, region, and number of children, the model can estimate insurance costs for new individuals. It offers a practical tool for insurers, individuals, and policymakers to understand the factors influencing health insurance premiums and make informed decisions.


## AUTHOR
👤 **Nwaigbo Olive**
- Github:  [@olivenwaigbo](https://github.com/Olivenwaigbo?tab=following)    

- LinkedIn:  [olive nwaigbo](https://www.linkedin.com/in/olive-nwaigbo-95707a151)

## 🤝**Contributing**
Contributions to this project are welcome. If you find any issues or have ideas for improvements, please open an issue or submit a pull request. We appreciate your feedback and collaboration.


## **Show your support**
Give a ⭐️ if you like this project

## **Acknowledgements**
- Thank you Data Thinkers  for guiding me through this project.
## 📝 License 
This project is [MIT](./MIT.md) licensed

