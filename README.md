# Predicting Accident Severity Using Spark Big Data

## Overview
This project aims to predict the severity of traffic accidents using historical data. By classifying accidents as either "Slight" or "Serious_to_Fatal", the goal is to improve emergency response times and optimize resource allocation. Big Data Spark platform is used to process a dataset of over 2 million UK traffic accident records. We apply various machine learning models, including Logistic Regression, Random Forest, and Gradient Boosting, to predict accident severity.We also tested different Spark configurations and adjusted the number of partitions. This allowed us to analyse the change in execution times and improve the scalability

## Technologies Used
- **Apache Spark**
- **Python**

## File Structure
- **Dataset**: Contains the dataset used for the project.
- **Preprocessing**: Python script in `.ipynb` format, outlining the preprocessing steps performed on the dataset.
- **Before Downsampling**: Python scripts and results of model training **before** downsampling the data.
- **After Downsampling**: Python scripts and results of model training **after** applying downsampling.
- **Configuration**: Results of machine learning models (Logistic Regression, Random Forest, Gradient Boosting) with various configurations.
- **Partitions**: Results from partitioning experiments for the Gradient Boosting model.
- **Slurm File**: Contains the Slurm script for submitting Spark jobs on the cluster.
- **Project Report**:Contains the final report of the project.

## Challenges and Solutions
- **Imbalance**: The dataset was highly imbalanced, with many more "Slight" accidents. This was addressed using **downsampling** to balance the classes.

## Results
- **Logistic Regression**: Baseline performance.
- **Random Forest**: Improved accuracy and recall.
- **Gradient Boosting**: Best performing model with the highest recall and F1 score.

## Conclusion
We used PySpark on the Taki cluster to train Logistic Regression, Random Forest, and Gradient Boosting models for predicting accident severity. While all models performed similarly, Gradient Boosting slightly outperformed the others with a higher recall and F1 score.

## Future Work
- Use **SMOTE** to generate synthetic data and help balance the dataset further.
- Explore the use of **Neural Networks** to potentially improve model performance.
