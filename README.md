

# Job Change of Data Scientists Prediction using Microsoft Azure

A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses which conduct by the company. MCompany wants to know which of these candidates are really wants to work for the company after training or looking for a new employment because it helps to reduce the cost and time as well as the quality of training or planning the courses and categorization of candidates.

In this project, we will predict if an individual is looking for a new job based on different factors using 2 techiques(AutoML & Hyperdrive).

## Dataset

### Overview
This dataset has been exported from Kaggle. Here's the link https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

### Task
The task is to predict if an employee will look for another job or not after the company provides them with a training. Below are the features of this dataset:

* enrollee_id : Unique ID for candidate
* city: City code
* city_ development _index : Developement index of the city (scaled)
* gender: Gender of candidate
* relevent_experience: Relevant experience of candidate
* enrolled_university: Type of University course enrolled if any
* education_level: Education level of candidate
* major_discipline :Education major discipline of candidate
* experience: Candidate total experience in years
* company_size: No of employees in current employer's company
* company_type : Type of current employer
* lastnewjob: Difference in years between previous job and current job
* training_hours: training hours completed
* target: 0 – Not looking for job change, 1 – Looking for a job change

### Access

The data can be accessed by loading a csv file in the azure platform. 

![Alt text](https://github.com/shikhar42/nd00333-capstone/blob/master/dataset1.PNG?raw=true "Dataset")

After loading it to the portal we can access the data in the notebooks by importing the workspace.

![Alt text](https://github.com/shikhar42/nd00333-capstone/blob/master/dataset.PNG?raw=true "Dataset")

## Automated ML
Below is the screenshot of the configurations we used for automl.

![Alt text](https://github.com/shikhar42/nd00333-capstone/blob/master/automl_config.PNG?raw=true "config")

In the configurations of Auto ML, we have selected the experiment timeout time to be 30 mins. Since we are doing a classification task, we have selected classification in the task. The primary metric that we are using is Accuracy. The number of cross validations used here is 5.

### Results
![Alt text](https://github.com/shikhar42/nd00333-capstone/blob/master/runwidget_automl1.PNG?raw=true "automl")

![Alt text](https://github.com/shikhar42/nd00333-capstone/blob/master/runwidget_automl2.PNG?raw=true "automl")

![Alt text](https://github.com/shikhar42/nd00333-capstone/blob/master/runwidget_automl3.PNG?raw=true "automl")

From the screenshots above, we can see that Voting Ensemble was the best model with an accuracy of 80.02%
One of the major issue with this dataset is of class imbalance. That is something that can be taken care of to improve the prediction accuracy. Also, feature selection can be used to identify only the important features so as to improve the accuracy.

## Hyperparameter Tuning
For this technique, I decided to choose logistic regression as:

* It is the most basic algorithm when it comes to classification and one should always start from basic models
* It is easy to understand the results and simple to train
* The execution time is very fast

The hyperparameters that were used are:

* The regularization parameter was chosen from 0.001 to 1 to handle overfitting in the model.
* The total iterations was selected between 10 and 200.


### Results

![Alt text](https://github.com/shikhar42/nd00333-capstone/blob/master/runwidget_hd1.PNG?raw=true "hyperdrive")

![Alt text](https://github.com/shikhar42/nd00333-capstone/blob/master/runwidget_hd2.PNG?raw=true "hyperdrive")

As we can see from the screenshot above, the model with parameters of regularization as 1 and max iteration of 200 gave us the best accuracy of 84.67%. 

To improve these results, we can further try different ranges of hyperparameters here. However, the best thing would be to handle the class imbalance in the dataset.

## Model Deployment
The best model from AutoML(Voting Ensemble) was deployed. An inference config was created using the score.py file and the service was deployed using the following code:

![Alt text](https://github.com/shikhar42/nd00333-capstone/blob/master/deployment.PNG?raw=true "deployment")

![Alt text](https://github.com/shikhar42/nd00333-capstone/blob/master/deployment_active_state.PNG?raw=true "deployment")

From the screenshot above, we can also verify from the azure portal that the model was successfully deployed and is in a healthy state.

After this, we tested the model endpoint by providing dummy data to see the results. Below is the screenshot of the test data used to test the endoint:

![Alt text](https://github.com/shikhar42/nd00333-capstone/blob/master/deployment_test.PNG?raw=true "deployment")

In the screenshot above, we can see that we are providing to cases to test the deployed model. The model returns the output as 0 and 1. This means that based on Voting Ensemble model, the first set of parameters would mean that the employee is not looking for a job change. However, the second output is 1, that means the that specific employee is looking for a job change.


## Screen Recording

https://www.youtube.com/watch?v=S_Y7oSuJHcQ


