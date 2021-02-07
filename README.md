

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
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
