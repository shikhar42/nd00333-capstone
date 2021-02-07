from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset
from azureml.core import Workspace, Datastore

run = Run.get_context()
ws = run.experiment.workspace
data = ws.datasets['HR_Analytics']

def clean_data(df):
    # Clean and one hot encode data
    x = df.to_pandas_dataframe().dropna()
   
    gender = pd.get_dummies(x.gender, prefix="gender")
    x.drop("gender", axis=1, inplace=True)
    x.join(gender)
    
    x["relevent_experience"] = x.relevent_experience.apply(lambda s: 1 if s == "Has relevent experience" else 0)
    
    major_discipline = pd.get_dummies(x.major_discipline, prefix="major_discipline")
    x.drop("major_discipline", axis=1, inplace=True)
    x.join(major_discipline)
    

    education_level = pd.get_dummies(x.education_level, prefix="education_level")
    x.drop("education_level", axis=1, inplace=True)
    x.join(education_level)
    
    last_new_job = pd.get_dummies(x.last_new_job, prefix="last_new_job")
    x.drop("last_new_job", axis=1, inplace=True)
    x.join(last_new_job)

    
    company_type = pd.get_dummies(x.company_type, prefix="company_type")
    x.drop("company_type", axis=1, inplace=True)
    x.join(company_type)
    
    experience = pd.get_dummies(x.experience, prefix="experience")
    x.drop("experience", axis=1, inplace=True)
    x.join(experience)
    
    enrolled_university = pd.get_dummies(x.enrolled_university, prefix="enrolled")
    x.drop("enrolled_university", axis=1, inplace=True)
    x.join(enrolled_university)

    company_size = pd.get_dummies(x.company_size, prefix="company_size")
    x.drop("company_size", axis=1, inplace=True)
    x.join(company_size)

   
    y = x.pop("target")

    return x, y

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    x, y = clean_data(data)
  

    # Splitting data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperdrive_model.joblib')

if __name__ == '__main__':
    main()
