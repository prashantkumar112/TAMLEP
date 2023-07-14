# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modeled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## How to Run
1. Install anaconda/miniconda if not already installed
2. Create the conda environment using below command
    "conda env create --name <envname> --file=environments.yml"
3. activate the created environment
	"conda activate <envname>"
4. Install mlflow if not installed
5. Open a new terminal and activate the conda environment in that teminal as well and then run mlflow service (Below command will start the mlflow server on "localhost:5000" which can be accessed from browser )

	mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000


6. Go to src/my_package directory
7. Run run_housing.py

	"python run_housing.py --raw_data_dir=../../data/raw --processed_data_dir=../data/processed --model_dir=../../artifacts"
	
	Note : if user arguments are not provided then default values will be taken from config.json file
	
	This script will run the ingest, train and score scripts for this project and print the final results 
	in a table.
	You can also see the results in mlflow (localhost:5000) in a browser.