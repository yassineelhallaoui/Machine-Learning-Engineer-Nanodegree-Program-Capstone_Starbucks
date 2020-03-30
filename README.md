## Table of Contents
1. [Installation](#installation)
2. [Introducing a Dataset](#dataset-introduction)
3. [Project Motivation](#project-motivation)
4. [Data Preparation](#data-preparation)
5. [Data Visualization](#data-visualization)
6. [File Descriptions](#files)
7. [Results](#results)

### Installation <a name="installation"></a>
For running this project, the most important library is Python version of Anaconda Distribution. It installs all necessary packages for analysis and building models:
{
import re
import math
import json
import fastprogress
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
}
 
### Introducing a Dataset <a name="dataset-introduction"></a>
This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

### Project Motivation <a name="project-motivation"></a>
I chose this project to understand the success rate of offers being sent and analysis is done through addressing the following questions.
1. How many customers were provided with a specific offer? 

--> the feature importance given by all 3 models were that the tenure of a member is the biggest predictor of the effectiveness of an offer. Further study would be able to indicate what average tenure days would result in an effective BOGO offer.

2. What's the performance level of an offer?

-->  use 3 separate models to predict the effectiveness of each offer type ended up with good accuracy for the 2 of the models (82.83% for BOGO and 87.35% for discount), while slightly less accurate performance for another informational offers (75.3%). However, I would regard 75% as acceptable in a business setting, as for informational offers, there is no cost involved to inform users of a product. Meanwhile, an 80% and above accuracy in a business setting would be acceptable to show offers to people, even if the model misclassifies a few, the overall revenue increase might justify the few mistakes.

### Data Preparation <a name="data-preparation"></a>
There are three datasets provided and each dataset is cleaned and preprocessed for further analysis. The target features for analysis are offer_success, percent_success.

A. Portfolio - renaming id column name to offer_id, one-hot encoding of channels and offer_type columns
B. Profile - profile: renaming id column name to customer_id, replacing age value 118 to nan, creating readable date format in became_member_on column, dropping rows with no gender, income, age data, converting gender values to numeric 0s and 1s, adding start year and start month columns (for further analysis)
C. Transcript - renaming person column name to customer_id, creating separate columns for amount and offer_id from value col, dropping transaction rows whose customer_id is not in profile:customer_id, converting time in hours to time in days, segregating offer and transaction data, finally dropping duplicates if any

### Data Visualization <a name="data-visualization"></a>
1- Customers Global Data Pattern Visualization [Age, Income,] per years: Age distribution plot depicts that the median age of a customer is 60 and most of our customers belong to age range between 40 to 70. Income distribution plot shows that the number of customers whose average salary is less than 70K is high than the other side considering 70K to be median of the income distribution. Membership distribution has interesting results - 2017 has the highest registered customers than any starting from 2013. The plot also shows that there is an increasing trend in the number of registrations except for 2017:

2- Customer Income distribution per gender & ages: plots conclude that minimum and maximum income for both male and female are approximately same but the count of male customers in low-income level is slightly higher than that of female customers. concidering all highest income are from age od 40 above.

### File Descriptions <a name="files"></a>
There is a notebook available here to showcase work related to the above questions and wrangling process. There are 3 data file
please Note all json files should be in a folder named [data] before using the Jupyter notebook. [unzip the data file]

A. portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
B. profile.json - demographic data for each customer
C. transcript.json - records for transactions, offers received, offers viewed, and offers completed

## Results<a name="results"></a>
The main observations of the code are published https://github.com/yassineelhallaoui/Machine-Learning-Engineer-Nanodegree-Program-Capstone_Starbucks.
