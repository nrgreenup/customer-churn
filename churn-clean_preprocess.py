#### Clean and Preprocess Data for Telco Customer Churn Data ####

## HELP: https://stackoverflow.com/questions/31749448/how-to-add-percentages-on-top-of-bars-in-seaborn

### Import required packages for preprocessing and analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes 
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                     RandomizedSearchCV)
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)

"""
Package versions:
python        : 3.6.5
scitkit-learn : 0.19.1
pandas        : 0.23.0
numpy         : 1.14.3
matplotlib    : 2.2.2
seaborn       : 0.8.1
"""

### Import data
df = pd.read_csv('churn.csv')

### Get preliminary info about dataframe
print(df.info()) # Categorical variables are of type 'object'
print(df.isnull().sum()) # No NaNs

## Set TotalCharges to float 
df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

## Set senior citizen to type category
df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'yes', 0: 'no'}).astype('category')

## Check info of continuous features 
print(df.info()) # Conversion successful
print(df.describe()) # Disparate ranges, should be normalized

### Examine continuous features
## Examine correlations
df['tenuremonth'] = (df['tenure'] * df['MonthlyCharges']).astype(float)
df.corr()

### Get level counts of all categorical variables


## Define list of categorical variables
cat_vars = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'Churn']

## Print value counts of categorical variables
for var in cat_vars:
    print(df[var].value_counts())
    
## Collapse categories where appropriate
#  Multiple lines 'no phone service' to 'no'
df['MultipleLines'] = df['MultipleLines'].map({'No phone service': 'No',
                                               'Yes': 'Yes',
                                               'No': 'No'}).astype('category')
df['MultipleLines'].value_counts() #Conversion successful

#  6 features, convert 'no internet service' to 'no'
no_int_service_vars = ['OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection','TechSupport', 
                       'StreamingTV', 'StreamingMovies']

for var in no_int_service_vars:
    df[var] = df[var].map({'No internet service': 'No',
                           'Yes': 'Yes',
                           'No': 'No'}).astype('category')
    
for var in no_int_service_vars:
    print(df[var].value_counts())
    
## Plot distributions of categorical variables
for var in cat_vars:
    ax = sns.countplot(x = df[var], data = df, palette = 'colorblind')
    total = float(len(df[var])) 
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 10,
                '{:1.2f}'.format(height/total),
                ha="center")
    plt.title('Distribution of ' + str(var))
    plt.ylabel('Number of Customers')
    plt.figtext(0.55, 0.01, 
                'Decimal above bar is proportion of total in that class',
                horizontalalignment = 'center', fontsize = 8,
                style = 'italic')
    plt.xticks(rotation = 60)
    plt.tight_layout()
    plt.savefig('plot_dist-' + str(var) + '.png', dpi = 200)
    plt.show()


### Exploratory Plots
## Churn by tenure
plt.subplot(1,2,1)
sns.violinplot(x = df['Churn'], y = df['tenure'], data = df, inner = None,
               palette = 'colorblind')
plt.title('Churn by Customer Tenure')

## Churn by monthly charges
plt.subplot(1,2,2)
sns.violinplot(x = df['Churn'], y = df['MonthlyCharges'], data = df, inner = None,
               palette = 'colorblind')
plt.title('Churn by Monthly Charge')
plt.tight_layout()
plt.savefig('plot-churn_by_charges_tenure.png', dpi = 200)
plt.show()

## Churn by contract length
contract_churn = df.groupby(['Contract', 'Churn']).agg({'customerID': 'count'})
contract = df.groupby(['Contract']).agg({'customerID': 'count'})
contract_churn_pct = contract_churn.div(contract, level='Contract') * 100
contract_churn_pct = contract_churn_pct.reset_index()

sns.barplot(x = 'Contract' , y = 'customerID', hue = 'Churn',
            data = contract_churn_pct)
plt.title('Churn Rate by Contract Length')
plt.ylabel('Percent of Class')
plt.tight_layout()
plt.savefig('plot-churn_by_contract.png', dpi = 200)
plt.show()

## Churn by gender
gender_churn = df.groupby(['gender', 'Churn']).agg({'customerID': 'count'})
gender = df.groupby(['gender']).agg({'customerID': 'count'})
gender_churn_pct = gender_churn.div(gender, level='gender') * 100
gender_churn_pct = gender_churn_pct.reset_index()

sns.barplot(x = 'gender' , y = 'customerID', hue = 'Churn',
            data = gender_churn_pct)
plt.title('Churn Rate by Gender')
plt.ylabel('Percent of Class')
plt.tight_layout()
plt.savefig('plot-churn_by_gender.png', dpi = 200)
plt.show()

## Churn by internet service
internet_churn = df.groupby(['InternetService', 'Churn']).agg({'customerID': 'count'})
internet = df.groupby(['InternetService']).agg({'customerID': 'count'})
internet_churn_pct = internet_churn.div(internet, level='InternetService') * 100
internet_churn_pct = internet_churn_pct.reset_index()


sns.barplot(x = 'InternetService' , y = 'customerID', hue = 'Churn',
            data = internet_churn_pct)
plt.title('Churn Rate by Internet Service Status')
plt.ylabel('Percent of Class')
plt.tight_layout()
plt.savefig('plot-churn_by_internet.png', dpi = 200)
plt.show()

## Churn by phone service
phone_churn = df.groupby(['PhoneService', 'Churn']).agg({'customerID': 'count'})
phone = df.groupby(['PhoneService']).agg({'customerID': 'count'})
phone_churn_pct = phone_churn.div(phone, level='PhoneService') * 100
phone_churn_pct = phone_churn_pct.reset_index()


sns.barplot(x = 'PhoneService' , y = 'customerID', hue = 'Churn',
            data = phone_churn_pct)
plt.title('Churn Rate by Phone Service Status')
plt.ylabel('Percent of Class')
plt.tight_layout()
plt.savefig('plot-churn_by_phone.png', dpi = 200)
plt.show()

### Standardize continuous variables for distance based models
scale_vars = ['tenure', 'MonthlyCharges']
scaler = StandardScaler() 
df[scale_vars] = scaler.fit_transform(df[scale_vars])
df[scale_vars].describe()


### Drop ID and TotalCharges vars
df = df.drop(['customerID', 'TotalCharges', 'tenuremonth'],  axis = 1)
print(df.info())

### Encode data for analyses
## Binarize binary variables
df_enc = df.copy()
binary_vars = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'PhoneService', 'MultipleLines', 'OnlineSecurity', 
               'OnlineBackup','DeviceProtection', 'TechSupport', 
               'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
               'Churn']
enc = LabelEncoder()
df_enc[binary_vars] = df_enc[binary_vars].apply(enc.fit_transform)

## One-hot encode multi-category cat. variables
multicat_vars = ['InternetService', 'Contract', 'PaymentMethod']
df_enc = pd.get_dummies(df_enc, columns = multicat_vars)
df_enc.iloc[:,16:26] = df_enc.iloc[:,16:26].astype(int)
print(df_enc.info())



"""
### EXTRA
## Change categorical variables to type 'category', reduces memory size by >1/2
df[cat_vars] = df[cat_vars].astype('category')
print(df.info()) # Conversion successful
"""


print(time.localtime)
