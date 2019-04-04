import pandas as pd
import numpy as np

# Reading in adult.data from UCL Database
adult_data_raw = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)


# Renaming Columns and stripping strings of whitespace
cols = ['Age','Workclass','fnlwgt','Education','Education_num','Marital Status','Occupation','Relationship','Race',
       'Sex','Capital Gain','Capital Loss','Hours per Week','Country','Salary']
adult_data_raw.columns = cols
adult_data_raw = adult_data_raw.applymap(lambda x: x.strip() if type(x) is str else x)

# Dropping all data entries with unknown values and entries where GDP information is unavailable
na_indices = []
unknown_countries = []
for i in range(adult_data_raw.shape[0]):
    if '?' in adult_data_raw.iloc[i].values:
        na_indices.append(i)
    if 'South' in adult_data_raw.iloc[i].values:
        unknown_countries.append(i)
    if 'Outlying-US(Guam-USVI-etc)' in adult_data_raw.iloc[i].values:
        unknown_countries.append(i)
    if 'Laos' in adult_data_raw.iloc[i].values:
        unknown_countries.append(i)
    if 'Taiwan' in adult_data_raw.iloc[i].values:
        unknown_countries.append(i)

dropped_indices = na_indices + unknown_countries
print(len(dropped_indices))
print(len(na_indices))
adult_data_raw = adult_data_raw.drop(dropped_indices).reset_index(drop=True)

# Ensuring all values in Age are numeric
adult_data_raw['Age'] = pd.to_numeric(adult_data_raw['Age'])

# Consolidating and encoding Workclass to:
# Not-working: 0, Private: 1, Self-emp: 2, Government: 3
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].astype(str)
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].replace('Never-worked', 'Not-working')
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].replace('Without-pay', 'Not-working')
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].replace('Self-emp-not-inc', 'Self-emp')
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].replace('Self-emp-inc', 'Self-emp')
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].replace('Local-gov', 'Government')
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].replace('State-gov', 'Government')
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].replace('Federal-gov', 'Government')
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].replace('Not-working', 0)
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].replace('Private', 1)
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].replace('Self-emp', 2)
adult_data_raw['Workclass'] = adult_data_raw['Workclass'].replace('Government', 3)
adult_data_raw['Workclass'] = pd.to_numeric(adult_data_raw['Workclass'])

# Dropping fnlwgt column beacuse it does not appear to provide value to classification
adult_data_raw = adult_data_raw.drop('fnlwgt', axis=1)

# Consolidating and encoding Education to:
# Dropout: 0, HS-grad: 1, Prof-school: 2, Associates: 3, Bachelors: 4, Masters: 5, Doctorate: 6
adult_data_raw['Education'] = adult_data_raw['Education'].astype(str)
adult_data_raw['Education'] = adult_data_raw['Education'].replace('Some-college', 'Dropout')
adult_data_raw['Education'] = adult_data_raw['Education'].replace('Assoc-voc', 'Associates')
adult_data_raw['Education'] = adult_data_raw['Education'].replace('11th', 'Dropout')
adult_data_raw['Education'] = adult_data_raw['Education'].replace('Assoc-acdm', 'Associates')
adult_data_raw['Education'] = adult_data_raw['Education'].replace('10th', 'Dropout')
adult_data_raw['Education'] = adult_data_raw['Education'].replace('7th-8th', 'Dropout')
adult_data_raw['Education'] = adult_data_raw['Education'].replace('9th', 'Dropout')
adult_data_raw['Education'] = adult_data_raw['Education'].replace('12th', 'Dropout')
adult_data_raw['Education'] = adult_data_raw['Education'].replace('5th-6th', 'Dropout')
adult_data_raw['Education'] = adult_data_raw['Education'].replace('1st-4th', 'Dropout')
adult_data_raw['Education'] = adult_data_raw['Education'].replace('Preschool', 'Dropout')
adult_data_raw['Education'] = adult_data_raw['Education'].replace('Dropout', 0)
adult_data_raw['Education'] = adult_data_raw['Education'].replace('HS-grad', 1)
adult_data_raw['Education'] = adult_data_raw['Education'].replace('Prof-school', 2)
adult_data_raw['Education'] = adult_data_raw['Education'].replace('Associates', 3)
adult_data_raw['Education'] = adult_data_raw['Education'].replace('Bachelors', 4)
adult_data_raw['Education'] = adult_data_raw['Education'].replace('Masters', 5)
adult_data_raw['Education'] = adult_data_raw['Education'].replace('Doctorate', 6)
adult_data_raw['Education'] = pd.to_numeric(adult_data_raw['Education'])

# Dropping Education_num in favor of using encoded Education column
adult_data_raw = adult_data_raw.drop('Education_num', axis=1)


# Consolidating and encoding Marital Status to:
# Married: 0 , Single: 1, Divorced: 2, Separated: 3, Widowed: 4
adult_data_raw['Marital Status'] = adult_data_raw['Marital Status'].astype(str)
adult_data_raw['Marital Status'] = adult_data_raw['Marital Status'].replace('Married-civ-spouse', 'Married')
adult_data_raw['Marital Status'] = adult_data_raw['Marital Status'].replace('Never-married', 'Single')
adult_data_raw['Marital Status'] = adult_data_raw['Marital Status'].replace('Married-spouse-absent', 'Married')
adult_data_raw['Marital Status'] = adult_data_raw['Marital Status'].replace('Married-AF-spouse', 'Married')
adult_data_raw['Marital Status'] = adult_data_raw['Marital Status'].replace('Married', 0)
adult_data_raw['Marital Status'] = adult_data_raw['Marital Status'].replace('Single', 1)
adult_data_raw['Marital Status'] = adult_data_raw['Marital Status'].replace('Divorced', 2)
adult_data_raw['Marital Status'] = adult_data_raw['Marital Status'].replace('Separated', 3)
adult_data_raw['Marital Status'] = adult_data_raw['Marital Status'].replace('Widowed', 4)
adult_data_raw['Marital Status'] = pd.to_numeric(adult_data_raw['Marital Status'])


# Consolidating and encoding Occupation to:
# White-collar:0 , Blue-collar: 1, Pink-collar:2 , Other-service: 3
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].astype(str)
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Prof-specialty', 'White-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Craft-repair', 'Blue-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Exec-managerial', 'White-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Adm-clerical', 'White-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Sales', 'Pink-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Machine-op-inspct', 'Blue-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Transport-moving', 'Blue-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Handlers-cleaners', 'Blue-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Tech-support', 'Pink-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Farming-fishing', 'Blue-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Protective-serv', 'Pink-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Priv-house-serv', 'Blue-collar')
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('White-collar', 0)
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Blue-collar', 1)
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Pink-collar', 2)
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Other-service', 3)
adult_data_raw['Occupation'] = adult_data_raw['Occupation'].replace('Armed-Forces', 5)
adult_data_raw['Occupation'] = pd.to_numeric(adult_data_raw['Occupation'])

# Consolidating and encoding Relationship to:
# Not-in-family: 0, Unmarried: 1, Other-relative: 2, Own-child: 3, Wife: 4, Husband: 5
adult_data_raw['Relationship'] = adult_data_raw['Relationship'].astype(str)
adult_data_raw['Relationship'] = adult_data_raw['Relationship'].replace('Not-in-family', 0)
adult_data_raw['Relationship'] = adult_data_raw['Relationship'].replace('Unmarried', 1)
adult_data_raw['Relationship'] = adult_data_raw['Relationship'].replace('Other-relative', 2)
adult_data_raw['Relationship'] = adult_data_raw['Relationship'].replace('Own-child', 3)
adult_data_raw['Relationship'] = adult_data_raw['Relationship'].replace('Wife', 4)
adult_data_raw['Relationship'] = adult_data_raw['Relationship'].replace('Husband', 5)
adult_data_raw['Relationship'] = pd.to_numeric(adult_data_raw['Relationship'])

# Consolidating and encoding Race to:
# Other: 0, Amer-Indian-Eskimo: 1, Asian-Pac-Islander: 2, Black: 3, White: 4
adult_data_raw['Race'] = adult_data_raw['Race'].astype(str)
adult_data_raw['Race'] = adult_data_raw['Race'].replace('Other', 0)
adult_data_raw['Race'] = adult_data_raw['Race'].replace('Amer-Indian-Eskimo', 1)
adult_data_raw['Race'] = adult_data_raw['Race'].replace('Asian-Pac-Islander', 2)
adult_data_raw['Race'] = adult_data_raw['Race'].replace('Black', 3)
adult_data_raw['Race'] = adult_data_raw['Race'].replace('White', 4)
adult_data_raw['Race'] = pd.to_numeric(adult_data_raw['Race'])

# Converting Sex to 1 if Male and 0 if Female
adult_data_raw['Sex'] = adult_data_raw['Sex'].apply(lambda x: 1 if x == 'Male' else 0)

# Ensuring all Capital Gain, Capital Loss, and Hours per week values are numeric
adult_data_raw['Capital Gain'] = pd.to_numeric(adult_data_raw['Capital Gain'])
adult_data_raw['Capital Loss'] = pd.to_numeric(adult_data_raw['Capital Loss'])
adult_data_raw['Hours per Week'] = pd.to_numeric(adult_data_raw['Hours per Week'])

# Consolidating certain countries to conincide with information on GDP per Country 
adult_data_raw['Country'] = adult_data_raw['Country'].astype(str)
adult_data_raw['Country'] = adult_data_raw['Country'].replace('Scotland', 'United Kingdom')
adult_data_raw['Country'] = adult_data_raw['Country'].replace('England', 'United Kingdom')
adult_data_raw['Country'] = adult_data_raw['Country'].replace('Trinadad&Tobago', 'Trinidad and Tobago')
adult_data_raw['Country'] = adult_data_raw['Country'].replace('Hong', 'Hong Kong SAR, China')
adult_data_raw['Country'] = adult_data_raw['Country'].replace('Holand-Netherlands', 'Netherlands')
adult_data_raw['Country'] = adult_data_raw['Country'].replace('Columbia', 'Colombia')

# Converting Salary to 1 if >50k and 0 if <= to 50k
adult_data_raw['Salary'] = adult_data_raw['Salary'].apply(lambda x: 1 if x == '>50K' else 0)

# Unique Countries present in data and respective GDPs
countries = ['United-States','Cuba','Jamaica','India','Mexico','Puerto-Rico','Honduras','United Kingdom',
             'Canada','Germany','Iran','Philippines', 'Poland','Colombia','Cambodia','Thailand','Ecuador',
             'Haiti','Portugal', 'Dominican-Republic','El-Salvador','France','Guatemala','Italy','China',
             'Japan','Yugoslavia','Peru','Trinidad and Tobago','Greece','Nicaragua','Vietnam',
             'Hong Kong SAR, China','Ireland','Hungary','Netherlands']

gdp_per_country = [30068.23092, 2282.389619, 2875.886078, 396.0146123, 4294.978983, 12173.16369, 687.4814199,
                   24219.62285, 21183.22008, 30564.24781, 1955.146029, 1159.589292, 4140.983541, 2553.549692, 
                   319.3632544, 3042.903989, 2159.150814, 365.0110071, 12185.06389, 2271.943595, 1690.136559,  
                   26871.82902, 1472.275343, 23020.09994, 709.4137551, 38436.92631, 5177.748864, 2260.637733,   
                   4577.004528, 13749.11515, 916.5187095, 322.8570476, 24818.15455, 20860.59765, 4525.140121,
                   28698.66602]

# Populating new feature GDP per Cap with respective GDP per Capita per Country
dictionary = dict(zip(countries,gdp_per_country))

gdp_list = []
adult_data_raw['GDP per Cap'] = 0
for i in range(adult_data_raw.shape[0]):
    gdp_list.append(dictionary.get(adult_data_raw['Country'].iloc[i]))

adult_data_raw['GDP per Cap'] = gdp_list

# Adding column Constant composed of ones for the bias in dot product
adult_data_raw['Constant'] = np.ones((adult_data_raw.shape[0], 1), dtype=int)

# Rearranging columns for readability
new_cols = ['Age','Workclass','Education','Marital Status','Occupation','Relationship','Race',
            'Sex','Capital Gain','Capital Loss','Hours per Week','Country','GDP per Cap','Constant','Salary']
adult_data_raw = adult_data_raw[new_cols]

# Exporting data as cleaned csv file
# adult_data_raw.to_csv(r'~/Downloads/adult_data_cleaned.csv')