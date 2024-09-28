import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

diabetes_df = pd.read_csv('datasets/diabetes.csv',sep='\t')
diabetes_df['is_diabetic'] = diabetes_df['Y'] > 100
diabetes_df.drop(columns=['Y'],inplace=True)
X = diabetes_df[['AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']]
Y = diabetes_df[['is_diabetic']]

rand_state = 50

#80% of data is used for training and 20% is used for testing
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=rand_state)

#build and train model
model = LogisticRegression(random_state=rand_state)
model.fit(X_train,Y_train)

