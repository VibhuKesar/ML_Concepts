# LinearRegression, unsupervised learning; split training and testing data, compute on training data and result outcome with testing data to check model 
# overfit or underfit.

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

customers = pd.read_csv('e_c')

customers.head()	#Display the information

## Display graph using sns
# sns.jointplot(data =customers,x='Time on Website',y='Yearly Amount Spent')
# sns.jointplot(data =customers,x='Time on App',y='Yearly Amount Spent')
# sns.jointplot(data =customers,x='Time on App',y='Length of Membership',kind ='hex')


## Length of Membership using normal Visualization and maximize fit model
sns.lmplot(data =customers,x='Length of Membership',y='Yearly Amount Spent') #both +ve correlation hence one increase other do too
sns.pairplot(customers) #display or visualize all the relationship graphs


## Training and testing data
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership','Yearly Amount Spent']]
#We will find that people spend more money but spend less avg. session

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,random_state = 101) 
# Divide in 70-30 ratio, By default, the value is set to 0.25, the random number generator is the RandomState instance  

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train) #fit the model using Least Square and Sum of Squares
print(lm.coef_)

predictions = lm.predict(X_test) #predict values from our model
plt.scatter(y_test, predictions) #plot scatter graph
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
# plt.show()

#Evaluating Model

from sklearn import metrics
print("MAE", metrics.mean_absolute_error(y_test,predictions)) #Mean Average Error
print("MSE", metrics.mean_squared_error(y_test,predictions)) #Mean Square Error
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test,predictions))) #Root Mean Square Error

print(metrics.explained_variance_score(y_test,predictions)) #Best possible score is 1.0, lower values are worse.

sns.distplot((y_test - predictions)) #see residuals points of testing data from predicted, outcome Variance is lower, hence more fit model we have.
plt.show()

cd = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff']) #list, columns, name of col.
print(cd)

