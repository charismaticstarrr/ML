# ML
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
data = pandas.read_csv('datttebayo2.csv')
data.describe()

X = DataFrame(data , columns=['production_budget_usd'])
y = DataFrame(data , columns=['worldwide_revenue_usd'])

plt.figure(figsize=(10,8))
plt.scatter(X,y,alpha=0.3)
plt.title('Film cost vs Global revenue')
plt.xlabel('Production budget $')
plt.ylabel('Worldwide gross $')
plt.ylim(0, 3000000000)
plt.xlim(0,450000000)
plt.show()

regression=LinearRegression()
regression.fit(X,y)
regression.coef_#theta_1
#intercept
regression.intercept_

plt.figure(figsize=(10,8))
plt.scatter(X,y,alpha=0.3)
plt.plot(X,regression.predict(X),color='red',linewidth='6')
plt.title('Film cost vs Global revenue')
plt.xlabel('Production budget $')
plt.ylabel('Worldwide gross $')
plt.ylim(0, 3000000000)
plt.xlim(0,450000000)
plt.show()

regression.score(X,y)
