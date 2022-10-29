import prepare_data
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def l_regression(filename):
  x_train, x_test, y_train, y_test = prepare_data.preprocessing(filename)
  regressor = LinearRegression()
  regressor.fit(x_train, y_train)
  y_pred = regressor.predict(x_test)
  plt.scatter(x_train, y_train, color = 'red')
  plt.plot(x_train, regressor.predict(x_train), color = 'blue')
  plt.title('Salary vs Experience (Training Set')
  plt.xlabel('Years of Experience')
  plt.ylabel('Salary')
  plt.figure()
  plt.pause(0.1)