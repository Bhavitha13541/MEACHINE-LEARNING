import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
data = pd.read_csv('C:\\Users\\Bhavitha\\Desktop\\MEACHINE LEARNING\\test_scores.csv')
def predict_using_sklearn():
    data = pd.read_csv('C:\\Users\\Bhavitha\\Desktop\\MEACHINE LEARNING\\test_scores.csv')
    r = LinearRegression()
    r.fit(data[['math']], data['cs'])
    return r.coef_, r.intercept_

def test_score(x, y):
    m_curr = b_curr = 0
    iterations = 100
    n = len(x)
    learning_rate = 0.001

    cost_previous = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))
    return m_curr, b_curr

if __name__ == "__main__":
    df = pd.read_csv('C:\\Users\\Bhavitha\\Desktop\\MEACHINE LEARNING\\test_scores.csv')
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = test_score(x, y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))
    
    m_sklearn, b_sklearn = predict_using_sklearn()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))

    # output:
# m -6.407420133024764e+92, b -9.041420662667186e+90, cost 2.7102698853710618e+187, iteration 99
# Using gradient descent function: Coef -6.407420133024764e+92 Intercept -9.041420662667186e+90