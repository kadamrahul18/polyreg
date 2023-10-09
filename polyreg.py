import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline

class Poly_Reg():

    def __init__(self, order, lr, iter):

        self.order = order
        self.lr = lr
        self.iter = iter
    
    def fit(self, x, y):                                    # x == input, y == targets, y_hat == prediction

        self.x = x
        self.y = y
        self.m = self.x.shape[0]

        # self.theta = np.ones((self.order + 1, 1))           # initialize parameters to one
        self.theta = np.random.rand(self.order + 1, 1)        # initialize parameters to random

        x_t = self.design_matrix(self.x)

        # Gradient Descent
        for _ in range(self.iter):
            
            y_hat = self.predict(self.x)                    # predict 

            gradient = np.dot(x_t.T, (y_hat - y)) / self.m  # x_t.T, .T transposes the vector, (y_hat - y) is error
            self.theta -= self.lr * gradient                # update parameters (theta vector) accordingly

        return self
    
    def design_matrix(self, x):

        m = x.shape[0]
        x_t = np.ones((m,1))

        for i in range(1, self.order + 1):
                x_p = np.power(x, i)
                x_t = np.append(x_t, x_p, axis = 1)

        '''
        if x = [1]      x_t = [1 1 1 ...]
               [2]            [1 2 4 ...]
               [3] then       [1 3 9 ...]
        '''
        return x_t

    def predict(self, x):

        x_t = self.design_matrix(x)
        y_hat = np.dot(x_t, self.theta)

        return y_hat

    def get_risk(self, x, y_pred, y):                            # mean squared error

        risk = 0
        m = x.shape[0]

        risk = (y_pred - y)**2
        risk = risk / (2*m)

        return risk

  
    # def plot_d_vs_risk(self, train_r, test_r):
        
    #     train_costs = {}
    #     test_costs = {}

    #     for i in range(1, 10):
    #         if i not in train_costs:
    #             train_costs[i] = []
    #         if i not in test_costs:
    #             test_costs[i] = []
    #         train_costs[i].append(train_r[ : , i-1])
    #         test_costs[i].append(test_r[ : , i-1])
        
    #     for i in range(1, 10):
    #         plt.plot(i, train_costs[i], label=f'Train - Order {i}')
    #         plt.plot(i, test_costs[i], label=f'Test - Order {i}')
        
    #     plt.xlabel("order of the polynomial (d)")
    #     plt.ylabel("risk")
    #     plt.show()

    def plot_d_vs_risk(self):

        df = pd.read_csv(r"/Users/sunilkadam/Desktop/ML_HW/HW_1/Solution/min_risk.csv")

        order = df["order"].to_numpy(dtype = float)
        train_risk = df["min_train_risk"].to_numpy(dtype = float)
        test_risk = df["min_test_risk"].to_numpy(dtype = float)

        train_risk = (train_risk - np.min(train_risk)) / (np.max(train_risk) - np.min(train_risk))         # normalize train_risk (scaled 0 - 1)
        test_risk = (test_risk - np.min(test_risk)) / (np.max(test_risk) - np.min(test_risk))              # normalize test_risk (scaled 0 - 1)

        train_risk_Spline = make_interp_spline(order, train_risk)     # make_interp_spline is used to get a smooth curve
        test_risk_Spline = make_interp_spline(order, test_risk)

        x = np.linspace(order.min(), order.max(), 500)
        y = train_risk_Spline(x)

        y_t = test_risk_Spline(x)

        plt.plot(x, y, color = "blue")                    # train risk plot
        plt.plot(x, y_t, color = "orange")                # test risk plot
        plt.xlabel("order of the polynomial (d)")
        plt.ylabel("risk")
        plt.show()
            

def main():

    df = pd.read_csv(r"/Users/sunilkadam/Desktop/ML_HW/HW_1/Problem/problem1_csv.csv", dtype = float)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    
    '''
    x_train = np.array([1, 2, 3, 4])
    y_train = np.array([3, 5, 7, 9])
    '''

    x = x.reshape((-1,1))
    y = y.reshape((-1,1))

    x = (x - np.mean(x)) / np.std(x)                                        # standardize x
    y = (y - np.mean(y)) / np.std(y)                                        # standardize y

    x_train, x_test = np.split(x, [int(len(x)*0.7)])                        # split x into x_train (70% of x) and x_test (30% of x)
    y_train, y_test = np.split(y, [int(len(y)*0.7)])                        # split y into y_train (70% of y) and x_test (30% of y)



    '''
    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train)) # normalize x (scaled 0 - 1)
    y_train = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train)) # normalize y (scaled 0 - 1)

    '''

    # train_r_all = np.empty((350,1))
    # test_r_all = np.empty((150,1))

    train_r_all = []
    test_r_all = []

    # train model and get minimum train risk and minimum test risk
    # for i in range(1, 10):                                                      # fitting for polynimials from 1 to 9

    model = Poly_Reg(order = 5, lr = 0.001, iter = 100000)
    model.fit(x_train, y_train)                                             # train model on gradient descent


    x_train_sorted = np.sort(x_train, axis=0)
    y_pred_on_x_train_sorted = model.predict(x_train_sorted)

    x_test_sorted = np.sort(x_test, axis=0)
    y_pred_on_x_test_sorted = model.predict(x_test_sorted)    

    # train_r = model.get_risk(x_train, y_pred_on_x_train_sorted, y_train)      # train risk
    # test_r = model.get_risk(x_test, y_pred_on_x_test_sorted, y_test)          # test risk

                
    # train_r_all.append(np.min(train_r))
    # test_r_all.append(np.min(test_r))

    # create (9,3) numpy array of order, minimum train risk and minimum test risk. numpy array created to store in .csv
    # order = np.array([1,2,3,4,5,6,7,8,9])
    # order = order.reshape((-1,1))
    # train_r_all = np.array(train_r_all)
    # train_r_all = train_r_all.reshape((-1,1))
    # test_r_all = np.array(test_r_all)
    # test_r_all = test_r_all.reshape((-1,1))
    # d = np.empty((9,1))
    # d = np.append(d, order, axis = 1)
    # d = np.append(d, train_r_all, axis = 1)
    # d = np.append(d, test_r_all, axis = 1)
    # d = np.delete(d, obj = 0, axis = 1)

    # data_f = pd.DataFrame(d)
    # data_f.to_csv("/Users/sunilkadam/Desktop/ML_HW/HW_1/Solution/min_risk.csv", index = False, header = ["order", "min_train_risk", "min_test_risk"])


    # # train_r_all = np.delete(train_r_all, obj = 0, axis = 1)                         # remove extra firt column created by np.empty()
    # # test_r_all = np.delete(test_r_all, obj = 0, axis = 1)

    #model.plot_d_vs_risk()                                                            # plot order vs cost

    # model = Poly_Reg(order = i, lr = 0.001, iter = 100000)
    # model.fit(x_train, y_train)                                             # train model on gradient descent

    x_train_sorted = np.sort(x_train, axis=0)
    y_pred_on_x_train_sorted = model.predict(x_train_sorted)

    x_test_sorted = np.sort(x_test, axis=0)
    y_pred_on_x_test_sorted = model.predict(x_test_sorted)    

    train_r = model.get_risk(x_train, y_pred_on_x_train_sorted, y_train)      # train risk
    test_r = model.get_risk(x_test, y_pred_on_x_test_sorted, y_test)          # test risk

    plt.scatter(x_train, y_train, color = 'blue')
    plt.plot(x_train_sorted, y_pred_on_x_train_sorted, color = 'orange')
    plt.title('x vs y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    main()