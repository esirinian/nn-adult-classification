import pandas as pd
import numpy as np
from scipy import stats

adult_data = pd.read_csv('adult_data_cleaned.csv')
adult_data = adult_data.drop(adult_data.columns[0], axis=1)

# Separating train from test
# X, Y for training
# xx, yy for testing

train = adult_data.loc[:20000,:]
X = train[['Age','Workclass','Education','Marital Status','Occupation','Relationship','Race','Sex','Capital Gain',
               'Capital Loss','Hours per Week','GDP per Cap']]
Y = train['Salary'].values.reshape(-1, 1)

test = adult_data.loc[20000:,:]
xx = test[['Age','Workclass','Education','Marital Status','Occupation','Relationship','Race','Sex','Capital Gain',
               'Capital Loss','Hours per Week','GDP per Cap']]
yy = test['Salary'].values.reshape(-1, 1)


# Impact encoding
categorical_variables = ['Workclass','Education','Marital Status','Occupation','Relationship','Race']

temp = train.loc[train['Salary'] == 1]
for category in categorical_variables:
    unique_vals = list(train[category].unique())
    x_value_counts = dict(temp[category].value_counts())
    missing_vals = list(set(unique_vals).difference(list(x_value_counts.keys())))
    for missing in missing_vals:
        x_value_counts.update({missing: 0})
    y_value_counts = dict(X[category].value_counts())
    likelihood = {k: x_value_counts[k] / y_value_counts[k] for k in y_value_counts if k in x_value_counts}
    adult_data[category].replace(likelihood, inplace=True)

X = X.apply(stats.zscore)
xx = xx.apply(stats.zscore)

X['Constant'] = train[['Constant']]
xx['Constant'] = test[['Constant']]

def myTanH(h):
    return np.tanh(h)

class NeuralNetwork:
    def __init__(self,x,y):
        
        # Input takes on value of 2d x array
        self.input = x.values
        
        # 2 layers, 4 neurons in hidden layer, 1 neuron for output
        self.layer1W = np.ones((self.input.shape[1],2)) / 2
        self.layer2W = np.ones((2,5)) / 2
        self.layer3W = np.ones((5,3)) / 2
        self.layer4W = np.ones((3,1)) / 2
    
        # Actual classification
        self.y = y
        
        # Predicted classification
        self.output = np.zeros(y.shape)
        
    def feedforward(self, test = None):
        if test is None:
            self.layer1R = myTanH(np.dot(self.input, self.layer1W))
        else:
            self.layer1R = myTanH(np.dot(test.values, self.layer1W))
            
        self.layer2R = myTanH(np.dot(self.layer1R, self.layer2W))
        self.layer3R = myTanH(np.dot(self.layer2R, self.layer3W))
        self.output = myTanH(np.dot(self.layer3R, self.layer4W))
        
    def backprop(self):
        epsilon = 0.3
        
        delta_W41 = epsilon * (1 - np.square(self.output)) * (self.y - self.output) * self.layer3R[:, 0].reshape(-1,1)
        delta_W42 = epsilon * (1 - np.square(self.output)) * (self.y - self.output) * self.layer3R[:, 1].reshape(-1,1)
        delta_W43 = epsilon * (1 - np.square(self.output)) * (self.y - self.output) * self.layer3R[:, 2].reshape(-1,1)
        
        for delta1, delta2, delta3 in zip(delta_W41,delta_W42,delta_W43):
            self.layer4W[0] = self.layer4W[0] + delta1
            self.layer4W[1] = self.layer4W[1] + delta2
            self.layer4W[2] = self.layer4W[2] + delta3
        
        top_error = (1 - np.square(self.output)) * (self.y - self.output)

        ec31 = np.sum(np.array([self.layer4W[0] * e for e in top_error]))
        ec32 = np.sum(np.array([self.layer4W[1] * e for e in top_error]))
        ec33 = np.sum(np.array([self.layer4W[2] * e for e in top_error]))
        
        # All connections to neuron 3,1
        delta_W31 = epsilon * (1 - np.square(self.layer3R[:,0].reshape(-1,1))) * ec31 * self.layer2R
        
        # All connections to neuron 3,2
        delta_W32 = epsilon * (1 - np.square(self.layer3R[:,1].reshape(-1,1))) * ec32 * self.layer2R
        
        delta_W33 = epsilon * (1 - np.square(self.layer3R[:,2].reshape(-1,1))) * ec32 * self.layer2R
        
        for delta1, delta2, delta3 in zip(delta_W31, delta_W32, delta_W33):
            self.layer3W[:,0] = self.layer3W[:,0] + delta1
            self.layer3W[:,1] = self.layer3W[:,1] + delta2
            self.layer3W[:,2] = self.layer3W[:,2] + delta3
        
        errorL31 = (1 - np.square(self.layer3R[:,0].reshape(-1,1))) * ec31 
        errorL32 = (1 - np.square(self.layer3R[:,1].reshape(-1,1))) * ec32
        errorL33 = (1 - np.square(self.layer3R[:,2].reshape(-1,1))) * ec33
        self.errorL3 = np.column_stack((errorL31, errorL32, errorL33))
        
        
        ec21 = np.sum(np.array([self.layer3W[0,:] * e for e in self.errorL3]))
        ec22 = np.sum(np.array([self.layer3W[1,:] * e for e in self.errorL3]))
        ec23 = np.sum(np.array([self.layer3W[2,:] * e for e in self.errorL3]))
        ec24 = np.sum(np.array([self.layer3W[3,:] * e for e in self.errorL3]))
        ec25 = np.sum(np.array([self.layer3W[4,:] * e for e in self.errorL3]))
        
        delta_W21 = epsilon * (1 - np.square(self.layer2R[:,0].reshape(-1,1))) * ec21 * self.layer1R
        delta_W22 = epsilon * (1 - np.square(self.layer2R[:,1].reshape(-1,1))) * ec22 * self.layer1R
        delta_W23 = epsilon * (1 - np.square(self.layer2R[:,2].reshape(-1,1))) * ec23 * self.layer1R
        delta_W24 = epsilon * (1 - np.square(self.layer2R[:,3].reshape(-1,1))) * ec24 * self.layer1R
        delta_W25 = epsilon * (1 - np.square(self.layer2R[:,4].reshape(-1,1))) * ec25 * self.layer1R
        
        for delta1, delta2, delta3, delta4, delta5 in zip(delta_W21, delta_W22, delta_W23, delta_W24, delta_W25):
            self.layer2W[:,0] = self.layer2W[:,0] + delta1
            self.layer2W[:,1] = self.layer2W[:,1] + delta2
            self.layer2W[:,2] = self.layer2W[:,2] + delta3
            self.layer2W[:,3] = self.layer2W[:,3] + delta4
            self.layer2W[:,4] = self.layer2W[:,4] + delta5
            
        errorL21 = (1 - np.square(self.layer2R[:,0].reshape(-1,1))) * ec21
        errorL22 = (1 - np.square(self.layer2R[:,1].reshape(-1,1))) * ec22
        errorL23 = (1 - np.square(self.layer2R[:,2].reshape(-1,1))) * ec23
        errorL24 = (1 - np.square(self.layer2R[:,3].reshape(-1,1))) * ec24 
        errorL25 = (1 - np.square(self.layer2R[:,4].reshape(-1,1))) * ec25
        self.errorL2 = np.column_stack((errorL21, errorL22, errorL23, errorL24, errorL25))
        
        ec11 = np.sum(np.array([self.layer2W[0,:] * -e for e in self.errorL2]))
        ec12 = np.sum(np.array([self.layer2W[1,:] * -e for e in self.errorL2]))
        
        delta_W11 = epsilon * (1 - np.square(self.layer1R[:,0].reshape(-1,1))) * ec11 * self.input
        delta_W12 = epsilon * (1 - np.square(self.layer1R[:,1].reshape(-1,1))) * ec11 * self.input
        for delta1, delta2 in zip(delta_W11, delta_W12):
            self.layer1W[:,0] = self.layer1W[:,0] + delta1
            self.layer1W[:,1] = self.layer1W[:,1] + delta2
        
        
# nn = NeuralNetwork(X, Y)
# for i in range(1500):
#     nn.feedforward()
#     nn.backprop()
    
    
# nn.feedforward(xx)
# p = nn.output
# p = (p + 1) / 2
# count = 0
# for pred, act in zip(p, yy):
#     if pred == act:
#         count = count + 1
# print(count / len(yy))