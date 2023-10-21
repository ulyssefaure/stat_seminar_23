# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD
import math

# go through imports

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from matplotlib.collections import LineCollection

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state


# create neural networks :

class StdNNRegressionModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(StdNNRegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1))

    def forward(self, x):
        out = self.model(x)
        return out


class PositiveLayer(nn.Module):
    def __init__(self,input_dim=1, output_dim=1):
        super(PositiveLayer, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        exp_weights = torch.exp(0.05*self.layer.weight)
        out = torch.matmul(x, exp_weights.T)+self.layer.bias
        return out
class MonotoneNNModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(MonotoneNNModel, self).__init__()
        self.model = nn.Sequential(
            PositiveLayer(1, 20),
            PositiveLayer(20, 1))

    def forward(self, x):
        out = self.model(x)
        return out



#### create data

n = 100
x = np.arange(n)
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(n,)) + 50.0 * np.log1p(np.arange(n))
# (the function is 50 log (1+x) + \epsilon


###########
# Apply regression #

ir = IsotonicRegression(out_of_bounds="clip")
y_ = ir.fit_transform(x, y)

lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression


# apply NN model regression

nnreg = StdNNRegressionModel()
nnreg_monotone = MonotoneNNModel()
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(nnreg.parameters(), lr=0.01)
optimizer_monotone = optim.Adam(nnreg_monotone.parameters(), lr=0.05)

import tqdm
from sklearn.model_selection import train_test_split

# train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# training parameters
n_epochs = 2000  # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf  # init to infinity
best_mse_monotone = np.inf
best_weights = None
best_weights_monotone = None
history = []
history_monotone = []

# training loop
for epoch in range(n_epochs):
    nnreg.train()
    nnreg_monotone.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]
            # forward pass
            y_pred = nnreg(X_batch.reshape((len(X_batch),1)))
            y_pred_mon = nnreg_monotone(X_batch.reshape((len(X_batch),1)))
            loss = loss_fn(y_pred, y_batch)
            loss_mon = loss_fn(y_pred_mon, y_batch)
            # backward pass
            optimizer.zero_grad()
            optimizer_monotone.zero_grad()
            loss.backward()
            loss_mon.backward()
            # update weights
            optimizer.step()
            optimizer_monotone.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    nnreg.eval()
    nnreg_monotone.eval()

    y_pred = nnreg(X_test.reshape((len(X_test),1)))
    y_pred_mon = nnreg_monotone(X_test.reshape((len(X_test),1)))
    mse = loss_fn(y_pred, y_test)
    mse_mon = loss_fn(y_pred_mon, y_test)
    mse, mse_mon = float(mse), float(mse_mon)


    history.append(mse)
    history_monotone.append(mse_mon)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(nnreg.state_dict())
    if mse_mon < best_mse_monotone:
        best_mse_monotone = mse_mon
        best_weights_monotone = copy.deepcopy(nnreg_monotone.state_dict())

# restore model and return best accuracy
nnreg.load_state_dict(best_weights)
nnreg_monotone.load_state_dict(best_weights_monotone)


###########
# PLOT #

segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(np.full(n, 0.5))
x_as_tensor = torch.tensor(x, dtype=torch.float32).reshape(len(x), 1)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))

ax0.plot(x, y, "C0.", markersize=12)
ax0.plot(x, y_, "C1.-", markersize=12)

ax0.plot(x, nnreg(x_as_tensor).detach(), markersize=12, color="black")
ax0.plot(x, nnreg_monotone(x_as_tensor).detach(), markersize=12, color="green")

ax0.add_collection(lc)
ax0.legend(("Training data", "Isotonic fit", "NN fit", "Monotone NN fit"), loc="lower right")
ax0.set_title("Isotonic regression fit on noisy data (n=%d)" % n)

x_test = np.linspace(0, 110, 1000)
x_test_as_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(len(x_test), 1)

ax1.plot(x_test, ir.predict(x_test), "C1-")
ax1.plot(ir.X_thresholds_, ir.y_thresholds_, "C1.", markersize=12)
ax1.plot(x_test, nnreg(x_test_as_tensor).detach(), color="black")
ax1.set_title("Prediction function (%d thresholds)" % len(ir.X_thresholds_))




plt.show()


# print parameters :

print ("layer 0 : " , nnreg_monotone.model[0].layer.weight)
print ("layer 1 : ", nnreg_monotone.model[1].layer.weight)



