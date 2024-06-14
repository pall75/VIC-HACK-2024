# libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
# import pdb
import random
import pdb

# PyTorch - neural network qMRI class
class Network_qmri_pt(nn.Module):
    def __init__(self, b_values, model, n_lat_params):
        super(Network_qmri_pt, self).__init__()
        self.b_values = b_values
        self.model = model
        self.n_lat_params = n_lat_params
        # network
        self.fc_layers = nn.ModuleList()
        for i in range(3):  # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values), len(b_values)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values), self.n_lat_params))
    #
    def forward(self, X):
        # X is is input data for batch
        # Xp is predicted (decoded) data for batch
        # params_pred are encoded parameters for the batch
        params_pred = torch.abs(self.encoder(X))  # (n_batch -by- n_params)
        Xp = self.model(params_pred, self.b_values)
        return Xp, params_pred


# Python function
def ricianloss_py(predictions, inputs, sigma):
    term1 = np.log(inputs / (sigma ** 2))
    term2 = -(inputs ** 2 + predictions ** 2) / (2 * (sigma ** 2))
    z = (inputs * predictions) / (sigma ** 2)
    I0e = scipy.special.i0e(z)
    lI0e = np.log(I0e)
    term3 = lI0e + z
    log_pdf = term1 + term2 + term3
    n_batch = inputs.shape[0]
    loss = -np.sum(log_pdf) / n_batch
    return loss

# PyTorch - train neural network
def train_network_pt(net, X_train, loss_fun, max_epochs=200, patience_n=15, netparams_init=[]):
    #
    # Data format
    X_train = torch.tensor(X_train).float()
    # Torch min and smallest
    torch_max = torch.finfo(torch.float).max
    torch_smallest = torch.finfo(torch.float).tiny
    #
    # Train network
    #
    # Loss function
    criterion = loss_fun
    # pdb.set_trace()
    #
    # Optimizer
    adam_lr = 0.001
    adam_betas = [0.9, 0.999]
    optimizer = optim.Adam(net.parameters(), betas=adam_betas, lr = adam_lr) #lr = 0.0001) # I'm assuming weight decay is 0
    """Create batches"""
    batch_size = 256
    # num_batches = len(X_train) // batch_size
    trainloader = utils.DataLoader(X_train,
                                    batch_size = batch_size,
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)
    #
    """## Train model"""
    # set parameters
    best = 1e20
    num_bad_epochs = 0
    train_losses = []
    # Initial network parameters (optional)
    if netparams_init: net.load_state_dict(netparams_init)
    # Train
    all_losses = []
    break_flag = 0
    for epoch in range(max_epochs):
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.
        for i, X_batch in enumerate(tqdm(trainloader), 0):
            # checking for 0's in batch
            if torch.sum(X_batch==0)>0:
                X_batch[X_batch == 0] = torch_smallest
                print('batch contains one or more <=0! Setting to smallest value.')
            # zero the parameter gradients
            optimizer.zero_grad()
            pdb.set_trace()
            # forward + backward + optimize
            X_pred, params_pred = net(X_batch)
            X_pred[torch.isinf(X_pred)] = torch_max
            X_pred[X_pred<=0] = torch_smallest # for IVIM NCC (see note below)
            # Training loss
            loss = criterion(X_pred, X_batch)
            all_losses += [loss.item()]
            # testing for nan/inf loss
            if torch.isinf(loss) or torch.isnan(loss):
                print('Break due to loss == nan/inf!')
                print(i)
                break_flag = 1
                breakpoint()
                # bugs for loss==nan (NCC)
                # loss=nan: when Fp>1, there can be -ve signals because 1-Fp<0 -> set minimum prediction to be 0 (Added above)
                # loss=inf: inside load_loss_fun('NCC'), torch_model -> minimum esb(z) made to be >0 to prevent underflow
                break
            loss.backward() # calculate gradients
            # testing for nan/inf gradient
            for param in net.parameters():
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    print('One or more Parameter gradient (weight matrix or bias table) == nan/inf!!!!!')
                    # In future should figure out why this is Nan or Inf and prevent this from occuring
                    print('Re-setting affected parameter gradient(s) (nan/inf) to 1 and continuing training....(!!!!!!!!!!!!!)')
                    param.grad[torch.isnan(param.grad)] = 1
                    param.grad[torch.isinf(param.grad)] = 1
                    # breakpoint()
                    print(i)
                    # break_flag = 2
                    # break
            if break_flag==2: break
            # Parameter update
            optimizer.step()
            running_loss += loss.item() # holds the total loss for the epoch
        #
        if break_flag==1 or break_flag==2: return tuple([break_flag for _ in range(4)])
        print("Loss: {}".format(running_loss))
        train_losses.append(running_loss)
        # early stopping
        if running_loss < best:
            print("Saving good model (Loss < best (" + str(best) + ")")
            final_net = net
            final_model = net.state_dict() # network parameter
            final_loss = running_loss
            num_bad_epochs = 0
            best = np.min(train_losses)
        else:
            print('running loss: ' + str(running_loss) + ", best: " + str(best) + "...incrementing bad epochs")
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience_n:
                print("Done (patience), best loss: {}".format(best))
                break
        if epoch == max_epochs:
            print("Done (max epochs), best loss: {}".format(best))
            break
    #
    print("Done")
    # Network parameters: final_model (a dictionary)
    # Network: net (a nn class)
    # Loss function value (final_loss)
    return final_net, final_model, final_loss, train_losses


def test_network_pt(net_trained, X_test_data):
    #
    # libraries
    import torch
    import numpy as np
    # Convert data to torch
    X_test_data = torch.tensor(X_test_data).float()
    #
    # Test parameter estimation
    n_data = X_test_data.shape[0]
    n_params = len(net_trained(X_test_data[0,:].unsqueeze(0))[1] .tolist()[0]) # len(gt_params[0])
    # param_errors = np.zeros(shape=(n_data, n_params)) * np.nan
    param_pred = np.zeros(shape=(n_data, n_params)) * np.nan
    X_test_pred = X_test_data * np.nan
    for i in range(n_data):
        Xi = X_test_data[i,:]
        X_pred, params = net_trained(Xi.unsqueeze(0))
        # param_errors[i,:] = np.array(params.tolist()[0]) - gt_params[i]
        param_pred[i, :] = params.tolist()[0]
        X_test_pred[i,:] = X_pred
    # Return: param prediction errors, signal predictions
    X_test_pred = X_test_pred.detach().numpy()
    return param_pred, X_test_pred

