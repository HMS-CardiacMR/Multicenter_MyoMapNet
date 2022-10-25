import torch
import random
from torch.optim import Adam
from torch.nn.modules.loss import L1Loss
from torch.autograd import Variable
from architectures.Unet import Unet, save_models
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

nb_epochs = 3000
batch_size = 64
early_stopping = 70
learning_rate = 0.001

model_name = "UNet_mixed"

seed_num = 1964

torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)

print("Loading BMC data")
x_train_BMC = np.load("Data/train_BMC_t1_weighted.npy")
y_train_BMC = np.load("Data/train_BMC_t1_maps.npy")
x_train_BMC = shuffle(x_train_BMC, random_state=seed_num)
y_train_BMC = shuffle(y_train_BMC, random_state=seed_num)

x_validation_BMC = np.load("Data/validation_BMC_t1_weighted.npy")
y_validation_BMC = np.load("Data/validation_BMC_t1_maps.npy")

print("Loading Cornell data")
x_train_Cornell = np.load("Data/train_Cornell_t1_weighted.npy")
y_train_Cornell = np.load("Data/train_Cornell_t1_maps.npy")
x_train_Cornell = shuffle(x_train_Cornell, random_state=seed_num)
y_train_Cornell = shuffle(y_train_Cornell, random_state=seed_num)

x_validation_Cornell = np.load("Data/validation_Cornell_t1_weighted.npy")
y_validation_Cornell = np.load("Data/validation_Cornell_t1_maps.npy")

print("Loading BIDMC data")
x_train_BIDMC = np.load("Data/train_BIDMC_t1_weighted.npy")
y_train_BIDMC = np.load("Data/train_BIDMC_t1_maps.npy")
x_train_BIDMC = shuffle(x_train_BIDMC, random_state=seed_num)
y_train_BIDMC = shuffle(y_train_BIDMC, random_state=seed_num)
x_validation_BIDMC = np.load("Data/validation_BIDMC_t1_weighted.npy")
y_validation_BIDMC = np.load("Data/validation_BIDMC_t1_maps.npy")

print("Concatinating data")
x_train = np.concatenate((x_train_BIDMC, x_train_BMC, x_train_Cornell), axis=0)
y_train = np.concatenate((y_train_BIDMC, y_train_BMC, y_train_Cornell), axis=0)
x_train = shuffle(x_train, random_state=seed_num)
y_train = shuffle(y_train, random_state=seed_num)

x_validation = np.concatenate((x_validation_BIDMC, x_validation_BMC, x_validation_Cornell), axis=0)
y_validation = np.concatenate((y_validation_BIDMC, y_validation_BMC, y_validation_Cornell), axis=0)
x_validation = shuffle(x_validation, random_state=seed_num)
y_validation = shuffle(y_validation, random_state=seed_num)


print("x train shape: ", x_train.shape)
print("y train shape: ", y_train.shape)
print("x validation shape: ", x_validation.shape)
print("y validation shape: ", y_validation.shape)

total_size_training_data = x_train.shape[0]
total_size_validation_data = x_validation.shape[0]

cuda_available = torch.cuda.is_available()
net = Unet(10, 1)
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
loss_nn = L1Loss()
best_training_loss = 0
no_loss_improvement = 0
global_training_loss_list = []
global_validation_loss_list = []
epoch_numbers = []

print("Start training")

for epoch in range(1, nb_epochs):
    epoch_numbers.append(epoch)
    training_loss = 0
    training_accuracy = 0

    nb_batches = 0
    for index in tqdm(range(0, total_size_training_data, batch_size)):
        x = Variable(torch.FloatTensor(x_train[index: index+batch_size])).to("cuda:0")
        y = Variable(torch.FloatTensor(y_train[index: index + batch_size])).to("cuda:0")
        y = y.reshape(y.shape[0], 160, 160)
        optimizer.zero_grad()
        y_pred = net(x)
        y_pred = y_pred.reshape(y_pred.shape[0], 160, 160)
        loss = loss_nn(y_pred, y)

        loss.backward()
        optimizer.step()


        training_loss += loss.cpu().data

        y_pred = y_pred.cpu().data.numpy()
        y = y.cpu().data.numpy()

        nb_batches += 1

    avg_training_loss = training_loss / nb_batches
    no_loss_improvement += 1
    global_training_loss_list.append(avg_training_loss)

    validation_loss = 0

    nb_batches = 0
    with torch.no_grad():

        for index in range(0, x_validation.shape[0], batch_size):
            x_val = Variable(torch.FloatTensor(x_validation[index: index + batch_size])).to("cuda:0")
            y_val = Variable(torch.FloatTensor(y_validation[index: index + batch_size])).to("cuda:0")
            y_val = y_val.reshape(y_val.shape[0], 160, 160)

            y_pred_val = net(x_val)
            y_pred_val = y_pred_val.reshape(y_pred_val.shape[0], 160, 160)

            loss_val = loss_nn(y_pred_val, y_val)
            validation_loss += loss_val.cpu().data

            nb_batches += 1

    avg_validation_loss = validation_loss / nb_batches
    global_validation_loss_list.append(avg_validation_loss)

    print("Epoch {} - training loss: {} Validation loss: {}".format(epoch, avg_training_loss, avg_validation_loss))


    if (epoch == 1 or avg_validation_loss < best_validation_loss):
        save_models(path_to_save_weights, net, model_name, epoch)
        best_validation_loss = avg_validation_loss
        no_loss_improvement = 0

    else:
        print("Loss did not improve from {}".format(best_validation_loss))
    if(no_loss_improvement == early_stopping):
        print("Early stopping - no improvement afer {} iterations of training".format(early_stopping))
        break

# Curve for training and validation
plt.plot(epoch_numbers, global_training_loss_list)
plt.plot(epoch_numbers, global_validation_loss_list)
plt.xlabel("epoch")
plt.ylabel("loss (ms)")
plt.legend(["training", "validation"], loc="upper right")
plt.savefig(path_to_save_learning_curve)
plt.close()