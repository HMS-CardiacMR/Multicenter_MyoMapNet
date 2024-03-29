import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.pyplot as plt
from architecture.Unet import Unet
import torch
from torch.autograd import Variable
import numpy as np
import numpy.matlib
from tqdm import tqdm
import os
import re

def plot(img):
    plt.imshow(img)
    plt.show()

pixelDims = (10, 160, 160)

path = "Path_to_BIDMC_Data"
list_of_data = os.listdir(path)

model_name = "UNet_mixed"
print("Loading the model")
net = Unet(10, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
net.load_state_dict(torch.load("model/" + model_name +".model"))

f = open("BIDMC.txt", "w")
i = 1

for element in tqdm(list_of_data):
    name = re.findall("[0-9]*", element)[0]
    path_to_data = path + name
    img = np.load(path_to_data + "img.npy")[:4]
    img = img / img.max()
    inversion_time = np.load(path_to_data + "inversion_time_1.npy")[
                     :4] / 1000.0

    inversion_time = np.matlib.repmat(inversion_time, 160 * 160, 1)
    inversion_time = inversion_time.reshape(160, 160, 4)
    inversion_time = inversion_time.transpose(2, 0, 1)

    field_strength = np.matlib.repmat(3, 160 * 160, 1)
    field_strength = field_strength.reshape(160, 160, 1)
    field_strength = field_strength.transpose(2, 0, 1)

    vendor = np.matlib.repmat(1, 160 * 160, 1)
    vendor = vendor.reshape(160, 160, 1)
    vendor = vendor.transpose(2, 0, 1)

    images = np.zeros(pixelDims)
    images[:4] = img
    images[4:8] = inversion_time
    images[8] = field_strength
    images[9] = vendor

    images = images.reshape(1, 10, 160, 160)
    x = Variable(torch.FloatTensor(images)).to("cuda:0")

    mat_file = loadmat(path + element)
    t1_map = net(x)
    t1_map = t1_map.cpu().data.numpy()


    mask_blood = mat_file.get("mask_blood")
    mask_myocardium = mat_file.get("mask_myocardium")

    blood = t1_map * mask_blood
    mean_blood = np.sum(blood) / np.count_nonzero(blood)
    std_blood = np.nanstd(np.where(np.isclose(blood, 0), np.nan, blood))

    myocardium = t1_map * mask_myocardium
    mean_myo = np.sum(myocardium) / np.count_nonzero(myocardium)
    std_myo = np.nanstd(np.where(np.isclose(myocardium, 0), np.nan, myocardium))

    molli = loadmat(path + element)
    molli = molli.get("TxMap")
    difference = t1_map[0, 0, :, :] - molli

    if(not os.path.exists(path_to_save_results)):
        os.makedirs(path_to_save_results)
        
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 1
    
    fig.add_subplot(rows, columns, 1)
    plt.imshow(molli, cmap="jet", vmin=0, vmax=2000)
    plt.axis("off")
    
    fig.add_subplot(rows, columns, 2)
    plt.imshow(t1_map[0, 0, :, :], cmap="jet", vmax=2000, vmin=0)
    plt.axis("off")
    
    fig.add_subplot(rows, columns, 3)
    plt.imshow(difference, cmap="jet", vmin=-150, vmax=150)
    plt.axis("off")
    plt.savefig(path_to_save_results+str(i)+".eps")
    plt.close()
    i += 1

    f.write(str(round(mean_myo, 1)) + ", " + str(round(std_myo, 1)) + ", " +
            str(round(mean_blood, 1)) + ", " + str(round(std_blood, 1)) + "\n")

f.close()
