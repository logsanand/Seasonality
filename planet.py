import os, time, re, datetime
import numpy as np
from random import sample
import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from osgeo import gdal, ogr
from torchsummary import summary

#from ssim_loss import ssim
from models.AE_fully_convolutional_model_resnet_march2021_dual_64 import *    #we chose fully_conv or conv model
from codes.imgtotensor_patches_samples_list import ImageDataset
from codes.image_processing import extend, open_tiff
from codes.loader import dsloader
from codes.check_gpu import on_gpu
from codes.plot_loss import plotting
from codes.plot_loss_clusters import plot_clusters
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils   
#from pytorch_msssim import ssim

#create new directory if does not exist
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


print(torch.cuda.is_available())
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
gpu = on_gpu()
print("ON GPU is "+str(gpu))


#Parameters
patch_size = 64#224#128#256
bands_to_keep = 4#6#4  
bands=4#6#4 
epoch_nb = 1000#1000
batch_size = 32#32#20#20#120
learning_rate = 0.000001#0.0001#0.0005#0.005
weighted = False    # if we weight patches loss (center pixel has higher loss)
sigma = 2           # sigma for weighted loss
shuffle = True      # shuffle patches before training
satellite = "Plan" # ["S2"]
num_layers=50#50#18
pretrained=False
start_time = time.perf_counter()
run_name = "."+str(time.strftime("%Y-%m-%d_%H%M"))
print (run_name)


# We define the input and output paths
folder_results = "Res_50_4bands" + str(epoch_nb) + "_patch_" + str(patch_size) +"_fc"+ run_name
path_results = os.path.expanduser('/data/volume_2/look_space/results/Feature_extraction/model_results/'+str(satellite)+'_trdresmodel/') + folder_results + "/"

path_datasets = os.path.expanduser('/data/volume_2/look_space/autoenco_model/data/planet_texel/')#nomas_planet_noharm_pzt/')
create_dir(path_results)
path_model = path_results + 'model'+run_name+"/" #we will save the pretrained encoder/decoder models here
create_dir(path_model)


# We open all the images of time series images and mirror the borders.
# Then we create 4D array with all the images of the dataset
images_list = os.listdir(path_datasets)
path_list = []
#print(images_list)
list_image_extended= []
for image_name_with_extention in images_list:
    if image_name_with_extention.endswith(".tif") and not image_name_with_extention.endswith("band.TIF"):
        img_path = path_datasets + image_name_with_extention
        path_list.append(img_path)
        image_array, H, W, geo, proj, bands_nb = open_tiff(path_datasets, os.path.splitext(image_name_with_extention)[0])
        # We keep only essential bands if needed
        if bands_nb>4:
            #if satellite == "S2":
            image_array = np.delete(image_array, 4, axis=0)#4-SWIR
        #image_array = np.delete(image_array, 3, axis=0)
        #print("Height:",H)
        #print("Width:",W)
        toadd=H%patch_size
        #print(toadd)
        image_extended = image_array#extend(image_array, toadd)    # We mirror the border rows and cols
        list_image_extended.append(image_extended)
list_image_extended = list_image_extended#np.asarray(list_image_extended, dtype=float)
#bands_nb=3
# We normalize all the images from 0 to 1 (1 is the max value of the whole dataset, not an individual image)
#list_norm = []
#for band in range(len(list_image_extended[0])):
    #all_images_band = list_image_extended[:, band, :, :].flatten()
    #min = np.min(all_images_band)
    #max = np.max(all_images_band)
    #list_norm.append([min, max])
refval=[2.56390114339e-05,2.70297206283e-05,3.0135014507e-05,4.56099536666e-05]
for i in range(len(list_image_extended)):
    for band in range(len(list_image_extended[0])):
        min_im=0#-9999.0
        max_im=10000
        #list_image_extended[i][band]=(list_image_extended[i][band])*(refval[band])
        #list_image_extended[i][band][list_image_extended[i][band]<0]=-2
        #list_image_extended[i][band]=(list_image_extended[i][band]-min_im)/(max_im-min_im)
       # list_image_extended[i][band] = (list_image_extended[i][band]-np.min(list_image_extended[i][band]))/(np.max(list_image_extended[i][band])-np.min(list_image_extended[i][band]))


driver_tiff = gdal.GetDriverByName("GTiff")
driver_shp = ogr.GetDriverByName("ESRI Shapefile")


# We create a training dataset with patches
image = None   # Dataset with the sample of patches from all images
image1 = []

nbr_patches_per_image = int(H*W/len(list_image_extended))   # We sample H*W/ number of images patches from every image
# We create a dateset for every image separately and then we concatenate them
Xtrain = np.zeros(shape=(bands_nb, patch_size, patch_size, 0), dtype=np.float32)
for ii in range(len(list_image_extended)):

    image = ImageDataset(list_image_extended[ii], patch_size)  
    image1.append(image)	

image_all = torch.utils.data.ConcatDataset(image1)

print(len(image_all))
total_val=len(image_all)
train_count=int(0.8*total_val)
val_count=int(0.1*total_val)
test_count=total_val-train_count-val_count
train_data, validate_data, test_data = torch.utils.data.random_split(image_all,(train_count,val_count,test_count))
loader = dsloader(train_data, gpu, batch_size, shuffle=True) # dataloader
loader_val = dsloader(validate_data, gpu, batch_size, shuffle=True) # dataloader
loader_test = dsloader(test_data, gpu, batch_size, shuffle=True) # dataloader
print("There are {:d} training items {:d} validation items\n and {:d} Test items\n".format(len(train_data), len(validate_data), len(test_data)))
#image_all=None
#image = None


list_image_extended = None


# We write stats to file
with open(path_results+"stats.txt", 'a') as f:
    f.write("Relu activations for every layer except the last one. L2" + "\n")
    if weighted:
        f.write("sigma=" + str(sigma) + "\n")
    else:
        f.write("Loss not weighted,kl-0.0001" + "\n")
    f.write("patch_size=" + str(patch_size) + "\n")
    f.write("epoch_nb=" + str(epoch_nb) + "\n")
    f.write("batch_size=" + str(batch_size) + "\n")
    f.write("learning_rate=" + str(learning_rate) + "\n")
    f.write("bands_to_keep= " + str(bands_to_keep) + "\n")
    f.write("Nbr patches " + str(len(image_all)) + "\n")
    f.write("ResNetlayer" +str(num_layers)+"\n")
f.close()
image_all=None
image=None

writer = SummaryWriter('/data/volume_2/look_space/results/planetruns/resnet_experiment_1'+run_name+'/')

# Create model
resnet_vae=AutoEncoder(num_layers, pretrained, bands,patch_size)
resnet_vae.apply(initialize_weights)
if torch.cuda.device_count()>1:
   print("Lets use", torch.cuda.device_count(),"GPUs")
   resnet_vae=nn.DataParallel(resnet_vae)
model_params = list(resnet_vae.parameters())
optimizer = torch.optim.Adam(model_params, lr=learning_rate)

print(resnet_vae)
resnet_vae.to(device)


def loss_function(recon_x, x, mu, logvar):

    #MSE = F.binary_cross_entropy(recon_x.to(device='cuda:0'), x.to(device='cuda:0'), reduction='sum').to(device='cuda:0')
    mse=nn.MSELoss(reduction='sum')#sum
    MSE=mse(recon_x.to(device),x.to(device)).to(device)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).to(device)
   # ssim_val =ssim(x.to(device='cuda:0'),recon_x.to(device='cuda:0'),data_range=1,size_average=True).to(device='cuda:0')
    return  MSE + (0.0001*KLD)#+ssim_val #+ proj_loss.mean() #0.001-original

# We write the encoder model to stats
with open(path_results+"stats.txt", 'a') as f:
    f.write(str(resnet_vae) + "\n")
f.close()

start_time = time.perf_counter()


# Function to pretrain the model, pretty much standart
epoch_loss_list = []
val_loss_list=[]
for epoch in range(epoch_nb):
    resnet_vae.train()
    total_loss = 0
    for batch_idx, img_data in enumerate(loader,0):
        img_data = img_data.to(device)
        #print("img",img_data[:,:3,:,:])
        optimizer.zero_grad()
        decoded, z, mu, logvar,x_l1,x_l2,x_l3,d_l3,d_l4 = resnet_vae(img_data.to(device))  
        loss=loss_function(decoded.to(device),img_data.to(device),mu.to(device),logvar.to(device))
        total_loss += loss.item()
        #print("img2",img_data[:,:3,:,:])
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 10 == 0:#200
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(
                (epoch+1), (batch_idx+1) * batch_size, len(loader.dataset),
                100. * (batch_idx+1) / len(loader), loss.item()))
        if batch_idx%10==0:
            print(epoch*(len(loader))+batch_idx)
            img_data=img_data[:,:3,:,:]
            decoded=decoded[:,:3,:,:]
            #print(img_data)
            #print(decoded)
            writer.add_scalar('Training loss',total_loss,epoch*(len(loader))+batch_idx)
            writer.add_image('Train.1.Image', vutils.make_grid(img_data, nrow=4, normalize=True), epoch)           
            writer.add_image('Pred.1.Image', vutils.make_grid(decoded, nrow=4, normalize=True), epoch)
            x_l1=x_l1[:,:3,:,:]
            #x_l3=x_l3[:,:3,:,:]
            writer.add_image('Enc-64',vutils.make_grid(x_l1, nrow=4, normalize=True), epoch)
            #writer.add_image('Enc-1024',vutils.make_grid(x_l3, nrow=4, normalize=True), epoch)
            #d_l3=d_l3[:,:3,:,:]
            d_l4=d_l4[:,:3,:,:]
            #writer.add_image('Dec-64',vutils.make_grid(d_l3, nrow=4, normalize=True), epoch)
            writer.add_image('Dec-32',vutils.make_grid(d_l4, nrow=4, normalize=True), epoch)
    epoch_loss = total_loss / len(loader)
    epoch_loss_list.append(epoch_loss)
    epoch_stats = "Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch + 1, epoch_loss)
    print(epoch_stats)
    with open(path_results + "stats.txt", 'a') as f:
        f.write(epoch_stats+"\n")
    f.close()
    # we save the model
    if (epoch+1)%50==0:
        torch.save(resnet_vae.state_dict(), (path_model+'ae-model_ep_'+str(epoch+1)+"_loss_"+str(round(epoch_loss, 7))+run_name+'.pth'))  # save motion_encoder
    #torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))
    # We plot the loss values
    #if (epoch+1) % 5 == 0:
        #plotting(epoch+1, epoch_loss_list, path_results)
        #if epoch+1>20:
            #plotting(epoch+1,epoch_loss_list[20:],path_results)
        #else:
            #plotting(epoch+1,epoch_loss_list,path_results)
    #print("epochover")

    #model=AutoEncoder(num_layers, pretrained, bands_nb,patch_size)
    resnet_vae.eval()
    val_loss = 0
    all_X, all_y, all_z, all_mu, all_logvar = [], [], [], [], []
    with torch.no_grad():
        for val in loader_val:
            # distribute data to device
            X=val.to(device)
            X_reconst, zval, muval, logvarval,_,_,_,_,_ = resnet_vae(X)
            loss = loss_function(X_reconst, X, muval, logvarval)
            val_loss += loss.item()  # sum up batch loss
            writer.add_scalar('Validation loss',val_loss,epoch)
            #all_X.extend(X.data.cpu().numpy())
            #all_y.extend(y.data.cpu().numpy())
            #all_z.extend(z.data.cpu().numpy())
            #all_mu.extend(mu.data.cpu().numpy())
            #all_logvar.extend(logvar.data.cpu().numpy())
    val_loss /= len(validate_data)
    #all_X = np.stack(all_X, axis=0)
    #all_y = np.stack(all_y, axis=0)
    #all_z = np.stack(all_z, axis=0)
    #all_mu = np.stack(all_mu, axis=0)
    #all_logvar = np.stack(all_logvar, axis=0)
    val_loss_list.append(val_loss)
    # show information
    print('\nVal set ({:d} samples): Average loss: {:.4f}\n'.format(len(validate_data), val_loss))
    #return all_X, all_y, all_z, all_mu, all_logvar, val_loss
    #if (epoch+1) % 5 == 0:
        #plot_clusters(epoch+1,muval,X,path_results)
#for epoch in range(epoch_nb):
    #print(epoch)
    #train(epoch)
    #we plot loss values of training and testing:
    #if (epoch+1)%50 ==0:
        #plotting(epoch+1, epoch_loss_list,val_loss_list,path_results)
    with open(path_model+'lossvalues.txt','w') as f1:
        for item in zip(epoch_loss_list,val_loss_list):
            f1.write("%s\n" %str(item))


# Some stats about the best epoch loss and learning time
best_epoch = np.argmin(np.asarray(epoch_loss_list))+1
best_epoch_loss = epoch_loss_list[best_epoch-1]


print("best epoch " + str(best_epoch))
print("best epoch loss " + str(best_epoch_loss))


end_time = time.perf_counter()

total_time_learning = end_time - start_time
total_time_learning = str(datetime.timedelta(seconds=total_time_learning))
print("Total time learning =", total_time_learning)


with open(path_results+"stats.txt", 'a') as f:
    f.write("best epoch " + str(best_epoch) + "\n")
    f.write("best epoch loss " + str(best_epoch_loss) + "\n")
    f.write("Total time learning=" + str(total_time_learning) + "\n"+"\n")
f.close()

print("completed")
