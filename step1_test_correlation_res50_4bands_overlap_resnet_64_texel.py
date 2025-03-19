import os, time, re, datetime
import numpy as np
from random import sample
import sys
import torch
from torch import nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
from osgeo import gdal, ogr
import scipy.ndimage.filters as fi
from torchsummary import summary
import seaborn as sns
from scipy import signal
#import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd
from models.AE_fully_convolutional_model_resnet_march2021_64 import *    #we chose fully_conv or conv model
from codes.imgtotensor_patches_samples_list import ImageDataset_test
from codes.image_processing import extend, open_tiff
from codes.loader import dsloader
from codes.check_gpu import on_gpu
from codes.plot_loss import plotting
from codes.plot_loss_clusters import plot_clusters
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils   
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm

#create new directory if does not exist
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

start_time = time.perf_counter()
run_name = "."+str(time.strftime("%Y-%m-%d_%H%M"))
print (run_name)
#parameters
satellite='S2'
patch_size=64
num_layers=50#50
pretrained=False
bands_to_keep = 4#6#4 #5
bands=4#6#4 #5
#5
gpu=False
#Full image size 3994*8623

PATH=r"D:\Utrecht_macrozoobenthos_data_feb2021\Data_planetscope\Texel_new\model_result/"
# We define the input and output paths
test_results = os.path.expanduser(r'D:\Utrecht_macrozoobenthos_data_feb2021\Data_planetscope\Texel_new\test_result') + "/"
create_dir(test_results)
path_datasets = os.path.expanduser(r'D:\Utrecht_macrozoobenthos_data_feb2021\Data_planetscope\Texel_new\ref\1/')

images_list = os.listdir(path_datasets)
path_list = []

#stats file
with open(test_results+'parameters.txt', 'w') as f:
    f.write("ae-model_ep_350_loss_4.6757665.2024-08-28_1717.pth")
    f.write("Bands used-B,G,R,NIR")
#print(images_list)
list_image_extended= []
for image_name_with_extention in images_list:
    if image_name_with_extention.endswith(".tif") and not image_name_with_extention.endswith("band.TIF"):
        img_path = path_datasets + image_name_with_extention
        path_list.append(img_path)
        image_array, H, W, geo, proj, bands_nb = open_tiff(path_datasets, os.path.splitext(image_name_with_extention)[0])
        if bands_nb>4:
        #print(toadd)
            image_array =  np.delete(image_array, 4, axis=0)#0 for swir #4 for b #image_array#extend(image_array, toadd)    # We mirror the border rows and cols
        image_extended = image_array#extend(image_array, toadd)    # We mirror the border rows and cols
        list_image_extended.append(image_extended)
#list_image_extended = list_image_extended#np.asarray(list_image_extended, dtype=float)
bands_nb=4#5#6#4

#for i in range(len(list_image_extended)):
    #for band in range(len(list_image_extended[i])):
        #print(np.min(list_image_extended[i][band]))		
        #print(np.max(list_image_extended[i][band]))	
        #max=-9999.0
        #min_im=0#0
        #max_im=10000
        #list_image_extended[i][band] = (list_image_extended[i][band]-np.min(list_image_extended[i][band]))/(np.max(list_image_extended[i][band])-np.min(list_image_extended[i][band]))
        #list_image_extended[i][band] = (list_image_extended[i][band]-min_im)/(max_im-min_im)		
        #print(np.min(list_image_extended[i][band]))		
        #print(np.max(list_image_extended[i][band]))	

def find_min_withoutzero(array):
    plus_array = array[np.nonzero(array)]
    min_elem = min(plus_array)
    return min_elem

def scatter(layer1, layer2,bandname1,bandname2,count):
  
    plt.scatter(layer1,layer2,color='red')
    min_lay1=find_min_withoutzero(layer1)
    min_lay2=find_min_withoutzero(layer2)
    #print(min_lay1)
    max_lay1=layer1.max()
    max_lay2=layer2.max()
    #print(max_lay1)
    min_lay=np.minimum(min_lay1,min_lay2)
    max_lay=np.maximum(max_lay1,max_lay2)
    plt.xlim(min_lay,max_lay)
    plt.ylim(min_lay,max_lay)
    plt.xlabel(bandname1)
    plt.ylabel(bandname2)
    #plt.show()
    

for i in range(len(list_image_extended)):
    plt.imshow(list_image_extended[0][1,:,:])
    plt.title("Single band image")
    plt.show()
    print(list_image_extended[0].shape)
    list_image_extended=np.moveaxis(list_image_extended[i],-1,0)
    list_image_extended=np.moveaxis(list_image_extended,-1,0)
    original_image=list_image_extended[:,:,:3]
    # plot original image after normalisation
    plt.imshow(original_image)
    plt.title("Three band image")
    plt.show()
    print(np.min(list_image_extended[:,:,1]))		
    print(np.max(list_image_extended[:,:,1]))	
    ## plot scatter plot for all band combinations
    # count=0  
    # fig=plt.figure(figsize=(4,4))	
    # columns_img = 4
    # rows_img = 4
    # for b in range(bands_nb):
        # for k in range(bands_nb):
            # count=count+1
            # fig.add_subplot(rows_img, columns_img, count)
            # scatter(list_image_extended[:,:,b],list_image_extended[:,:,k],"Layer"+"_"+str(b),"Layer"+"_"+str(k),count)
    # plt.show()
    # stretching the bands
    for b in range(bands_nb):   
        #plt.hist(list_image_extended[:,:,b])
        #plt.show()
        val_nonzero =	list_image_extended[:,:,b][np.nonzero(list_image_extended[:,:,b])]
        mean_layer=np.mean(val_nonzero)
        #print(mean_layer)
        std_layer=np.std(val_nonzero)
        #print(std_layer)
        minimum_layer =np.min(val_nonzero)
        print(minimum_layer)
        maximum_layer=np.max(list_image_extended[:,:,b][list_image_extended[:,:,b]!=1])
        #print(maximum_layer)
        percent_lower=mean_layer-(std_layer*2)
        percent_upper=mean_layer+(std_layer*2)
        #list_image_extended[:,:,b][list_image_extended[:,:,b]<minimum_layer]=minimum_layer
        #list_image_extended[:,:,b][list_image_extended[:,:,b]>0.8]=0.8
        #list_image_extended[:,:,b]=(list_image_extended[:,:,b]-minimum_layer)/(maximum_layer-minimum_layer)
    #plt.imshow(list_image_extended[:,:,0],vmin=0,vmax=1,cmap='Reds')
    #plt.imshow(list_image_extended[:,:,1],vmin=0,vmax=1,cmap='Greens')
    #list_image_extended[:,:,2]=0   
    #plt.imshow(list_image_extended[:,:,:3],vmin=0,vmax=1,cmap='Blues')
    #plt.clim(vmin=0, vmax=1)
    #plt.cm.jet()
    #vmin, vmax = plt.gci().get_clim()
    #print('vmin',vmin)
    #print('vmax',vmax)
    #plt.show()
print(list_image_extended.shape)
# store the original image after normalisation for better visualisation
driver = gdal.GetDriverByName('GTiff')
newRaster = driver.Create(test_results+'original_image'+'_range'+'.tif',list_image_extended.shape[1],
list_image_extended.shape[0], bands_nb, gdal.GDT_Float32)
prj1 = proj # define the new raster dataset projection and geotrasform
newRaster.SetProjection(prj1)
gt = geo
newTLCoord=gdal.ApplyGeoTransform(gt, 0, 0)
image_fin=list_image_extended
newRaster.SetGeoTransform([newTLCoord[0], 3, 0, newTLCoord[1], 0, -3])
newBand1 = newRaster.GetRasterBand(1) 
newBand1.WriteArray(image_fin[:,:,0])
newBand2 = newRaster.GetRasterBand(2) 
newBand2.WriteArray(image_fin[:,:,1])
newBand3 = newRaster.GetRasterBand(3) 
newBand3.WriteArray(image_fin[:,:,2])
newBand4= newRaster.GetRasterBand(4) 
newBand4.WriteArray(image_fin[:,:,3]) 
#newBand5= newRaster.GetRasterBand(5) 
#newBand5.WriteArray(image_fin[:,:,4]) 
#newBand6= newRaster.GetRasterBand(6) 
#newBand6.WriteArray(image_fin[:,:,5]) 
print("saved original image")

#Plot correlation table
def corr_plot(corr_val,band_val,title):
    fig, ax =plt.subplots(1,1)
    column_labels=["Layer_"+str(num) for num in range(band_val)]
    row_labels=["Layer_"+str(num) for num in range(band_val)]
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=corr_val,rowLabels=row_labels,colLabels=column_labels,rowColours =["yellow"] * band_val, colColours =["yellow"] * band_val,loc="center")
    plt.title(title)
    plt.show()
#correlation between bands 
Width=list_image_extended.shape[1]
Height=list_image_extended.shape[0]
corr_array1=np.zeros((bands_nb,bands_nb))
for b in range(bands_nb):
    for k in range(bands_nb):
        val_nonzero1 =	list_image_extended[:,:,b][list_image_extended[:,:,b]>0]
        val_nonzero2 =	list_image_extended[:,:,k][list_image_extended[:,:,k]>0]
        Layer1=list_image_extended[:,:,b]
        Layer2=list_image_extended[:,:,k]
        #layer1_mean=np.mean(list_image_extended[:,:,b])
        #layer2_mean=np.mean(list_image_extended[:,:,k])	
        #layer1_std=np.std(list_image_extended[:,:,b])
        #layer2_std=np.std(list_image_extended[:,:,k])
        layer1_mean=np.mean(val_nonzero1)
        layer2_mean=np.mean(val_nonzero2)	
        layer1_std=np.std(val_nonzero1)
        layer2_std=np.std(val_nonzero2)
        #cov=(Layer1-layer1_mean)*(Layer2-layer2_mean)
        cov=(val_nonzero1-layer1_mean)*(val_nonzero2-layer2_mean)
        covariance=np.mean(cov)#/((Height*Width)-1)
        correlation=covariance/(layer1_std*layer2_std)
        corr_array1[b,k]=correlation
corr_plot(corr_array1,bands_nb,"Correlation_full_image")
        #print("correlation:"+str(b)+','+str(k),correlation)

b=None
k=None
covariance=None
cov=None
correlation=None	

#subset of images			
list_image_extended=list_image_extended[1000:5000,50:3000,:]
#list_image_extended=list_image_extended[2500:2700,2700:2900,:]
#list_image_extended=list_image_extended[1000:1200,6000:6200,:]
#list_image_extended=list_image_extended[0:1500,7000:8623,:]#subset1(top part)
#list_image_extended=list_image_extended[600:900,7000:8000,:]
#list_image_extended=list_image_extended[0:1300,6500:8623,:]#subset1(top part)#1300*2163#UFV-final
#list_image_extended=list_image_extended[500:800,7500:8400,:]#UFV-Example
#list_image_extended=list_image_extended[400:2000,700:2800,:]#UGV
#list_image_extended=list_image_extended[700:1000,7000:7500,:]#subset1(top part)#700 is rows,#7000 is columns
plt.imshow(list_image_extended[:,:,:3])
plt.show()
px=1000#400#0
py=50#700#6500
print(list_image_extended.shape)

# correlation between bands subset 
corr_array2=np.zeros((bands_nb,bands_nb))
for b in range(bands_nb):
    for k in range(bands_nb):
        val_nonzero1 =	list_image_extended[:,:,b][list_image_extended[:,:,b]>0]
        val_nonzero2 =	list_image_extended[:,:,k][list_image_extended[:,:,k]>0]
        Layer1=list_image_extended[:,:,b]
        Layer2=list_image_extended[:,:,k]
        #layer1_mean=np.mean(list_image_extended[:,:,b])
        #layer2_mean=np.mean(list_image_extended[:,:,k])	
        #layer1_std=np.std(list_image_extended[:,:,b])
        #layer2_std=np.std(list_image_extended[:,:,k])
        layer1_mean=np.mean(val_nonzero1)
        layer2_mean=np.mean(val_nonzero2)	
        layer1_std=np.std(val_nonzero1)
        layer2_std=np.std(val_nonzero2)
        #cov=(Layer1-layer1_mean)*(Layer2-layer2_mean)
        cov=(val_nonzero1-layer1_mean)*(val_nonzero2-layer2_mean)
        covariance=np.mean(cov)#/((Height*Width)-1)
        correlation=covariance/(layer1_std*layer2_std)
        corr_array2[b,k]=correlation
corr_plot(corr_array2,bands_nb,"Correlation_subset_image")
        #print("correlation_subset:"+str(b)+','+str(k),correlation)

# Model Testing
print("Model Testing")
resnet_vae=AutoEncoder(num_layers, pretrained, bands,patch_size)
resnet_vae.load_state_dict({k.replace('module.',''):v for k,v in torch.load('{}/ae-model_ep_350_loss_4.6757665.2024-08-28_1717.pth'.format(PATH),map_location='cpu').items()})#we have added new lines now or use strict=False newly added for research cloud  because of nn.parallel
print("model loaded")
#resnet_vae.load_state_dict(torch.load('{}/ae-model_ep_900_loss_14.0524052.2021-06-03_1818.pth'.format(PATH),map_location='cpu'))
resnet_vae.eval()
print("evaluating")

batch_size=2
num_filters=4#6#4#5
H=list_image_extended.shape[0]
W=list_image_extended.shape[1]
input_rows=H
input_cols=W
in_rows_half = int(H/2)
in_cols_half = int(W/2)
patch_size=64
input_channels=4#6#4#5
pad_r=(64-(H%40))
pad_c=(64-(W%40))
p1_r=pad_r//2
p2_r=pad_r-(p1_r)
p1_c=pad_c//2
p2_c=pad_c-(p1_c)
image=np.pad(list_image_extended,((p1_r,p2_r),(p1_c,p2_c),(0,0)),'edge')
print("Image shape",image.shape)
plt.imshow(image[:,:,3])
plt.show()
num_rows=image.shape[0]
num_cols=image.shape[1]
no_patches=((num_rows//40))*((num_cols//40))#40+24, 76*84=6324 (3024*3064)
image_final = np.zeros((num_rows, num_cols, num_filters))
row_images = np.zeros((no_patches, patch_size,patch_size, input_channels))
patchsize=patch_size
coun=0
y=(num_cols/40)
x=(num_rows/40)
for i in range(int(x)):

    for j in range(int(y)):
                # cut small image patch
        print(y)
        #row_images[coun,...] = image[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size, :]
        row_images[coun,...] = image[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]
        coun=coun+1
        print(row_images.shape)
 
print("patches shape",row_images.shape)	
#moved the axis
row_images2=np.moveaxis(row_images,-1,1)
no_patches=coun+1
#creating empty array

input_image = np.zeros((no_patches, patch_size,patch_size, input_channels))
recons_output = np.zeros((no_patches, patch_size,patch_size, input_channels))
#deco64_output = np.zeros((no_patches, patch_size,patch_size, 64))
enco64_output = np.zeros((no_patches, patch_size,patch_size, 64))
#enco256_output = np.zeros((no_patches, patch_size,patch_size, 256))#256
#deco32_output = np.zeros((no_patches, patch_size,patch_size, 32))
#enco1024_output = np.zeros((no_patches, patch_size,patch_size, 1024))#1024

count=0
with torch.no_grad():
    for test in range(row_images2.shape[0]):

        X=torch.from_numpy((row_images2[test,...]).astype(np.float32))
        X=torch.unsqueeze(X,0)
        X_reconst, zval, muval, logvarval, enc_64,enc_256, enc_1024,dec_64,dec_32 = resnet_vae(X)
        #m=nn.Upsample(scale_factor=5,mode='bilinear')
        m=nn.Upsample(size=(64,64),mode='bilinear')
        #data_dec=m(dec_64)
        data_enc0=m(enc_64)
        data_enc=m(enc_256)
        #data_dec2=m(dec_32)
        #data_enc2=m(enc_1024)
        #data_dec = F.interpolate(data_dec, (64,64))
        #data_dec2 = F.interpolate(data_dec2, (64,64))
        #data_enc2 = F.interpolate(data_enc2[:,:,:,:], (64,64))
        #data_enc = F.interpolate(data_enc[:,:,:,:], (64,64))
        #data_enc0 = F.interpolate(data_enc0[:,:,:,:], (64,64))
        #decoder_out=torch.squeeze(data_dec[0,:,:,:])
        #decoder2_out=torch.squeeze(data_dec2[0,:,:,:])
        #encoder2_out=torch.squeeze(data_enc2[0,:,:,:])
        encoder_out=torch.squeeze(data_enc[0,:,:,:])
        encoder0_out=torch.squeeze(data_enc0[0,:,:,:])
        image_out=torch.squeeze(X[0,:,:,:])
        recons_out=torch.squeeze(X_reconst[0,:,:,:])
        #decoder_out1=decoder_out.permute(1,2,0)
        #decoder2_out1=decoder2_out.permute(1,2,0)
        encoder_out1=encoder_out.permute(1,2,0)
        #encoder2_out1=encoder2_out.permute(1,2,0)
        encoder0_out1=encoder0_out.permute(1,2,0)
        image_out1=image_out.permute(1,2,0)
        recons_out1=recons_out.permute(1,2,0)
        #decoder_out1.numpy()
        #decoder2_out1.numpy()
        encoder_out1.numpy()
        encoder0_out1.numpy()
        #encoder2_out1.numpy()
        image_out1.numpy()
        recons_out1.numpy()
        input_image[count,...]=image_out1
        #deco64_output[count,...]=decoder_out1
        #deco32_output[count,...]=decoder2_out1
        enco64_output[count,...]=encoder0_out1
        #enco256_output[count,...]=encoder_out1
        #enco1024_output[count,...]=encoder2_out1
        recons_output[count,...]=recons_out1
        count=count+1

image_final = np.zeros((num_rows, num_cols, num_filters))
#deco64_final = np.zeros((num_rows, num_cols, 64))
#deco32_final = np.zeros((num_rows, num_cols, 32))
enco64_final = np.zeros((num_rows, num_cols, 64))
#enco256_final = np.zeros((num_rows, num_cols, 256))#256
#enco1024_final = np.zeros((num_rows, num_cols, 1024))#1024
recons_final = np.zeros((num_rows, num_cols, num_filters))
patches=0
for i in range(int(x)):#num_rows/patch_size

    for j in range(int(y)):#num_cols/patch_size
        if patches==0:
        #image_final[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size, :]=input_image[patches,...]
            image_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=input_image[patches,...]
            enco64_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=enco64_output[patches,...]
            #enco256_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=enco256_output[patches,...]
            #enco1024_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=enco1024_output[patches,...]
            #deco32_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=deco32_output[patches,...]
            recons_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=recons_output[patches,...]
        elif i==0 and patches!=0:
        #image_final[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size, :]=input_image[patches,...]
            image_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6))+5:(((j+1)*int(patchsize/1.6))+int(24)),:]=input_image[patches,:,5:,:]
            enco64_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6))+5:(((j+1)*int(patchsize/1.6))+int(24)),:]=enco64_output[patches,:,5:,:]
            #enco256_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6))+10:(((j+1)*int(patchsize/1.6))+int(24)),:]=enco256_output[patches,:,10:,:]
            #enco1024_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6))+20:(((j+1)*int(patchsize/1.6))+int(24)),:]=enco1024_output[patches,:,20:,:]
            #deco32_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=deco32_output[patches,...]
            recons_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6))+5:(((j+1)*int(patchsize/1.6))+int(24)),:]=recons_output[patches,:,5:,:]
        elif j==0 and patches !=0:
        #image_final[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size, :]=input_image[patches,...]
            image_final[(i*int(patchsize/1.6))+5:(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=input_image[patches,5:,:,:]
            enco64_final[(i*int(patchsize/1.6))+5:(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=enco64_output[patches,5:,:,:]
            #enco256_final[(i*int(patchsize/1.6))+12:(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=enco256_output[patches,12:,:,:]
            #enco1024_final[(i*int(patchsize/1.6))+20:(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=enco1024_output[patches,20:,:,:]
            #deco32_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=deco32_output[patches,...]
            recons_final[(i*int(patchsize/1.6))+5:(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=recons_output[patches,5:,:,:]
        else:
        #image_final[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size, :]=input_image[patches,...]
            image_final[(i*int(patchsize/1.6))+5:(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6))+5:(((j+1)*int(patchsize/1.6))+int(24)),:]=input_image[patches,5:,5:,:]
            enco64_final[(i*int(patchsize/1.6))+5:(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6))+5:(((j+1)*int(patchsize/1.6))+int(24)),:]=enco64_output[patches,5:,5:,:]
            #enco256_final[(i*int(patchsize/1.6))+12:(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6))+10:(((j+1)*int(patchsize/1.6))+int(24)),:]=enco256_output[patches,12:,10:,:]
            #enco1024_final[(i*int(patchsize/1.6))+20:(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6))+20:(((j+1)*int(patchsize/1.6))+int(24)),:]=enco1024_output[patches,20:,20:,:]
            #deco32_final[(i*int(patchsize/1.6)):(((i+1)*int(patchsize/1.6))+int(24)),
									 #(j*int(patchsize/1.6)):(((j+1)*int(patchsize/1.6))+int(24)),:]=deco32_output[patches,...]
            recons_final[(i*int(patchsize/1.6))+5:(((i+1)*int(patchsize/1.6))+int(24)),
									 (j*int(patchsize/1.6))+5:(((j+1)*int(patchsize/1.6))+int(24)),:]=recons_output[patches,5:,5:,:]							 
		
        #plt.imshow(enco256_final[:,:,0])
        #plt.show()
        #plt.imshow(enco1024_final[:,:,0])
        #plt.show()
        #deco64_final[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size, :]=deco64_output[patches,...]
        #deco32_final[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size, :]=deco32_output[patches,...]
        #enco64_final[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size, :]=enco64_output[patches,...]
        #enco256_final[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size, :]=enco256_output[patches,...]
        #enco1024_final[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size, :]=enco1024_output[patches,...]
        #recons_final[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size, :]=recons_output[patches,...]
        patches=patches+1

#input_image=None
#deco64_output=None
#deco32_output=None
#enco256_output=None
#enco1024_output=None
#recons_output=None
print(image_final.shape)
image_final=image_final[p1_r:-p2_r,p1_c:-p2_c,:]
image_final1=image_final[:,:,:3]
#deco64_final=deco64_final[pad_row:H+pad_row,pad_col:W+pad_col,:]
#deco32_final=deco32_final[pad_row:H+pad_row,pad_col:W+pad_col,:]
enco64_final=enco64_final[p1_r:-p2_r,p1_c:-p2_c,:]
#enco256_final=enco256_final[pad_row:H+pad_row,pad_col:W+pad_col,:]
#enco1024_final=enco1024_final[pad_row:H+pad_row,pad_col:W+pad_col,:]
recons_final=recons_final[p1_r:-p2_r,p1_c:-p2_c,:]
recons_final1=recons_final[:,:,:3]
print("original image shape",recons_final.shape)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image_final1)
plt.title("Original image")
plt.subplot(1,2,2)
plt.imshow(recons_final1)
plt.title("Reconstructed image")
plt.show()

plt.imshow(enco64_final[:,:,5])
plt.show()
plt.imshow(enco64_final[:,:,0])
plt.show()
s=0
q=4#32
c=0
fig=plt.figure(figsize=(8, 8))
def fea_subplot(fea_array,s,q,c):
    for i in range(s,q): 
        rows=2#8
        columns=2#4	
        plt.subplot(rows, columns, c+1)
        plt.axis('off')
        plt.imshow(fea_array[:,:,i]) 
        c=c+1		
for g in range(4):#(int(features/32)):
    fea_subplot(enco64_final,s,q,c)
    plt.show()
    #fea_subplot(enco256_final,s,q,c)
    #plt.show()
    #fea_subplot(enco1024_final,s,q,c)
    #plt.show()
    s=s+4#32
    q=q+4#32

# store reconstructed & original subset image
driver = gdal.GetDriverByName('GTiff')
newRaster2 = driver.Create(test_results+'reconstructed_image'+'_subset23_513'+'.tif',list_image_extended.shape[1],
list_image_extended.shape[0], input_channels, gdal.GDT_Float32)#columns,rows
newRaster1 = driver.Create(test_results+'original_image'+'_subset23_513'+'.tif',list_image_extended.shape[1],
list_image_extended.shape[0], input_channels, gdal.GDT_Float32)
prj1 = proj # define the new raster dataset projection and geotransform
newRaster2.SetProjection(prj1)
newRaster1.SetProjection(prj1)
gt = geo
newTLCoord=gdal.ApplyGeoTransform(gt, py, px)
newRaster2.SetGeoTransform([newTLCoord[0], 3, 0, newTLCoord[1], 0, -3])
newRaster1.SetGeoTransform([newTLCoord[0], 3, 0, newTLCoord[1], 0, -3])
newBand1 = newRaster2.GetRasterBand(1) # get band 1, so we can fill it with data
newBand1.WriteArray(recons_final[:,:,0])
newBand2 = newRaster2.GetRasterBand(2) # get band 1, so we can fill it with data
newBand2.WriteArray(recons_final[:,:,1])
newBand3 = newRaster2.GetRasterBand(3) # get band 1, so we can fill it with data
newBand3.WriteArray(recons_final[:,:,2])
newBand4 = newRaster2.GetRasterBand(4) # get band 1, so we can fill it with data
newBand4.WriteArray(recons_final[:,:,3])
#newBand5 = newRaster2.GetRasterBand(5) # get band 1, so we can fill it with data
#newBand5.WriteArray(recons_final[:,:,4])
#newBand6 = newRaster2.GetRasterBand(6) # get band 1, so we can fill it with data
#newBand6.WriteArray(recons_final[:,:,5])
newBandimg1 = newRaster1.GetRasterBand(1) # get band 1, so we can fill it with data
newBandimg1.WriteArray(image_final[:,:,0])
newBandimg2 = newRaster1.GetRasterBand(2) # get band 1, so we can fill it with data
newBandimg2.WriteArray(image_final[:,:,1])
newBandimg3 = newRaster1.GetRasterBand(3) # get band 1, so we can fill it with data
newBandimg3.WriteArray(image_final[:,:,2])
newBandimg4 = newRaster1.GetRasterBand(4) # get band 1, so we can fill it with data
newBandimg4.WriteArray(image_final[:,:,3])
#newBandimg5 = newRaster1.GetRasterBand(5) # get band 1, so we can fill it with data
#newBandimg5.WriteArray(image_final[:,:,4])
#newBandimg6 = newRaster1.GetRasterBand(6) # get band 1, so we can fill it with data
#newBandimg6.WriteArray(image_final[:,:,5])
# store encoder 256feature image
# driver1 = gdal.GetDriverByName('GTiff')
# features=256
# newfeature = driver1.Create(test_results+'Encoder256_image'+'_subset1'+'.tif',list_image_extended.shape[1],
# list_image_extended.shape[0], features, gdal.GDT_Float32)
# newfeature.SetProjection(prj1)
# newTLCoord=gdal.ApplyGeoTransform(gt, py, px)
# newfeature.SetGeoTransform([newTLCoord[0], 10, 0, newTLCoord[1], 0, -10])
# for i in range(features):
    # globals()['new'+str(i)] = newfeature.GetRasterBand(i+1) # get band 1, so we can fill it with data
    # globals()['new'+str(i)].WriteArray(enco256_final[:,:,i])
# print("Encoder_256features")
# store encoder 64feature image
driver1 = gdal.GetDriverByName('GTiff')
features=64
newfeature = driver1.Create(test_results+'Encoder64_image'+'_subset23_513'+'.tif',list_image_extended.shape[1],
list_image_extended.shape[0], features, gdal.GDT_Float32)
newfeature.SetProjection(prj1)
newTLCoord=gdal.ApplyGeoTransform(gt, py, px)
newfeature.SetGeoTransform([newTLCoord[0], 3, 0, newTLCoord[1], 0, -3])
for i in range(features):
    globals()['new'+str(i)] = newfeature.GetRasterBand(i+1) # get band 1, so we can fill it with data
    globals()['new'+str(i)].WriteArray(enco64_final[:,:,i])
print("Encoder_64features")
# store encoder 1024feature image
# driver1 = gdal.GetDriverByName('GTiff')
# features=256
# newfeature = driver1.Create(test_results+'Encoder256_image'+'_subset1'+'.tif',list_image_extended.shape[1],
# list_image_extended.shape[0], features, gdal.GDT_Float32)
# newfeature.SetProjection(prj1)
# newTLCoord=gdal.ApplyGeoTransform(gt, py, px)
# newfeature.SetGeoTransform([newTLCoord[0], 10, 0, newTLCoord[1], 0, -10])
# for i in range(features):
    # globals()['new'+str(i)] = newfeature.GetRasterBand(i+1) # get band 1, so we can fill it with data
    # globals()['new'+str(i)].WriteArray(enco1024_final[:,:,i])
# print("Encoder_256features")
# b=None
# k=None
# covariance=None
# cov=None
# correlation=None
# cor_val=[]
# x1=[]
# features1=256
# corr_array3=np.zeros((features1,features1))
# for b in range(features1):
    # x=0
    # print(b)
    # for k in range(features1):
        # #print(k)
        # Layer1=enco256_final[:,:,b]
        # Layer2=enco256_final[:,:,k]
        # layer1_mean=np.mean(enco256_final[:,:,b])
        # layer2_mean=np.mean(enco256_final[:,:,k])	
        # layer1_std=np.std(enco256_final[:,:,b])
        # layer2_std=np.std(enco256_final[:,:,k])
        # #for i in range(int(Width-1)):
            # #for j in range(int(Height-1)):	
        # cov=(Layer1-layer1_mean)*(Layer2-layer2_mean)
                # #cov1=cov1+cov
        # covariance=np.mean(cov)#/((Height*Width)-1)
        # correlation=covariance/(layer1_std*layer2_std)
        # cor_val.append(correlation)
        # corr_array3[b,k]=correlation
        # x=x+1
        # x1.append(x)
        # #print("correlation_features:"+str(b)+','+str(k),correlation)
    # #print(len(x1))
    # #print(len(cor_val))
    # plt.scatter(x1,cor_val)
    # plt.title("Correlation-"+"Layer"+str(b)+","+"Layer"+str(k))
    # plt.savefig(test_results+"Layer"+str(b)+"_"+"Layer"+str(k)+'.jpg')
    # plt.clf()
    # #plt.show()
    # x1=[]
    # cor_val=[]
# #print(np.around(corr_array3,3))
# df = pd.DataFrame(corr_array3)
# column_title=["Layer_"+str(num) for num in range(features)]
# row_title=["Layer_"+str(num) for num in range(features)]
# np.savetxt(test_results+'encoder256'+'_subset1'+'.csv', np.around(corr_array3,3), delimiter=',', fmt='%s')
# np.save(test_results+'encoder256'+'_subset1'+'.npy',np.around(corr_array3,3))
# #Heatmap for correlation matrix
# s=0
# q=2
# def heat_map(cor_ar):
    # sns.heatmap(cor_ar)#annot=True
    # plt.title("Encoder correlation first 10 features")
    # plt.savefig(test_results+'correlation_heatmap'+'.jpg')

# #for g in range(int(features/32)):
# heat_map(np.around(corr_array3[:10,:10],3))
    # #s=s+32
    # #q=q+32
    
# #corr_plot(np.around(corr_array3,3),features,"Correlation_feature_image")



print("completed")
