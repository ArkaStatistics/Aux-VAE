import matplotlib.pylab as plt
import numpy as np
import h5py
import os
import sys
sys.path.insert(0, '/lcrc/project/cosmo_ai/aganguli/beta-vae')
import data_utils_galaxy_sim as data_utils

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import unittest
#import models_edited 
from torchsummary import summary
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset


dirIn = '/lcrc/project/cosmo_ai/nramachandra/Projects/SkyEmu/Data/output_tests_arka/'

# fileIn = dirIn + 'lhc_16384_5_training' + '.txt'  ## Training set N=256
fileIn = dirIn + 'lhc_64_5_test' + '.txt'   ## Test set N=64


# Read image params from the text file
image_params = data_utils.read_input_params(fileIn)
print('Parameter shape (nGal, param_size):', image_params.shape)
column_labels = ['flux', 'radius', 'g1', 'g2', 'psf']
image_params_reordered = image_params[:, [column_labels.index(label) for label in ['radius','g1', 'g2','flux', 'psf']]]
data_utils.plot_scatter_and_histogram_labeled(image_params, column_labels)

fileIn = dirIn + 'train_16384_5_training' + '.hdf5'  ## Training set N=256
# fileIn = dirIn + 'test_64_5_testing' + '.hdf5' ## Test sey N=64

# Read from the HDF5 file
images = data_utils.read_hdf5_outputs(fileIn)
image_params = data_utils.read_input_params('/lcrc/project/cosmo_ai/nramachandra/Projects/SkyEmu/Data/lhc_16384_5_training.txt')
image_params_reordered = image_params[:, [column_labels.index(label) for label in ['radius','g1', 'g2', 'flux', 'psf']]]
image_params=image_params_reordered
print('Galaxy image shape (nGal, pixel_size, pixel_size): ', np.array(images).shape)

########################################################################################################################################
########################################################################################################################################

# Extract the 'image' data and reshape it
#image_list = [np.array(item["image"]).reshape(128, 128) for item in data]
image_list = images #[np.array(item["image"]).reshape(64,64) for item in data]
# scaling ta data

# Extract the 'image' data and reshape it

# Convert to a NumPy array
image_array = np.array(image_list)

# Scale the pixel values between [0, 1] adaptively for each image
min_values = np.min(image_array)#, axis=(1, 2), keepdims=True)
max_values = np.max(image_array)#, axis=(1, 2), keepdims=True)

# Avoid division by zero
#max_values[max_values == min_values] = 1.0

image_array_normalized = (image_array - min_values) / (max_values - min_values)
image_list=image_array_normalized

#scaling the u-array

# Compute the minimum and maximum values for each column
min_values = np.min(image_params, axis=0)
max_values = np.max(image_params, axis=0)

# Perform Min-Max scaling for each column
scaled_image_params = (image_params - min_values) / (max_values - min_values)


u_list_all=scaled_image_params.tolist()

# Initialize empty lists to store the combined data
enumerated_dataset = []
u_list = []
x_list = []

# Enumerate through the lists and combine them
for j, (u, image) in enumerate(zip( u_list_all, image_list)):
    # Create a dictionary to represent the enumerated dataset
    data_entry = {
        'j': j,
        'u': u,  # Combine the first three lists into 'u'
        'x': image  # Store the image from reshaped_images as 'x'   
    }
    
    # Append the data_entry to the enumerated dataset
    enumerated_dataset.append(data_entry)
    u_list.append(u)
    x_list.append(image)


from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, u_list, x_list, device="cuda"):
        self.u_list = torch.tensor(u_list, dtype=torch.float32, device=device)
        self.x_list = torch.tensor(x_list, dtype=torch.float32, device=device)
    
    def __len__(self):
        return len(self.u_list)
    
    def __getitem__(self, idx):
        sample = {
            'u': self.u_list[idx],
            'x': self.x_list[idx]
        }
        return sample

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create an instance of CustomDataset with CUDA-enabled data
custom_dataset = CustomDataset(u_list, x_list, device=device)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, u_list, x_list, batch_size=64):
        super().__init__()
        self.x_list = x_list
        self.u_list = u_list
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        dataset = CustomDataset(self.u_list, self.x_list)
        n = len(dataset)
        n_train = int(0.8 * n)  # 80% train, 20% validation
        n_val = n - n_train
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        # If you have a test set, you can define it here
        pass

# Create an instance of MyDataModule
data_module = MyDataModule(u_list, x_list)
data_module.setup()
# Call train_dataloader on the data_module instance
train_loader = DataLoader(data_module.train_dataset, batch_size=64, shuffle=True) #data_module.train_dataloader(batch_size=256)
test_loader= DataLoader(data_module.train_dataset, batch_size=64, shuffle=True) #data_module.val_dataloader(batch_size=256)


########################################################################################################################################
########################################################################################################################################

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import multiprocessing
import time
import preprocess as prep
import models_edited as models
import utils
from torchvision.utils import save_image


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0
    recon_losses = 0
    kl_diverges = 0
    
    for batch_idx, data in enumerate(train_loader):
        x = data['x'].unsqueeze(1).to(device)
        u = data['u'].to(device)
        batch_size=x.shape[0]
        #x= x.view(batch_size, -1)
        optimizer.zero_grad()
        output, z, mu, logvar = model(x)
        #print('mu:', mu.shape)
        #print('logvar:', logvar.shape)
        loss, recon_loss, kl_diverge = model.loss_log_mse_reduced_cor(output, x, z, u, mu, logvar)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        recon_losses += recon_loss.item()
        kl_diverges += kl_diverge.item()
        
        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()), epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    
    train_loss /= len(train_loader)
    recon_losses /= len(train_loader)
    kl_diverges /= len(train_loader)
    print('Train set Average loss:', train_loss, 'recon_loss', recon_losses, 'kl_diverge', kl_diverges)
    return train_loss, recon_losses, kl_diverges


def test(model, device, test_loader, return_images=0, log_interval=None):
    model.eval()
    test_loss = 0
    
    # two np arrays of images
    original_images = []
    rect_images = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            x = data['x'].unsqueeze(1).to(device)
            u = data['u'].to(device)
            batch_size=x.shape[0]
            #x = x.view(batch_size, -1)
            output, z, mu, logvar = model(x)
            loss, recon_loss, kl_diverge = model.loss_log_mse_reduced_cor(output, x, z, u, mu, logvar)
            test_loss += loss.item()
            
            if return_images > 0 and len(original_images) < return_images:
                original_images.append(x[0].cpu())
                rect_images.append(output[0].cpu())
            
            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))
    
    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)
    
    if return_images > 0:
        return test_loss, original_images, rect_images
    
    return test_loss


# parameters
INPUT_SIZE=33
BATCH_SIZE = 100
TEST_BATCH_SIZE = 10
EPOCHS = 1000

LATENT_SIZE = 10
LEARNING_RATE = 1e-3

USE_CUDA = True
PRINT_INTERVAL = 100
LOG_PATH = './logs/log.pkl'
MODEL_PATH = './checkpoints/'
COMPARE_PATH = './comparisons/'

use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())

print('latent size:', LATENT_SIZE)
# model = models.BetaVAE(latent_size=LATENT_SIZE).to(device)
model = models.BetaVAE_CNN_aux(input_channels=1,latent_size=LATENT_SIZE, u_idx=[0,1,2,3,4], beta=20, reg_inter=.40, reg_intra=.50).to(device) #(10,10,80)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if __name__ == "__main__":
    
    start_epoch = model.load_last_model(MODEL_PATH) + 1
    train_losses, test_losses = utils.read_log(LOG_PATH, ([], []))
    train_recon_losses=[]
    train_kl_diverges=[]
    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_recon_loss, train_kl_diverge = train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
        #print("training_loss computation done")
        test_loss, original_images, rect_images = test(model, device, test_loader, return_images=5)
        #print("testing_loss computation done")
        
        if epoch % 100 == 0:  # Save the plots only for epochs divisible by 20
            # Create a single plot with two rows for original and reconstructed images
            fig, axs = plt.subplots(2, len(original_images), figsize=(15, 8))
            
            for i in range(len(original_images)):
                im0 = axs[0, i].imshow(original_images[i].view(INPUT_SIZE, INPUT_SIZE), cmap='viridis')
                axs[0, i].set_title("Original Image")
                axs[0, i].axis('off')
                fig.colorbar(im0, ax=axs[0, i], orientation='horizontal')
                
                im1 = axs[1, i].imshow(rect_images[i].view(INPUT_SIZE, INPUT_SIZE), cmap='viridis')
                axs[1, i].set_title("Reconstructed Image")
                axs[1, i].axis('off')
                fig.colorbar(im1, ax=axs[1, i], orientation='horizontal')
            
            plt.tight_layout()
            plt.savefig(COMPARE_PATH + f'{epoch}_image_plots.png', bbox_inches='tight')
            plt.close()
        
        train_losses.append((epoch, train_loss))
        train_recon_losses.append(train_recon_loss)
        train_kl_diverges.append(train_kl_diverge)
        test_losses.append((epoch, test_loss))
        utils.write_log(LOG_PATH, (train_losses, test_losses))
        
        model.save_model(MODEL_PATH + '%03d.pt' % epoch)


# Create an array of epochs
epochs = np.arange(EPOCHS) + 1

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot the MSE loss
axes[0].plot(epochs[50:], train_recon_losses[50:])
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('MSE Loss vs. Epoch')
axes[0].grid(True)

# Plot the KL divergence
axes[1].plot(epochs[50:], train_kl_diverges[50:])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('KL Divergence')
axes[1].set_title('KL Divergence vs. Epoch')
axes[1].grid(True)

# Adjust the spacing between subplots
plt.tight_layout()
#plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/traing_recon_VS_kl.png')
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/traing_recon_VS_kl_reduced_flux_psf_AuxVAE.png')



model.eval()

for batch_idx, data in enumerate(DataLoader(data_module.train_dataset, batch_size=10000, shuffle=True)):
    x = data['x'].unsqueeze(1).to(device)
    u = data['u'].to(device) #shape=(10000,2)
    batch_size=x.shape[0]
    #x = x.view(batch_size, -1)
    output,z, mu, logvar = model(x) #shape of mu= (10000,4)
    print(mu.shape)
    if batch_idx==0:
        break


cor_reduced=torch.abs(torch.corrcoef(torch.cat((z, u[:,[0,1,2]]),axis=1).T))[:(z.shape[1]),(z.shape[1]):]
m_reduced= torch.sum(torch.max(cor_reduced, axis=0).values/torch.sum(cor_reduced, axis=0))/3
# Extract the first two dimensions of mu and u
mu_10d = z[:,:10].cpu().detach().numpy()
#sigma_2d=logvar[:,:5].exp().cpu().detach().numpy()
u_5d = u.cpu().detach().numpy()

mu_10d.shape
u_5d.shape


# Define the dimensions for the subplots
num_rows = 5
num_cols = 10

# Create a 5x5 matrix of scatter plots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))

# Labels for dimensions
u_labels = ['radius', 'g1', 'g2', 'flux', 'psf']
mu_labels = ['mu_dim1', 'mu_dim2', 'mu_dim3', 'mu_dim4', 'mu_dim5', 'mu_dim6', 'mu_dim7', 'mu_dim8', 'mu_dim9', 'mu_dim10']

# Iterate through each pair of dimensions
for i in range(num_rows):
    for j in range(num_cols):
        # Scatter plot for cross-correlations (off-diagonal)
        axes[i, j].scatter(u_5d[:, i], mu_10d[:, j], alpha=0.5, s=10)
        axes[i, j].set_xlabel(u_labels[i])
        axes[i, j].set_ylabel(mu_labels[j])

fig.suptitle('disentanglement metric=' + str(m_reduced.detach().cpu().numpy()), x=0.5, y=0.98, ha='center', fontsize=16)
plt.tight_layout()
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/Z-vs-U_reduced_flux_psf_cor_CNN.png')
plt.clf()



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Transform mu using t-SNE
tsne = TSNE(n_components=2, random_state=40)
mu_2d = tsne.fit_transform(z[:,:3].cpu().detach().numpy())#mu_10d)
row_names = [ 'radius', 'g1', 'g2', 'flux', 'psf']
# Plot t-SNE with respect to each column in u_5d side by side
plt.figure(figsize=(16, 6))  # Adjust the figure size as needed

# Loop through each column in u_5d
for i in range(5):
    plt.subplot(1, 5, i+1)  # Create subplots in a single row
    plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=u_5d[:, i], cmap='viridis', alpha=0.5)
    plt.colorbar(label=row_names[i], orientation='horizontal' )  # Add colorbar with label
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(row_names[i])

plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/t-sne_full_reduced_cor_CNN.png')


#histograms of latent features

# Create histograms for mu_10d
plt.figure(figsize=(16, 8))  # Adjust figure size as needed

# Plot histograms for mu_10d
for i in range(mu_10d.shape[1]):
    plt.subplot(2, 5, i + 1)  # Create subplots in a 2x5 grid
    plt.hist(mu_10d[:, i], bins=20, color='blue', alpha=0.7)
    plt.xlabel(f'Dimension {i+1} of mu_10d')
    plt.ylabel('Frequency')

# Plot histograms for the first three dimensions of u_5d
for i in range(3):
    plt.subplot(2, 5, i+1)  # Create subplots in the second row
    plt.hist(u_5d[:, i], bins=20, color='orange', alpha=0.7)
    plt.xlabel(f'Dimension {i+1} of u_5d')
    plt.ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/histograms_u-vs-latent_mean_poly.png')
plt.clf()
# Iterate through each pair of dimensions
for i in range(num_rows):
    for j in range(num_cols):
        # Scatter plot for cross-correlations (off-diagonal)
        axes[i, j].scatter(u_5d[:, i], u_5d[:, j], alpha=0.5, s=10)
        axes[i, j].set_xlabel(u_labels[i])
        axes[i, j].set_ylabel(u_labels[j])

plt.tight_layout()
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/U-vs-U.png')
#plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/Z-vs-U_reduced.png')


z=[0.995822,  0.6284,  0.4133, -0.6370,  1.5005, -0.996042,  -1.1068,  0.3171, 0.7993,  0.6270]
image=model.decoder(torch.tensor(z).to(device))

# Create a figure and plot the image
plt.figure(figsize=(6, 6))
plt.imshow(image.reshape(33,33).detach().cpu().numpy(), cmap= 'viridis')  # Assuming z represents grayscale values
plt.title('Image Plot of z')
plt.axis('off')  # Turn off axis labels
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/reconstructed_x.png')
########################################################################################
#testing

model.eval()


x_numpy = x.cpu().detach().numpy().reshape(10000, 33, 33)
output_numpy = output.cpu().detach().numpy().reshape(10000, 33, 33)


# Create subplots with 2 rows and 7 columns
fig, axs = plt.subplots(2, 7, figsize=(15, 8))

# Plot x in the first row and add variance to the caption
for i in range(7):
    variance_x = np.var(x_numpy[i]) * 1000
    im0 = axs[0, i].imshow(x_numpy[i], cmap='viridis')
    axs[0, i].set_title(f'x[{i}] (Var: {variance_x:.2f})')
    axs[0, i].axis('off')
    fig.colorbar(im0, ax=axs[0, i], orientation='horizontal')

# Plot output in the second row and add variance to the caption
for i in range(7):
    variance_output = np.var(output_numpy[i]) * 1000
    im1 = axs[1, i].imshow(output_numpy[i], cmap='viridis')
    axs[1, i].set_title(f'output[{i}] (Var: {variance_output:.2f})')
    axs[1, i].axis('off')
    fig.colorbar(im1, ax=axs[1, i], orientation='horizontal')

plt.tight_layout()

# Save the figure
#plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/image_plots_testing2.png')
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/image_plots_testing2_CNN.png')
plt.clf()
# Create subplots with 2 rows and 7 columns
fig, axs = plt.subplots(2, 7, figsize=(15, 8))

# Plot x in the first row and add variance to the caption
for i in range(7):
    variance_x = np.var(x_numpy[i])*1000
    axs[0, i].imshow(x_numpy[i], cmap='viridis')
    axs[0, i].set_title(f'x[{i}] (Var: {variance_x:.2f})')
    axs[0, i].axis('off')

# Plot output in the second row and add variance to the caption
for i in range(7):
    variance_output = np.var(output_numpy[i])*1000
    axs[1, i].imshow(output_numpy[i], cmap='viridis')
    axs[1, i].set_title(f'output[{i}] (Var: {variance_output:.2f})')
    axs[1, i].axis('off')

# Add a single colorbar for all the subplots
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])  # Adjust the position and size as needed
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
plt.tight_layout()
# Save the figure
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/image_plots_testing2.png')
plt.clf()

#plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/image_plots_testing1_reduced.png')

########################################################################################################################################################################################################################################################################

# Create a color map with a unique color for each row
colors = plt.cm.viridis(np.linspace(0, 1, mu.shape[0]))

# Plot each row of the matrix as a line plot with corresponding color
plt.figure(figsize=(10, 6))
for i in range(mu.shape[0]):
    plt.plot(mu[i].detach().cpu().numpy(), color=colors[i])

# Add color bar
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=mu.shape[0]-1))
sm.set_array([])
plt.colorbar(sm, label='Row Index')

plt.xlabel('Column Index')
plt.ylabel('Value')

# Plot each row of the matrix as a line plot
plt.figure(figsize=(10, 6))
for i in range(25):
    plt.plot(mu[i].detach().cpu().numpy(), label=f'Row {i+1}')

plt.xlabel('Column Index')
plt.ylabel('Value')
plt.title('Line Plot of latent_factors')
plt.legend()
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/visualizing_latent_factors_poly.png')
plt.clf()


image_id=6
first_row = x[image_id].detach().cpu().numpy().reshape(-1,1)
image = first_row.reshape(33, 33)

# Plot the image
plt.imshow(image, cmap='viridis')  # Choose a colormap as per your preference
plt.colorbar()  # Add color bar for reference
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/original_sample.png')
plt.clf()

u_original=u[image_id]
mu_est=mu[image_id]
mu_zero_start= torch.tensor([mu_est[0],mu_est[1],mu_est[2],0,0,0,mu_est[6],mu_est[7],mu_est[8],mu_est[9]])
mu_zero_end= torch.tensor([mu_est[0],mu_est[1],mu_est[2],mu_est[3],mu_est[4],mu_est[5],0,0,0,0])
mu_low=torch.tensor([mu_est[0],mu_est[1],mu_est[2],-mu_est[3],-mu_est[4],-mu_est[5],mu_est[6],mu_est[7],mu_est[8],mu_est[9]])
mu_high=torch.tensor([mu_est[0],mu_est[1],mu_est[2],mu_est[3],mu_est[4],mu_est[5],-mu_est[6],-mu_est[7],-mu_est[8],-mu_est[9]])

generated_images_org = model.decode(mu_est.to(device)).cpu().detach().numpy()
generated_images_zero_start = model.decode(mu_zero_start.to(device)).cpu().detach().numpy()
generated_images_zero_end = model.decode(mu_zero_end.to(device)).cpu().detach().numpy()
generated_images_low = model.decode(mu_low.to(device)).cpu().detach().numpy()
generated_images_high = model.decode(mu_high.to(device)).cpu().detach().numpy()


# Create a subplot with 1 row and 5 columns
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

# Plot each generated image
axes[0].imshow(generated_images_org.reshape(33, 33), cmap='viridis')
axes[0].set_title('Estimated')
axes[1].imshow(generated_images_zero_start.reshape(33, 33), cmap='viridis')
axes[1].set_title('Zero Start')
axes[2].imshow(generated_images_zero_end.reshape(33, 33), cmap='viridis')
axes[2].set_title('Zero End')
axes[3].imshow(generated_images_low.reshape(33, 33), cmap='viridis')
axes[3].set_title('Low')
axes[4].imshow(generated_images_high.reshape(33, 33), cmap='viridis')
axes[4].set_title('High')

# Hide the axes
for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/DSprites/generated_images_side_by_side1_reduced_g1_g2.png')
plt.clf()
########################################################################################################################################################################################################################################################################
# Reshape the first row to a 33x33 matrix

# Get the first row of the tensor
image_id=1
first_row = x[image_id].detach().cpu().numpy().reshape(-1,1)
image = first_row.reshape(33, 33)

# Plot the image
plt.imshow(image, cmap='viridis')  # Choose a colormap as per your preference
plt.colorbar()  # Add color bar for reference
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/original_sample.png')
plt.clf()
# Define the fixed values for the last 5 latent variables

# Define the fixed values for the last 5 latent variables
fixed_latent_vars = mu[image_id].clone().repeat(5, 1)  # Repeat the same values 5 times

# Move fixed_latent_vars to the same device as u
device = mu.device
fixed_latent_vars = fixed_latent_vars.to(device)

# Add one last column with a single random value repeated 5 times
#random_value = torch.rand(5).to(device)  # Generate a single random value and move it to the same device as fixed_latent_vars
#last_column = random_value.repeat(5, 1)  # Repeat the random value 5 times in the last column
#fixed_latent_vars = torch.cat((fixed_latent_vars, last_column), dim=1)

# Create a figure with a 5x10 grid of subplots
fig, axs = plt.subplots(10, 5, figsize=(25, 25))

# Define row names
#row_names = [ 'radius', 'g1', 'g2', 'flux', 'psf']
row_names = [ 'radius', 'g1', 'g2', 'flux', 'psf']

# Create a common colorbar
#cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position and size as needed

# Loop through the first 5 latent variables
for i in range(10):
    # Generate a range of values for the i-th latent variable from 0.1 to 1
    latent_var_values = torch.linspace(-1.75, 3, 5) #torch.linspace(0.1, 1, 5)
    
    # Repeat the fixed latent variables for this dimension
    latent_vars = fixed_latent_vars.clone()
    latent_vars[:, i] = latent_var_values
    
    # Generate images using the modified latent variables (latent_vars)
    #generated_images = model.decode(latent_vars.to(device)).cpu().detach().numpy()
    generated_images = model.decode(latent_vars.to(device)).cpu().detach().numpy()
    # Loop through the 10 generated images for this latent variable
    for j in range(5):
        ax = axs[i, j]
        im = ax.imshow(generated_images[j].reshape(33, 33), cmap='viridis')
        #im = ax.plot(generated_images[j].reshape(33, 33))
        #if j==0:
            #ax.set_xlabel(row_names[i])
            #ax.set_ylabel(row_names[i])
        ax.axis('off')

# Set column names as 0.1, 0.2, ..., 0.9, 1
for j in range(5):
    axs[0, j].set_title(f'{latent_var_values[j]}') #(f'{0.1 + (2*j) * 0.1:.1f}')

for j in range(5):
    axs[j, 0].set_ylabel(f'{row_names[j]}') #(f'{0.1 + (2*j) * 0.1:.1f}')
 

# Add a single colorbar for all subplots
#fig.colorbar(im, cax=cbar_ax)

# Adjust spacing between subplots
plt.tight_layout()

# Save the figure
#plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/reconstructed_from_latent.png')
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/reconstructed_from_latent_reduced_flux_psf_CNN.png')
plt.clf()

##################################################

# Create a figure with a 5x10 grid of subplots
fig, axs = plt.subplots(5, 5, figsize=(5, 3))

# Define row names
#row_names = [ 'radius', 'g1', 'g2', 'flux', 'psf']
row_names = [ 'radius', 'g1', 'g2', 'flux', 'psf']

# Create a common colorbar
#cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position and size as needed

# Loop through the first 5 latent variables
for i in range(5):
    # Generate a range of values for the i-th latent variable from 0.1 to 1
    latent_var_values = torch.linspace(-2, 2, 5) #torch.linspace(0.1, 1, 5)
    
    # Repeat the fixed latent variables for this dimension
    latent_vars = fixed_latent_vars.clone()
    latent_vars[:, 5+i] = latent_var_values
    
    # Generate images using the modified latent variables (latent_vars)
    #generated_images = model.decode(latent_vars.to(device)).cpu().detach().numpy()
    generated_images = model.decode(latent_vars.to(device)).cpu().detach().numpy()
    # Loop through the 10 generated images for this latent variable
    for j in range(5):
        ax = axs[i, j]
        im = ax.imshow(generated_images[j].reshape(33, 33), cmap='viridis')
        #im = ax.plot(generated_images[j].reshape(33, 33))
        #if j==0:
            #ax.set_xlabel(row_names[i])
            #ax.set_ylabel(row_names[i])
        ax.axis('off')

# Set column names as 0.1, 0.2, ..., 0.9, 1
for j in range(5):
    axs[0, j].set_title(f'{latent_var_values[j]}') #(f'{0.1 + (2*j) * 0.1:.1f}')

for j in range(5):
    axs[j, 0].set_ylabel(f'{row_names[j]}') #(f'{0.1 + (2*j) * 0.1:.1f}')
 

# Add a single colorbar for all subplots
#fig.colorbar(im, cax=cbar_ax)

# Adjust spacing between subplots
plt.tight_layout()

# Save the figure
#plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/reconstructed_from_latent.png')
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/reconstructed_from_latent_z-recon_full_mean_poly.png')
plt.clf()

first_row = x[0].detach().cpu().numpy().reshape(-1, 1)
image = first_row.reshape(33, 33)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))

# Plot the original image on the left subplot
im1 = ax1.imshow(image, cmap='viridis')  # Choose a colormap as per your preference
ax1.set_title('Original Sample')
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)  # Add color bar for reference

# Define the fixed values for the last 5 latent variables
fixed_latent_vars = u[0].clone().repeat(5, 1)  # Repeat the same values 5 times

# Move fixed_latent_vars to the same device as u
device = u.device
fixed_latent_vars = fixed_latent_vars.to(device)

# Add one last column with a single random value repeated 5 times
random_value = torch.rand(1).to(device)  # Generate a single random value and move it to the same device as fixed_latent_vars
last_column = random_value.repeat(5, 1)  # Repeat the random value 5 times in the last column
fixed_latent_vars = torch.cat((fixed_latent_vars, last_column), dim=1)

# Create a 5x5 array of subplots for reconstructed images
ax2 = fig.subplots(5, 5, gridspec_kw={'left': 0.55})

# Loop through the first 5 latent variables
for i in range(5):
    # Generate a range of values for the i-th latent variable from 0.1 to 1
    latent_var_values = torch.linspace(0.1, 1, 5)
    
    # Repeat the fixed latent variables for this dimension
    latent_vars = fixed_latent_vars.clone()
    latent_vars[:, i] = latent_var_values
    
    # Generate images using the modified latent variables (latent_vars)
    generated_images = model.decode(latent_vars.to(device)).cpu().detach().numpy()
    
    # Loop through the 10 generated images for this latent variable
    for j in range(5):
        im2 = ax2[j, i].imshow(generated_images[j].reshape(33, 33), cmap='viridis')
        ax2[j, i].axis('off')

# Set column names as 0.1, 0.2, ..., 0.9, 1
for i in range(5):
    ax2[0, i].set_title(f'{0.1 + (2 * i) * 0.1:.1f}')

# Set row names
row_names = ['flux', 'radius', 'g1', 'g2', 'psf']
for i, row_name in enumerate(row_names):
    ax2[i, 0].set_ylabel(row_name)

# Add a single colorbar for all subplots
cbar2 = fig.colorbar(im2, ax=ax2.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.04)

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.3)

# Save the figure
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/reconstructed_from_latent1_reduced.png')
plt.clf()
########################################################################################################################################################################################################################################################################

for batch_idx, data in enumerate(DataLoader(data_module.train_dataset, batch_size=10000, shuffle=True)):
    x = data['x'].to(device)
    u = data['u'].to(device) #shape=(10000,2)
    batch_size=x.shape[0]
    x = x.view(batch_size, -1)
    output, mu, logvar = model(x) #shape of mu= (10000,4)
    print(mu.shape)
    if batch_idx==0:
        break



cor_reduced_1_1=torch.abs(torch.corrcoef(torch.cat((mu, u),axis=1).T))[:(mu.shape[1]),(mu.shape[1]):]
cor_reduced_1_2=torch.abs(torch.corrcoef(torch.cat((mu, u**2),axis=1).T))[:(mu.shape[1]),(mu.shape[1]):]
cor_reduced_1_3=torch.abs(torch.corrcoef(torch.cat((mu, u**3),axis=1).T))[:(mu.shape[1]),(mu.shape[1]):]
cor_reduced_2_1=torch.abs(torch.corrcoef(torch.cat((mu**2, u),axis=1).T))[:(mu.shape[1]),(mu.shape[1]):]
cor_reduced_2_2=torch.abs(torch.corrcoef(torch.cat((mu**2, u**2),axis=1).T))[:(mu.shape[1]),(mu.shape[1]):]
cor_reduced_2_3=torch.abs(torch.corrcoef(torch.cat((mu**2, u**3),axis=1).T))[:(mu.shape[1]),(mu.shape[1]):]
cor_reduced_3_1=torch.abs(torch.corrcoef(torch.cat((mu**3, u),axis=1).T))[:(mu.shape[1]),(mu.shape[1]):]
cor_reduced_3_2=torch.abs(torch.corrcoef(torch.cat((mu**3, u**2),axis=1).T))[:(mu.shape[1]),(mu.shape[1]):]
cor_reduced_3_3=torch.abs(torch.corrcoef(torch.cat((mu**3, u**3),axis=1).T))[:(mu.shape[1]),(mu.shape[1]):]
cor_reduced = (cor_reduced_1_1+cor_reduced_1_2+cor_reduced_1_3+cor_reduced_2_1+cor_reduced_2_2+cor_reduced_2_3+cor_reduced_3_1+cor_reduced_3_2+cor_reduced_3_3)/9
# Labels for dimensions
u_labels = ['radius', 'flux', 'psf', 'g1', 'g2']
mu_labels = ['$Z_1$', '$Z_2$', '$Z_3$', '$Z_4$', '$Z_5$', '$Z_6$', '$Z_7$', '$Z_8$', '$Z_9$', '$Z_{10}$']
import seaborn as sns

# Create heatmap
plt.figure(figsize=(5, 3))
#sns.heatmap(cor_reduced.T.detach().cpu().numpy(), annot=False, cmap='coolwarm', fmt=".2f", xticklabels=mu_labels, yticklabels=u_labels)
sns.heatmap(cor_reduced.T.detach().cpu().numpy(), annot=False, cmap='YlOrRd', fmt=".2f", xticklabels=mu_labels, yticklabels=u_labels, vmin=0, vmax=1)
#plt.title('Correlation Heatmap')
plt.xlabel('Latent factors')
plt.ylabel('Ground-truth factors')
plt.savefig('/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/heatmap_cor_Z_vs_U_reduced_g1_g2.png')
plt.clf()


##SSIM measure
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(original_images, reconstructed_images):
    ssim_scores = []
    for original_image, reconstructed_image in zip(original_images, reconstructed_images):
        score, _ = ssim(original_image, reconstructed_image, full=True, data_range=1.0)
        ssim_scores.append(score)
    return ssim_scores

# Example usage:
# original_images: numpy array of original images with shape (num_samples, height, width)
# reconstructed_images: numpy array of reconstructed images with shape (num_samples, height, width)

for batch_idx, data in enumerate(DataLoader(data_module.train_dataset, batch_size=1000, shuffle=True)):
    x = data['x'].to(device)
    u = data['u'].to(device) #shape=(10000,2)
    batch_size=x.shape[0]
    x = x.view(batch_size, -1)
    output, mu, logvar = model(x) #shape of mu= (10000,4)
    print(mu.shape)
    if batch_idx==0:
        break


# Generate some example data (replace this with your actual data)
x_numpy = x.cpu().detach().numpy().reshape(1000, 33, 33)
output_numpy = output.cpu().detach().numpy().reshape(1000, 33, 33)

# Calculate SSIM scores
ssim_scores = calculate_ssim(x_numpy, output_numpy)
print("SSIM scores:", ssim_scores)
print("Mean SSIM score:", np.mean(ssim_scores))
# Specify the file path
file_path = "/lcrc/project/cosmo_ai/aganguli/beta-vae/plots/ssim_scores.csv"

# Save the SSIM scores to a CSV file with append mode
with open(file_path, 'a') as f:
    np.savetxt(f, [ssim_scores], delimiter=",", fmt="%.6f")

torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
cor_exp= torch.corrcoef(torch.cat((mu, u[:,u_idx]),axis=1).T)
cor_theory=torch.cov(torch.cat((mu, u[:,u_idx]),axis=1).T)
z_sd_theory=torch.mean(torch.sqrt(torch.exp(logvar)), axis=0)
u_sd= torch.sqrt(torch.var(u[:,u_idx], axis=0))
all_sd=torch.cat((z_sd_theory, u_sd))
all_sd_exp= torch.sqrt(torch.var(torch.cat((z, u[:,u_idx]),axis=1), axis=0))
for i in range(cor_theory.shape[0]):
    for j in range(cor_theory.shape[1]):
        cor_theory[i,j]=cor_theory[i,j]/(all_sd_exp[i]*all_sd_exp[j])
    
