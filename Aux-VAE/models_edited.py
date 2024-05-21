import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mutual_info_score
import utils

# CelebA (VAE)
# Input 64x64x3.
# Adam 1e-4
# Encoder Conv 32x4x4 (stride 2), 32x4x4 (stride 2), 64x4x4 (stride 2),
# 64x4x4 (stride 2), FC 256. ReLU activation.
# Latents 32
# Decoder Deconv reverse of encoder. ReLU activation. Gaussian.
class BetaVAE64(nn.Module):
    
    def __init__(self, latent_size=32, beta=1):
        super(BetaVAE64, self).__init__()
        
        self.latent_size = latent_size
        self.beta = beta
        
        # encoder
        self.encoder = nn.Sequential(
            self._conv(1, 32),
            self._conv(32, 32),
            self._conv(32, 64),
            self._conv(64, 64),
        )
        self.fc_mu = nn.Linear(256, latent_size)
        self.fc_var = nn.Linear(256, latent_size)
        
        # decoder
        self.decoder = nn.Sequential(
            self._deconv(64, 64),
            self._deconv(64, 32),
            self._deconv(32, 32, 1),
            #self._deconv(32, 1),
            nn.ConvTranspose2d(32, 1,kernel_size=4, stride=2, output_padding=0),
            nn.Sigmoid()
        )
        self.fc_z = nn.Linear(latent_size, 256)
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256)
        return self.fc_mu(x), self.fc_var(x)
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 64, 2, 2)
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar
    
    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    # out_padding is used to ensure output size matches EXACTLY of conv2d;
    # it does not actually add zero-padding to output :)
    def _deconv(self, in_channels, out_channels, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, output_padding=out_padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        #print('recon_loss:', recon_loss)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #print('KL_div:', kl_diverge)
        return ((recon_loss + self.beta * kl_diverge) / x.shape[0]), recon_loss/ x.shape[0], kl_diverge/ x.shape[0]  # divide total loss by batch size
    
    def loss_log_mse_reduced_cor(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        u0 = torch.cat((u[:, :3], torch.zeros(u.shape[0], (mu.shape[1] - 3)).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        # Calculate maximum absolute correlation between last 5 dimensions of mu and u
        #max_corr = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        
        max_corr=torch.zeros(9)
        max_corr[0] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        max_corr[1] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]**2),axis=1).T)[:7,7:]),axis=1))/7
        max_corr[2] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]**3),axis=1).T)[:7,7:]),axis=1))/7
        max_corr[3] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:]**2, u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        max_corr[4] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:]**2, u[:,:3]**2),axis=1).T)[:7,7:]),axis=1))/7
        max_corr[5] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:]**2, u[:,:3]**3),axis=1).T)[:7,7:]),axis=1))/7
        max_corr[6] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:]**3, u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        max_corr[7] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:]**3, u[:,:3]**2),axis=1).T)[:7,7:]),axis=1))/7
        max_corr[8] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:]**3, u[:,:3]**3),axis=1).T)[:7,7:]),axis=1))/7
        
        mu_aux_corr=torch.corrcoef(torch.cat((mu[:, :3], u[:,:3]),axis=1).T)[:3,3:]
        aux_corr=torch.zeros(2)
        aux_corr[0] = torch.sum(1-torch.diag(mu_aux_corr))
        aux_corr[1] = torch.sum(torch.abs(mu_aux_corr))-torch.sum(torch.diag(torch.abs(mu_aux_corr)))

        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / x.size(0) + self.beta*torch.sum(max_corr)*self.reg_inter + self.beta*torch.sum(aux_corr)*self.reg_intra#3
        
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/x.size(0), kl_diverge / x.size(0)
    
    
    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)
    
    def load_model(self, file_path):
        utils.restore(self, file_path)
    
    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)


class BetaVAE_CNN_aux(nn.Module):
    def __init__(self, input_channels=3, latent_size=32, beta=1, reg_intra=1, reg_inter=1, u_idx=[0, 1, 2]):
        super(BetaVAE_CNN_aux, self).__init__()
        self.input_channels = input_channels
        self.latent_size = latent_size
        self.beta = beta
        self.reg_intra = reg_intra
        self.reg_inter = reg_inter
        self.u_idx = u_idx
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_size)
        self.fc_var = nn.Linear(256 * 2 * 2, latent_size)
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 5, 4, 0),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_var(x)
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        z = z.view(z.size(0), self.latent_size, 1, 1)
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, z, mu, logvar
    

    def loss_log_mse_VAE(self, recon_x, x, z, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        kl_diverge = -0.5 * torch.sum(1 + logvar - (mu).pow(2) - logvar.exp())
        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]) 
        
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]), kl_diverge / (x.shape[0]*x.shape[1]*x.shape[3])
    
    

    
    def loss_log_mse_reduced_cor(self, recon_x, x, z, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        #u0 = torch.cat((u[:, :3], torch.zeros(u.shape[0], (mu.shape[1] - 3)).to(u.device)), dim=1)
        u0 = torch.cat((u[:, self.u_idx], torch.zeros(u.shape[0], (mu.shape[1] - u[:, self.u_idx].shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        # Calculate maximum absolute correlation between last 5 dimensions of mu and u
        #max_corr = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        def create_interaction_array(u):
            n, p = u.shape
            interaction_array = torch.zeros((n, p * (p - 1) // 2), dtype=u.dtype, device=u.device)
            col_idx = 0
            for i in range(p):
                for j in range(i + 1, p):
                    interaction_array[:, col_idx] = u[:, i] * u[:, j]
                    col_idx += 1
            return interaction_array
        
        u_interaction= create_interaction_array(u[:,self.u_idx])
        
        d=len(self.u_idx)
        d_recon=self.latent_size-d
        max_corr=torch.zeros(10)
        max_corr[0] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[1] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[2] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[3] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[4] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[5] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[6] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[7] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[8] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[9] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u_interaction),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        
        z_aux_corr=torch.corrcoef(torch.cat((z[:, :d], u[:,self.u_idx]),axis=1).T)[:d,d:]
        aux_corr=torch.zeros(2)
        aux_corr[0] = torch.sum(1-torch.diag(z_aux_corr))
        aux_corr[1] = torch.sum(torch.abs(z_aux_corr))-torch.sum(torch.diag(torch.abs(z_aux_corr)))
        
        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]) + torch.sum(max_corr)*self.reg_inter + torch.sum(aux_corr)*self.reg_intra #+ self.beta*torch.sum(torch.sum((z[:,:3]-u[:,:3])**2, axis=0))*self.reg_inter/z.shape[0]#3
        
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]), kl_diverge / (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3])
    
    
    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)
    
    def load_model(self, file_path):
        utils.restore(self, file_path)
    
    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)




class BetaVAE_MLP_aux(nn.Module):
    def __init__(self, input_size=64*64, u_idx=[0,1,2], latent_size=32, beta=1, reg_intra=1, reg_inter=1):
        super(BetaVAE_MLP_aux, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.beta = beta
        self.u_idx = u_idx
        self.reg_intra = reg_intra
        self.reg_inter = reg_inter
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_size)
        self.fc_var = nn.Linear(128, latent_size)
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar
    
    def loss(self, recon_x, x, u, mu, logvar):
        #recon_loss = F.mse_loss(recon_x, x.view(x.size(0), -1), reduction='sum')
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, self.input_size), x.view(-1, self.input_size), reduction='sum')
        u0= torch.cat((u, torch.zeros(mu.shape[0], (mu.shape[1]-u.shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0-mu).pow(2) - logvar.exp())
        #kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return ((recon_loss + self.beta * kl_diverge) / x.size(0)), recon_loss / x.size(0), kl_diverge / x.size(0)
    
    def loss_log_mse_VAE(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        #u0 = torch.cat((u, torch.zeros(mu.shape[0], (mu.shape[1] - u.shape[1])).to(u.device)), dim=1)
        
        # Calculate KL divergence
        kl_diverge = -0.5 * torch.sum(1 + logvar - (mu).pow(2) - logvar.exp())
        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / (x.shape[0]*x.shape[1])
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/(x.shape[0]*x.shape[1]), kl_diverge / (x.shape[0]*x.shape[1])


    
    def loss_log_mse(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        u0 = torch.cat((u, torch.zeros(mu.shape[0], (mu.shape[1] - u.shape[1])).to(u.device)), dim=1)
        
        # Calculate KL divergence
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        
        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / x.size(0)
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/x.size(0), kl_diverge / x.size(0)

    def loss_log_mse_cor(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        u0 = torch.cat((u, torch.zeros(mu.shape[0], (mu.shape[1] - u.shape[1])).to(u.device)), dim=1)
        
        # Calculate KL divergence
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        # Calculate maximum absolute correlation between last 5 dimensions of mu and 
        max_corr = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, u.shape[1]:], u),axis=1).T)[:(mu.shape[1]-u.shape[1]),(mu.shape[1]-u.shape[1]):]),axis=1))/(mu.shape[1]-u.shape[1])
        cor_1=torch.abs(torch.corrcoef(mu[:,:u.shape[1]].T))
        #max_corr_1= (torch.sum(cor_1)-torch.sum(torch.diagonal(cor_1)))/u.shape[1]
        max_corr_1= torch.max(cor_1.flatten()[1:].view(u.shape[1]-1, u.shape[1]+1)[:,:-1].reshape(u.shape[1], u.shape[1]-1))
        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / x.size(0) + self.beta*max_corr/3 
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/x.size(0), kl_diverge / x.size(0)
    
    def loss_log_mse_reduced(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        u0 = torch.cat((u[:, :3], torch.zeros(u.shape[0], (mu.shape[1] - 3)).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())

        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / x.size(0)
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/x.size(0), kl_diverge / x.size(0)
    
    def loss_log_mse_reduced_cor(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        #u0 = torch.cat((u[:, :3], torch.zeros(u.shape[0], (mu.shape[1] - 3)).to(u.device)), dim=1)
        u0 = torch.cat((u[:, self.u_idx], torch.zeros(u.shape[0], (mu.shape[1] - u[:, self.u_idx].shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        # Calculate maximum absolute correlation between last 5 dimensions of mu and u
        #max_corr = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        def create_interaction_array(u):
            n, p = u.shape
            interaction_array = torch.zeros((n, p * (p - 1) // 2), dtype=u.dtype, device=u.device)
            col_idx = 0
            for i in range(p):
                for j in range(i + 1, p):
                    interaction_array[:, col_idx] = u[:, i] * u[:, j]
                    col_idx += 1
            return interaction_array
        
        u_interaction= create_interaction_array(u[:,self.u_idx])
        
        d=len(self.u_idx)
        d_recon=self.latent_size-d
        max_corr=torch.zeros(10)
        max_corr[0] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[1] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[2] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[3] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[4] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[5] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[6] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[7] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[8] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[9] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u_interaction),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon

        mu_aux_corr=torch.corrcoef(torch.cat((mu[:, :d], u[:,self.u_idx]),axis=1).T)[:d,d:]
        aux_corr=torch.zeros(2)
        aux_corr[0] = torch.sum(1-torch.diag(mu_aux_corr))
        aux_corr[1] = torch.sum(torch.abs(mu_aux_corr))-torch.sum(torch.diag(torch.abs(mu_aux_corr)))

        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / (x.shape[0]*x.shape[1]) + torch.sum(max_corr)*self.reg_inter + torch.sum(aux_corr)*self.reg_intra #+ self.beta*torch.sum(torch.sum((mu[:,:3]-u[:,:3])**2, axis=0))*self.reg_inter/mu.shape[0]#3
        
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/(x.shape[0]*x.shape[1]), kl_diverge / (x.shape[0]*x.shape[1])

    def loss_log_mse_reduced_cor_DSprites(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        u0 = torch.cat((u[:, self.u_idx], torch.zeros(u.shape[0], (mu.shape[1] - u[:, self.u_idx].shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        # Calculate maximum absolute correlation between last 5 dimensions of mu and u
        #max_corr = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        def create_interaction_array(u):
            n, p = u.shape
            interaction_array = torch.zeros((n, p * (p - 1) // 2), dtype=u.dtype, device=u.device)
            col_idx = 0
            for i in range(p):
                for j in range(i + 1, p):
                    interaction_array[:, col_idx] = u[:, i] * u[:, j]
                    col_idx += 1
            return interaction_array
        u_interaction= create_interaction_array(u[:,self.u_idx])

        d=len(self.u_idx)
        d_recon=self.latent_size-d
        max_corr=torch.zeros(10)
        max_corr[0] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[1] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[2] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[3] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[4] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[5] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[6] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[7] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[8] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[9] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u_interaction),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon

        mu_aux_corr=torch.corrcoef(torch.cat((mu[:, :d], u[:,self.u_idx]),axis=1).T)[:d,d:]
        aux_corr=torch.zeros(2)
        aux_corr[0] = torch.sum(1-torch.diag(mu_aux_corr))
        aux_corr[1] = torch.sum(torch.abs(mu_aux_corr))-torch.sum(torch.diag(torch.abs(mu_aux_corr)))
        

        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / (x.shape[0]*x.shape[1]) + torch.sum(max_corr)*self.reg_inter + torch.sum(aux_corr)*self.reg_intra #+ self.beta*100*torch.sum(torch.sum((mu[:,:4]-u[:,[1,2,4,5]])**2, axis=0))*self.reg_inter/mu.shape[0]#3
        
        #F.binary_cross_entropy(log_recon_x, log_x, reduction='sum')
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/(x.shape[0]*x.shape[1]), kl_diverge / (x.shape[0]*x.shape[1])
    
    def loss_log_mse_reduced_cor_gaussian(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        u0 = torch.cat((u, torch.zeros(u.shape[0], (mu.shape[1] - u.shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        # Calculate maximum absolute correlation between last 5 dimensions of mu and u
        #max_corr = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        
        max_corr=torch.zeros(9)
        max_corr[0] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 2:], u[:,:2]),axis=1).T)[:8,8:]),axis=1))/8
        max_corr[1] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 2:], u[:,:2]**2),axis=1).T)[:8,8:]),axis=1))/8
        max_corr[2] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 2:], u[:,:2]**3),axis=1).T)[:8,8:]),axis=1))/8
        max_corr[3] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 2:]**2, u[:,:2]),axis=1).T)[:8,8:]),axis=1))/8
        max_corr[4] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 2:]**2, u[:,:2]**2),axis=1).T)[:8,8:]),axis=1))/8
        max_corr[5] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 2:]**2, u[:,:2]**3),axis=1).T)[:8,8:]),axis=1))/8
        max_corr[6] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 2:]**3, u[:,:2]),axis=1).T)[:8,8:]),axis=1))/8
        max_corr[7] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 2:]**3, u[:,:2]**2),axis=1).T)[:8,8:]),axis=1))/8
        max_corr[8] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 2:]**3, u[:,:2]**3),axis=1).T)[:8,8:]),axis=1))/8
        
        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / x.size(0) + self.beta*torch.sum(max_corr)*.05 #3
        
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/x.size(0), kl_diverge / x.size(0)
    
    def loss_log_mse_reduced_cor_mean_var(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        var=torch.exp(logvar)
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        u0 = torch.cat((u[:, :3], torch.zeros(u.shape[0], (mu.shape[1] - 3)).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        # Calculate maximum absolute correlation between last 5 dimensions of mu and u
        #max_corr = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        
        max_corr_mean=torch.zeros(4)
        max_corr_mean[0] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        max_corr_mean[1] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]**2),axis=1).T)[:7,7:]),axis=1))/7
        max_corr_mean[2] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:]**2, u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        max_corr_mean[3] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:]**2, u[:,:3]**2),axis=1).T)[:7,7:]),axis=1))/7
        
        max_corr_var=torch.zeros(4)
        max_corr_var[0] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((var[:, 3:], u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        max_corr_var[1] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((var[:, 3:], u[:,:3]**2),axis=1).T)[:7,7:]),axis=1))/7
        max_corr_var[2] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((var[:, 3:]**2, u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        max_corr_var[3] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((var[:, 3:]**2, u[:,:3]**2),axis=1).T)[:7,7:]),axis=1))/7
        
        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / x.size(0) + self.beta*torch.sum(max_corr_mean)*.01 + self.beta*torch.sum(max_corr_var)*0 + self.beta*torch.sum(var[:,:3])*0.05 #3
        
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/x.size(0), kl_diverge / x.size(0)

    
    def loss_log_mse_reduced_two_aux(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        u0 = torch.cat((u[:, :2], torch.zeros(u.shape[0], (mu.shape[1] - 2)).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())

        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / x.size(0)
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/x.size(0), kl_diverge / x.size(0)
    
    def loss_log_mse_split_corr(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Calculate KL divergence terms
        kl_1 = -self.beta * (u - mu[:, :u.shape[1]])**2
        kl_2 = -(mu[:, u.shape[1]:])**2
        kl_3 = -self.beta * (logvar[:, :u.shape[1]].exp())
        kl_4 = -(logvar[:, u.shape[1]:].exp())
        kl_5 = self.beta * (logvar[:, :u.shape[1]])
        kl_6 = logvar[:, u.shape[1]:]
        
        # Compute KL divergence
        kl_diverge = -0.5 * torch.sum(self.beta + torch.sum(kl_1, axis=1) + torch.sum(kl_2, axis=1) + torch.sum(kl_3, axis=1) + torch.sum(kl_4, axis=1) + torch.sum(kl_5, axis=1) + torch.sum(kl_6, axis=1))
        
        # Calculate maximum absolute correlation between last 5 dimensions of mu and u
        max_corr = torch.abs(torch.max(torch.abs(torch.corrcoef(mu[:, -5:].T, u.T))))

        # Add maximum absolute correlation to the total loss
        total_loss = (mse_loss + kl_diverge + max_corr) / x.size(0)
        
        # Return individual components of the loss for logging purposes
        return total_loss, mse_loss/x.size(0), kl_diverge / x.size(0)
        
    def loss_log_mse_split(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        #u0 = torch.cat((u[:, :3], torch.zeros(u.shape[0], (mu.shape[1] - 3)).to(u.device)), dim=1)
        kl_1= -self.beta*(u-mu[:,:u.shape[1]])**2
        kl_2= -(mu[:,u.shape[1]:])**2
        kl_3= -self.beta*(logvar[:,:u.shape[1]].exp())
        kl_4= -(logvar[:,u.shape[1]:].exp())
        kl_5= self.beta*(logvar[:,:u.shape[1]])
        kl_6= logvar[:,u.shape[1]:]
        kl_diverge = -0.5 * torch.sum(self.beta + torch.sum(kl_1, axis=1)+ torch.sum(kl_2, axis=1)+ torch.sum(kl_3, axis=1)+ torch.sum(kl_4, axis=1)+ torch.sum(kl_5, axis=1)+ torch.sum(kl_6, axis=1))
        
        
        # Compute total loss
        total_loss = (mse_loss +  kl_diverge) / x.size(0)
        
        # Return individual components of the loss for logging purposes
        return total_loss, mse_loss/x.size(0), kl_diverge / x.size(0)
    
    def loss_log_mse_reduced_split_corr(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        #u0 = torch.cat((u[:, :3], torch.zeros(u.shape[0], (mu.shape[1] - 3)).to(u.device)), dim=1)
        kl_1= -self.beta*(u[:, :3]-mu[:,:3])**2
        kl_2= -(mu[:,3:])**2
        kl_3= -self.beta*(logvar[:,:3].exp())
        kl_4= -(logvar[:,3:].exp())
        kl_5= self.beta*(logvar[:,:3])
        kl_6= logvar[:,3:]
        kl_diverge = -0.5 * torch.sum(self.beta + torch.sum(kl_1, axis=1)+ torch.sum(kl_2, axis=1)+ torch.sum(kl_3, axis=1)+ torch.sum(kl_4, axis=1)+ torch.sum(kl_5, axis=1)+ torch.sum(kl_6, axis=1))
        
        # Calculate maximum absolute correlation between last 5 dimensions of mu and u
        max_corr = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        
        # Add maximum absolute correlation to the total loss
        total_loss = (mse_loss + kl_diverge) / x.size(0) + (self.beta*8)*max_corr
        
        # Return individual components of the loss for logging purposes
        return total_loss, mse_loss/x.size(0), kl_diverge / x.size(0)
    
    def loss_log_mse_reduced_split(self, recon_x, x, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        #u0 = torch.cat((u[:, :3], torch.zeros(u.shape[0], (mu.shape[1] - 3)).to(u.device)), dim=1)
        kl_1= -self.beta*(u[:, :3]-mu[:,:3])**2
        kl_2= -(mu[:,3:])**2
        kl_3= -self.beta*(logvar[:,:3].exp())
        kl_4= -(logvar[:,3:].exp())
        kl_5= self.beta*(logvar[:,:3])
        kl_6= logvar[:,3:]
        kl_diverge = -0.5 * torch.sum(self.beta + torch.sum(kl_1, axis=1)+ torch.sum(kl_2, axis=1)+ torch.sum(kl_3, axis=1)+ torch.sum(kl_4, axis=1)+ torch.sum(kl_5, axis=1)+ torch.sum(kl_6, axis=1))
        
        
        # Compute total loss
        total_loss = (mse_loss +  kl_diverge) / x.size(0)
        
        # Return individual components of the loss for logging purposes
        return total_loss, mse_loss/x.size(0), kl_diverge / x.size(0)
    
    def loss_spatial_var(self, recon_x, x, u, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x.view(x.size(0), -1), reduction='sum')
        u0= torch.cat((u, torch.zeros(mu.shape[0], (mu.shape[1]-u.shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0-mu).pow(2) - logvar.exp())
        #kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    

        def divide_image_into_blocks(image, block_size):
            p = image.shape[-1]  # Dimension of the image (assuming it's square)
            k = block_size  # Size of each block
            if p % k != 0:
                raise ValueError("Image dimensions should be divisible by the block size")
            
            num_blocks = p // k
            blocks = []
            for i in range(num_blocks):
                for j in range(num_blocks):
                    block = image[..., i * k:(i + 1) * k, j * k:(j + 1) * k]
                    blocks.append(block.reshape(-1))
            
            return torch.stack(blocks)
        
        def spatial_variability(x, k=14):
            num_samples, _, _ = x.shape
            spatial_variability_per_sample = []
            for i in range(num_samples):
                sample = x[i]
                blocks = divide_image_into_blocks(sample.unsqueeze(0), k)
                n_blocks = len(blocks)
                # Checking the average distance among the blocks
                zero_sd_blocks = [j for j, block in enumerate(blocks) if torch.std(block) == 0]
                valid_blocks_pos = list(range(1, n_blocks + 1))
                if zero_sd_blocks:
                    valid_blocks_pos = [j for j in range(1, n_blocks + 1) if j not in zero_sd_blocks]
                
                nonzero_blocks = [blocks[j - 1] for j in valid_blocks_pos]
                n_nonzero_blocks = len(nonzero_blocks)
                all_blocks_as_row = torch.cat(nonzero_blocks)
                dist_matrix = torch.cdist(all_blocks_as_row.view(n_nonzero_blocks, -1), all_blocks_as_row.view(n_nonzero_blocks, -1))
                # Checking the spatial variation via the average distance among blocks
                B = int(np.sqrt(n_blocks))
                def row_index(bl):
                    return bl // B if bl % B == 0 else bl // B + 1
                
                def col_index(bl):
                    return B if bl % B == 0 else bl % B
                
                def spatial_var_per_block(b):
                    weights_pos = [abs(row_index(valid_blocks_pos[b - 1]) - row_index(valid_blocks_pos[j - 1])) +
                                abs(col_index(valid_blocks_pos[b - 1]) - col_index(valid_blocks_pos[j - 1]))
                                for j in range(1, n_nonzero_blocks + 1)]
                    mask = np.ones(len(weights_pos), dtype=bool)
                    mask[b - 1] = False
                    spatial_var_measure = (torch.sum(torch.sqrt(torch.tensor(weights_pos, device=x.device))[mask] *
                                                    dist_matrix[b - 1][mask])
                                        / torch.sum(torch.sqrt(torch.tensor(weights_pos, device=x.device))[mask]))
                    return spatial_var_measure
                
                spatial_var_all = [spatial_var_per_block(b) for b in range(1, n_nonzero_blocks + 1)]
                #weights_intra = [torch.sum([(block - block[j]) ** 2 for j in range(k ** 2)]) / (k ** 2 * (k ** 2 - 1))
                                #for block in nonzero_blocks]
                weights_intra = [torch.var(block) for block in nonzero_blocks]
                spatial_var_final = -torch.sum(torch.Tensor(weights_intra) * torch.Tensor(spatial_var_all)) * torch.log(1 / torch.Tensor([n_blocks])) * (1 / torch.Tensor([n_blocks]))
                spatial_variability_per_sample.append(spatial_var_final)
            
            return torch.stack(spatial_variability_per_sample)
        
        print("Spatial var calculation starts")
        x_reshaped = x.view(-1, int(np.sqrt(self.input_size)), int(np.sqrt(self.input_size)))   # Reshape x to [7, 33, 33]
        x_variability = spatial_variability(x_reshaped, k=3)
        print("done for x")
        recon_x_reshaped = recon_x.view(-1, int(np.sqrt(self.input_size)), int(np.sqrt(self.input_size)))  # Reshape x to [7, 33, 33]
        recon_x_variability = spatial_variability(recon_x_reshaped, k=3)
        print("done for recon_x")
        variance_difference = torch.abs(x_variability - recon_x_variability)/x_variability

        total_loss = (recon_loss + self.beta * kl_diverge) / x.size(0)
        total_loss += 0.000001 * variance_difference.mean()  # Adjust some_weight as needed
        return total_loss, recon_loss / x.size(0), kl_diverge / x.size(0)
    

    def divide_image_into_blocks(self, image, block_size):
        p = image.shape[-1]  # Dimension of the image (assuming it's square)
        k = block_size  # Size of each block
        if p % k != 0:
            raise ValueError("Image dimensions should be divisible by the block size")
        
        num_blocks = p // k
        blocks = []
        for i in range(num_blocks):
            for j in range(num_blocks):
                block = image[..., i * k:(i + 1) * k, j * k:(j + 1) * k]
                blocks.append(block.reshape(-1))
        
        return torch.stack(blocks)
    
    def spatial_variability_batch(self, x, k=14):
        num_samples, height, width = x.shape
        spatial_variability_per_sample = []
        
        for i in range(num_samples):
            sample = x[i]
            blocks = self.divide_image_into_blocks(sample.unsqueeze(0), k)
            n_blocks = len(blocks)
            
            # Checking the average distance among the blocks
            zero_sd_blocks = [j for j, block in enumerate(blocks) if torch.std(block) == 0]
            valid_blocks_pos = list(range(1, n_blocks + 1))
            if zero_sd_blocks:
                valid_blocks_pos = [j for j in range(1, n_blocks + 1) if j not in zero_sd_blocks]
            
            nonzero_blocks = [blocks[j - 1] for j in valid_blocks_pos]
            n_nonzero_blocks = len(nonzero_blocks)
            all_blocks_as_row = torch.cat(nonzero_blocks)
            dist_matrix = torch.cdist(all_blocks_as_row.view(n_nonzero_blocks, -1), all_blocks_as_row.view(n_nonzero_blocks, -1))
            # Checking the spatial variation via the average distance among blocks
            B = int(np.sqrt(n_blocks))
            def row_index(bl):
                return bl // B if bl % B == 0 else bl // B + 1
            
            def col_index(bl):
                return B if bl % B == 0 else bl % B
            
            def spatial_var_per_block(b):
                weights_pos = [abs(row_index(valid_blocks_pos[b - 1]) - row_index(valid_blocks_pos[j - 1])) +
                            abs(col_index(valid_blocks_pos[b - 1]) - col_index(valid_blocks_pos[j - 1]))
                            for j in range(1, n_nonzero_blocks + 1)]
                mask = np.ones(len(weights_pos), dtype=bool)
                mask[b - 1] = False
                spatial_var_measure = (torch.sum(torch.sqrt(torch.tensor(weights_pos, device=x.device))[mask] *
                                                dist_matrix[b - 1][mask])
                                    / torch.sum(torch.sqrt(torch.tensor(weights_pos, device=x.device))[mask]))
                return spatial_var_measure
            
            spatial_var_all = [spatial_var_per_block(b) for b in range(1, n_nonzero_blocks + 1)]
            #weights_intra = [torch.sum([(block - block[j]) ** 2 for j in range(k ** 2)]) / (k ** 2 * (k ** 2 - 1))
                            #for block in nonzero_blocks]
            weights_intra = [torch.var(block) for block in nonzero_blocks]
            spatial_var_final = -torch.sum(torch.Tensor(weights_intra) * torch.Tensor(spatial_var_all)) * torch.log(1 / torch.Tensor([n_blocks])) * (1 / torch.Tensor([n_blocks]))
            spatial_variability_per_sample.append(spatial_var_final)
            
            spatial_variability_per_sample.append(spatial_var_final)
        
        return torch.stack(spatial_variability_per_sample)

    def loss_spatial_var_parallel(self, recon_x, x, u, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x.view(x.size(0), -1), reduction='sum')
        u0= torch.cat((u, torch.zeros(mu.shape[0], (mu.shape[1]-u.shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0-mu).pow(2) - logvar.exp())
        #kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        print("Spatial var calculation starts")
        x_reshaped = x.view(-1, int(np.sqrt(self.input_size)), int(np.sqrt(self.input_size)))  # Reshape x to [batch_size, num_channels, 33, 33]
        x_variability = self.spatial_variability_batch(x_reshaped, k=3)
        print("done for x")
        
        recon_x_reshaped = recon_x.view(-1, int(np.sqrt(self.input_size)), int(np.sqrt(self.input_size)))  # Reshape x to [batch_size, num_channels, 33, 33]
        recon_x_variability = self.spatial_variability_batch(recon_x_reshaped, k=3)
        print("done for recon_x")
        
        variance_difference = torch.abs(x_variability - recon_x_variability)/x_variability

        total_loss = (recon_loss + self.beta * kl_diverge) / x.size(0)
        total_loss += 0.000001 * variance_difference.mean()  # Adjust some_weight as needed
        return total_loss, recon_loss / x.size(0), kl_diverge / x.size(0)
    
    
    def loss_var(self, recon_x, x, u, mu, logvar):
        #recon_loss = F.binary_cross_entropy(recon_x.view(-1, self.input_size), x.view(-1, self.input_size), reduction='sum')
        #w = torch.full_like(x, 0)
        #w[x >= 0.03] = 0.8
        # Define the conditions
        condition1 = (x <= 0.001) | (x >= 0.1)
        condition2 = ~condition1
        # Initialize w with zeros
        w = torch.zeros_like(x)
        # Apply the conditions to update w accordingly
        w[condition1] = 1
        #w[condition2] = (x[condition2] - 0.00001) * (0.01 - x[condition2])*100000 + 1  
        #w[condition2] = (x[condition2] - 0.00001) * (0.1 - x[condition2])*3000 + 1  
        #w[condition2] = (x[condition2] - 0.00001) * (0.03 - x[condition2])*10000 + 1  
        #w[condition2] = (x[condition2] - 0.001) * (0.05 - x[condition2])*5000 + 1  
        w[condition2] = (x[condition2] - 0.001) * (0.1 - x[condition2])*1000 + 1 
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, self.input_size), x.view(-1, self.input_size),weight= w, reduction='sum')
        # Calculate the variance of the original and reconstructed images
        variance_x = torch.var(x.view(x.size(0), -1), dim=1)
        variance_recon_x = torch.var(recon_x.view(recon_x.size(0), -1), dim=1)
        # Calculate the absolute difference in variances
        variance_difference = torch.abs(variance_x - variance_recon_x)
        u0 = torch.cat((u, torch.zeros(mu.shape[0], (mu.shape[1] - u.shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        # Combine the terms in your loss function as needed
        total_loss = (recon_loss + self.beta * kl_diverge) / x.size(0)
        # You can add the variance difference term to the total loss
        #total_loss += 10000 * variance_difference.mean()  # Adjust some_weight as needed
        return total_loss, recon_loss / x.size(0), kl_diverge / x.size(0)
    
    def loss_var_local_weighting(self, recon_x, x, u, mu, logvar):
        # Calculate the 25th and 95th percentiles of x
        x_numpy = x.cpu().detach().numpy()
        # Calculate the 25th and 95th percentiles of x along both spatial dimensions
        lower_bound = torch.tensor(np.percentile(x_numpy, 25, axis=(1))).to(x.device).view(-1, 1)
        upper_bound = torch.tensor(np.percentile(x_numpy, 95, axis=(1))).to(x.device).view(-1, 1)
        
        
        # Define condition 1 based on percentiles
        condition1 = (x <= lower_bound) | (x >= upper_bound)
        condition2 = ~condition1
        
        # Initialize weights with zeros
        w = torch.zeros_like(x, dtype=torch.float)
        for i in range(x.shape[0]):
            condition1 = (x[i,:] <= lower_bound[i]) | (x[i,:] >= upper_bound[i])
            condition2 = ~condition1
            w[i,condition1]=1
            w[i,condition2]=((x[i,condition2] - lower_bound[i]) * (upper_bound[i] - x[i,condition2]) * 10000 + 1).to(w.dtype)
        
        # Calculate binary cross-entropy loss with weighted samples
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, self.input_size), x.view(-1, self.input_size),
                                            weight=w, reduction='sum')
        
        # Calculate KL divergence
        u0 = torch.cat((u, torch.zeros(mu.shape[0], (mu.shape[1] - u.shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        
        # Compute total loss
        total_loss = (recon_loss + self.beta * kl_diverge) / x.size(0)
        
        # Return individual components of the loss for logging purposes
        return total_loss, recon_loss / x.size(0), kl_diverge / x.size(0)
    
    def loss_var_reduced_u(self, recon_x, x, u, mu, logvar):
        # Define the conditions
        condition1 = (x <= 0.00001) | (x >= 0.03)
        condition2 = ~condition1
        # Initialize w with zeros
        w = torch.zeros_like(x)
        # Apply the conditions to update w accordingly
        w[condition1] = 1
        w[condition2] = (x[condition2] - 0.00001) * (0.03 - x[condition2]) * 10000 + 1
        
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, self.input_size), x.view(-1, self.input_size), weight=w, reduction='sum')
        
        # Calculate the variance of the original and reconstructed images
        variance_x = torch.var(x.view(x.size(0), -1), dim=1)
        variance_recon_x = torch.var(recon_x.view(recon_x.size(0), -1), dim=1)
        # Calculate the absolute difference in variances
        variance_difference = torch.abs(variance_x - variance_recon_x)
        # Use only the first three dimensions of u in u0
        u0 = torch.cat((u[:, :3], torch.zeros(u.shape[0], (mu.shape[1] - 3)).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        
        # Combine the terms in your loss function as needed
        total_loss = (recon_loss + self.beta * kl_diverge) / x.size(0)
        # You can add the variance difference term to the total loss
        # total_loss += 10000 * variance_difference.mean()  # Adjust some_weight as needed
        
        return total_loss, recon_loss / x.size(0), kl_diverge / x.size(0)
    
    def loss_var_gaussian(self, recon_x, x, u, mu, logvar):
        recon_loss = F.mse_loss(recon_x.view(-1, self.input_size), x.view(-1, self.input_size), reduction='sum')
        #w = torch.full_like(x, 0)
        #w[x >= 0.03] = 0.8
        # Define the conditions
        #condition1 = (x <= 0.00001) | (x >= 0.03)
        #condition2 = ~condition1
        # Initialize w with zeros
        #w = torch.zeros_like(x)
        # Apply the conditions to update w accordingly
        #w[condition1] = 1
        #w[condition2] = (x[condition2] - 0.00001) * (0.01 - x[condition2])*50000 + 1  
        #w[condition2] = (x[condition2] - 0.00001) * (0.1 - x[condition2])*3000 + 1  
        #w[condition2] = (x[condition2] - 0.00001) * (0.03 - x[condition2])*10000 + 1  
        #recon_loss = F.binary_cross_entropy(recon_x.view(-1, self.input_size), x.view(-1, self.input_size),weight= w, reduction='sum')
        # Calculate the variance of the original and reconstructed images
        variance_x = torch.var(x.view(x.size(0), -1), dim=1)
        variance_recon_x = torch.var(recon_x.view(recon_x.size(0), -1), dim=1)
        # Calculate the absolute difference in variances
        variance_difference = torch.abs(variance_x - variance_recon_x)
        u0 = torch.cat((u, torch.zeros(mu.shape[0], (mu.shape[1] - u.shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        # Combine the terms in your loss function as needed
        total_loss = (recon_loss + self.beta * kl_diverge) / x.size(0)
        # You can add the variance difference term to the total loss
        #total_loss += 10000 * variance_difference.mean()  # Adjust some_weight as needed
        return total_loss, recon_loss / x.size(0), kl_diverge / x.size(0)
    
    
    def loss_svar(self, recon_x, x, u, mu, logvar):
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, self.input_size), x.view(-1, self.input_size), reduction='sum')
        # Calculate the variance of the original and reconstructed images
        variance_x = torch.var(x.view(x.size(0), -1), dim=1)
        variance_recon_x = torch.var(recon_x.view(recon_x.size(0), -1), dim=1)
        variance_difference = torch.abs(variance_x - variance_recon_x)
        # Calculate the absolute difference in variances
        svariance_difference = torch.abs( F.pdist(x) - F.pdist(recon_x))
        u0 = torch.cat((u, torch.zeros(mu.shape[0], (mu.shape[1] - u.shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        # Combine the terms in your loss function as needed
        total_loss = (recon_loss + self.beta * kl_diverge) / x.size(0)
        # You can add the variance difference term to the total loss
        total_loss += 10 * svariance_difference.mean()  # Adjust some_weight as needed
        total_loss += 1000 * variance_difference.mean()  # Adjust some_weight as needed
        return total_loss, recon_loss / x.size(0), kl_diverge / x.size(0)
    
    def loss_relative(self, recon_x, x, u, mu, logvar):
        # Calculate the squared relative errors
        squared_relative_errors = ((x.view(x.size(0), -1) - recon_x).pow(2)) / x.view(x.size(0), -1)
        # Sum the squared relative errors
        relative_mse_loss = squared_relative_errors.sum()
        u0 = torch.cat((u, torch.zeros(mu.shape[0], (mu.shape[1] - u.shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        return ((relative_mse_loss + self.beta * kl_diverge) / x.size(0)), relative_mse_loss / x.size(0), kl_diverge / x.size(0)
    
    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)
    
    def load_model(self, file_path):
        utils.restore(self, file_path)
    
    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)



class BetaVAE_MLP(nn.Module):
    def __init__(self, input_size=64*64, latent_size=32, beta=1):
        super(BetaVAE_MLP, self).__init__()
        
        self.latent_size = latent_size
        self.beta = beta
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_size)
        self.fc_var = nn.Linear(128, latent_size)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction='sum')
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return ((recon_loss + self.beta * kl_diverge) / x.size(0)), recon_loss / x.size(0), kl_diverge / x.size(0)
    
    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)
    
    def load_model(self, file_path):
        utils.restore(self, file_path)
    
    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)


class BetaVAE_CNN_aux_Cars3D(nn.Module):
    def __init__(self, input_channels=3, latent_size=32, beta=1, reg_intra=1, reg_inter=1, u_idx=[0, 1, 2]):
        super(BetaVAE_CNN_aux_Cars3D, self).__init__()
        self.input_channels = input_channels
        self.latent_size = latent_size
        self.beta = beta
        self.reg_intra = reg_intra
        self.reg_inter = reg_inter
        self.u_idx = u_idx
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_size)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_size)
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 128, 4, 2, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 4, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_var(x)
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        z = z.view(z.size(0), self.latent_size, 1, 1)
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, z, mu, logvar
    
    def loss_log_mse_VAE(self, recon_x, x, z, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        kl_diverge = -0.5 * torch.sum(1 + logvar - (mu).pow(2) - logvar.exp())
        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]) 
        
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]), kl_diverge / (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3])
    
    
    def loss_log_mse_reduced_cor(self, recon_x, x, z, u, mu, logvar):
        # Apply log transformation to original and reconstructed images
        log_recon_x = torch.log10(recon_x + 1e-10)
        log_x = torch.log10(x + 1e-10)
        
        # Calculate relative MSE on log-transformed images
        relative_mse_loss = torch.sum(((log_recon_x - log_x))**2) 
        
        # Concatenate u with zeros to match the dimensionality of mu
        #u0 = torch.cat((u[:, :3], torch.zeros(u.shape[0], (mu.shape[1] - 3)).to(u.device)), dim=1)
        u0 = torch.cat((u[:, self.u_idx], torch.zeros(u.shape[0], (mu.shape[1] - u[:, self.u_idx].shape[1])).to(u.device)), dim=1)
        kl_diverge = -0.5 * torch.sum(1 + logvar - (u0 - mu).pow(2) - logvar.exp())
        # Calculate maximum absolute correlation between last 5 dimensions of mu and u
        #max_corr = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, 3:], u[:,:3]),axis=1).T)[:7,7:]),axis=1))/7
        def create_interaction_array(u):
            n, p = u.shape
            interaction_array = torch.zeros((n, p * (p - 1) // 2), dtype=u.dtype, device=u.device)
            col_idx = 0
            for i in range(p):
                for j in range(i + 1, p):
                    interaction_array[:, col_idx] = u[:, i] * u[:, j]
                    col_idx += 1
            return interaction_array
        
        u_interaction= create_interaction_array(u[:,self.u_idx])
        
        d=len(self.u_idx)
        d_recon=self.latent_size-d
        max_corr=torch.zeros(10)
        max_corr[0] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[1] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[2] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[3] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[4] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[5] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**2, u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[6] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[7] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]**2),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[8] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:]**3, u[:,self.u_idx]**3),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        max_corr[9] = torch.sum(torch.sum(torch.abs(torch.corrcoef(torch.cat((mu[:, d:], u_interaction),axis=1).T)[:d_recon,d_recon:]),axis=1))/d_recon
        
        z_aux_corr=torch.corrcoef(torch.cat((z[:, :d], u[:,self.u_idx]),axis=1).T)[:d,d:]
        aux_corr=torch.zeros(2)
        aux_corr[0] = torch.sum(1-torch.diag(z_aux_corr))
        aux_corr[1] = torch.sum(torch.abs(z_aux_corr))-torch.sum(torch.diag(torch.abs(z_aux_corr)))
        
        # Compute total loss
        total_loss = (relative_mse_loss + self.beta * kl_diverge) / (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]) + self.beta*torch.sum(max_corr)*self.reg_inter + self.beta*torch.sum(aux_corr)*self.reg_intra #+ self.beta*torch.sum(torch.sum((z[:,:3]-u[:,:3])**2, axis=0))*self.reg_inter/z.shape[0]#3
        
        
        # Return individual components of the loss for logging purposes
        return total_loss, relative_mse_loss/(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]), kl_diverge / (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3])
    
    
    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)
    
    def load_model(self, file_path):
        utils.restore(self, file_path)
    
    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)



class BetaVAE(nn.Module):
    
    def __init__(self, latent_size=32, beta=1):
        super(BetaVAE, self).__init__()
        
        self.latent_size = latent_size
        self.beta = beta
        
        # encoder
        self.encoder = nn.Sequential(
            self._conv(1, 32),
            self._conv(32, 32),
            self._conv(32, 64),
            self._conv(64, 64),
        )
        self.fc_mu = nn.Linear(256, latent_size)
        self.fc_var = nn.Linear(256, latent_size)
        
        # decoder
        self.decoder = nn.Sequential(
            self._deconv(64, 64),
            self._deconv(64, 32),
            self._deconv(32, 32, 1),
            self._deconv(32, 1),
            nn.Sigmoid()
        )
        self.fc_z = nn.Linear(latent_size, 256)
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256)
        return self.fc_mu(x), self.fc_var(x)
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)
    
    #def decode(self, z):
        #z = self.fc_z(z)
        #z = z.view(-1, 256, 1, 1)
        #z = z.view(-1, 64, 2, 2)
        #return self.decoder(z)
    
    def decode(self, z):
        z = self.fc_z(z)
        #print('decoder input shape:', z.shape)
        z = z.view(-1, 64, 2, 2)  # Adjust the output size here
        z = self.decoder(z)
        #z = z.view(-1, 1, 32, 32)  # Resize the output to [batch_size, 1, 32, 32]
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar
    
    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    # out_padding is used to ensure output size matches EXACTLY of conv2d;
    # it does not actually add zero-padding to output :)
    def _deconv(self, in_channels, out_channels, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=1, stride=2, output_padding=1 #out_padding
            ),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU()
        )
    
    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        #print('x-shape:', x.shape)
        #print('recon_x-shape:', recon_x.shape)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        #recon_loss = F.mse_loss(recon_x, x)
        
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size
    
    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)
    
    def load_model(self, file_path):
        utils.restore(self, file_path)
    
    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)


class DFCVAE(BetaVAE):

    def __init__(self, latent_size=100, beta=1):
        super(DFCVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.e1 = self._conv(3, 32)
        self.e2 = self._conv(32, 64)
        self.e3 = self._conv(64, 128)
        self.e4 = self._conv(128, 256)
        self.fc_mu = nn.Linear(4096, latent_size)
        self.fc_var = nn.Linear(4096, latent_size)

        # decoder
        self.d1 = self._upconv(256, 128)
        self.d2 = self._upconv(128, 64)
        self.d3 = self._upconv(64, 32)
        self.d4 = self._upconv(32, 3)
        self.fc_z = nn.Linear(latent_size, 4096)

    def encode(self, x):
        x = F.leaky_relu(self.e1(x))
        x = F.leaky_relu(self.e2(x))
        x = F.leaky_relu(self.e3(x))
        x = F.leaky_relu(self.e4(x))
        x = x.view(-1, 4096)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 256, 4, 4)
        z = F.leaky_relu(self.d1(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d2(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d3(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d4(F.interpolate(z, scale_factor=2)))
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(out_channels),
        )

