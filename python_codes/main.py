import os
import h5py
import random
import numpy as np
import pandas as pd

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR

import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None, cmap='viridis', vmin=-10, vmax=2):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    img = ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    return img

def plot_waveform(waveform, sample_rate=16000, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

def plot_and_save_spectrogram(filename, specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    img = ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.savefig(filename)
    plt.close()

def plot_and_save_waveform(filename, waveform, sample_rate=16000, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    plt.savefig(filename)
    plt.close()

def waveform_to_spectrogram(waveform, n_fft, win_length, hop_length, n_mels):
    # Convert to Mel-Spectrogram
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )(waveform)

    return spectrogram

def spectrogram_to_waveform(spectrogram, n_fft=1024, win_length=None, hop_length=512, n_mels=128):
    waveform = T.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )(spectrogram)
    
    return waveform


class MaestroDatasetHDF5(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            # Assuming that keys in 'orig' and 'fast' datasets correspond to each other
            self.keys = list(hdf5_file['orig'].keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            # Load the corresponding spectrogram from both 'orig' and 'fast'
            spectrogram_orig = hdf5_file['orig'][self.keys[idx]][()]
            spectrogram_fast = hdf5_file['fast'][self.keys[idx]][()]

        # Convert numpy arrays to torch tensors
        spectrogram_orig = torch.from_numpy(spectrogram_orig).float()
        spectrogram_fast = torch.from_numpy(spectrogram_fast).float()

        return spectrogram_orig, spectrogram_fast

class DASH(nn.Module):
    def __init__(self, num_channels, num_freq_bins):
        super(DASH, self).__init__()
        self.num_channels = num_channels
        self.num_freq_bins = num_freq_bins
        
        # Convolutional layers with He initialization and Batch Normalization
        self.conv1 = nn.Conv2d(num_channels, 4, kernel_size=(3, 25), padding=(1, 12))
        self.conv2 = nn.Conv2d(4, num_channels, kernel_size=(3, 25), padding=(1, 12))
        
        
    def forward(self, x):        
        # Apply convolutional layers with ReLU activation and Instance Normalization
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DATM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DATM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Assuming input_size is the number of frequency bins and the LSTM input is the time_steps
        self.lstm = nn.LSTM(1, hidden_size, 
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Initialize hidden and cell states
        batch_size, channels, freq_bins, time_steps = x.shape
        x = x.reshape(batch_size * channels * freq_bins, time_steps, 1)
        # h0 = torch.zeros(self.num_layers, batch_size * channels * freq_bins, self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, batch_size * channels * freq_bins, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x)
        
        # Decode the hidden state of the last time step
        out = self.linear(out)  # Now out has shape: (batch_size * time_steps, input_size)
        
        # Reshape to get separate channels and freq_bins again
        out = out.view(batch_size, channels, freq_bins, time_steps)
        return out

def spectral_convergence_loss(predicted, target):
    return torch.norm(torch.abs(predicted) - torch.abs(target), p='fro') / torch.norm(torch.abs(target), p='fro')

def log_spectral_distance_loss(predicted, target):
    log_diff = torch.log(torch.abs(predicted) + 1e-8) - torch.log(torch.abs(target) + 1e-8)
    return torch.mean(torch.sqrt(torch.mean(log_diff ** 2, dim=-1)))

def relative_error_loss(predicted, target, epsilon=1e-2):
    return torch.norm(torch.abs((predicted - target) / (target + epsilon)), p='fro')

def MLELoss(predicted, target, epsilon=1e-10):
    return nn.functional.mse_loss(torch.log10(torch.abs(predicted)+epsilon), torch.log10(torch.abs(target)+epsilon))

class DecibelLoss(nn.Module):
    def __init__(self):
        super(DecibelLoss, self).__init__()

    def forward(self, output, target):
        epsilon = 1e-10
        output_amplitude = torch.abs(output) + epsilon
        target_amplitude = torch.abs(target) + epsilon

        # Convert amplitude to decibel
        output_db = 20 * torch.log10(output_amplitude)
        target_db = 20 * torch.log10(target_amplitude)

        # Calculate MSE in the dB scale
        loss = nn.functional.mse_loss(output_db, target_db)

        return loss

def save_output(epoch, model, data_loader, device, criterion):
    with torch.no_grad():
        model.eval()
        input_, label = next(iter(data_loader))
        input_, label = input_.to(device), label.to(device)
        input_ = torch.log10(input_ + 1e-10)
        label = torch.log10(label + 1e-10)
        output = model(input_)
        output = F.interpolate(output, size=(label.size(-2), label.size(-1)), mode='bilinear', align_corners=False)
        print(criterion(output, label).item())
        # Convert and save outputs
        input_ = input_[0].cpu()
        output = output[0].cpu()
        label = label[0].cpu()
        
        fig, axs = plt.subplots(3, 1, figsize=(15, 20))
        img0 = plot_spectrogram(input_[0], title="Input Spectrogram", ax=axs[0])
        img1 = plot_spectrogram(output[0], title="Output Spectrogram", ax=axs[1])
        img2 = plot_spectrogram(label[0], title="Answer Spectrogram", ax=axs[2])

        # Add colorbars
        fig.colorbar(img0, ax=axs[0], format='%+2.0f dB')
        fig.colorbar(img1, ax=axs[1], format='%+2.0f dB')
        fig.colorbar(img2, ax=axs[2], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(f"output/spectrogram_{epoch+1}.png")
        plt.close(fig)
        
        input_ = 10 ** input_ + 1e-10
        output = 10 ** output + 1e-10
        label = 10 ** label + 1e-10
        
        waveform_input = spectrogram_to_waveform(input_)
        waveform_output = spectrogram_to_waveform(output)
        waveform_label = spectrogram_to_waveform(label)
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        plot_waveform(waveform_input, sample_rate=16000, title="Input Waveform", ax=axs[0])
        plot_waveform(waveform_output, sample_rate=16000, title="Output Waveform", ax=axs[1])
        plot_waveform(waveform_label, sample_rate=16000, title="Answer Waveform", ax=axs[2])
        plt.tight_layout()
        plt.savefig(f"output/waveform_{epoch+1}.png")
        plt.close(fig)
        
        torchaudio.save(f"output/input_{epoch+1}.wav", waveform_input, sample_rate=16000)
        torchaudio.save(f"output/output_{epoch+1}.wav", waveform_output, sample_rate=16000)
        torchaudio.save(f"output/answer_{epoch+1}.wav", waveform_label, sample_rate=16000)


def main():
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.01
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.cuda.empty_cache()
    
    train_dataset = MaestroDatasetHDF5('maestro_spectrograms.hdf5')
    test_dataset = MaestroDatasetHDF5('maestro_spectrograms_test.hdf5')
    print("Training data size:", len(train_dataset))
    print("Test data size:", len(test_dataset))

    # DataLoader for efficient batch loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=20)

    freq_bins = 513

    # model = DASH(2, freq_bins, 1.5)
    # model = model.to(device)
    
    hidden_size = 2  # You can tune this
    num_layers = 1  # Number of LSTM layers
    
    # Initialize the LSTM model
    model = DATM(2 * freq_bins, hidden_size, num_layers)
    # model = DASH(2, freq_bins)
    model = model.to(device)

    # Loss and optimizer
    
    os.makedirs('output', exist_ok=True)
    criterion1 = nn.MSELoss()
    decibel_loss = DecibelLoss().to(device)
    criterion2 = spectral_convergence_loss
    criterion3 = relative_error_loss

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1**epoch)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Loss: 0.000000000')
        for (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = torch.log10(inputs + 1e-10)
            labels = torch.log10(labels + 1e-10)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=(labels.size(-2), labels.size(-1)), mode='bilinear')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description("Loss: {:11.9f}".format(loss.item()))
        # current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss / len(train_loader)}')
        running_loss = 0.0
        pbar = tqdm(test_loader, desc=f'Loss: 0.000000000')
        for (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = torch.log10(inputs + 1e-10)
            labels = torch.log10(labels + 1e-10)
            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=(labels.size(-2), labels.size(-1)), mode='bilinear')
            loss = criterion(outputs, labels)
            pbar.set_description("Loss: {:11.9f}".format(loss.item()))
        print(f'Test - Loss: {running_loss / len(test_loader)}')
        # scheduler.step()
        save_output(epoch, model, test_loader, device, criterion)
        torch.save(model.state_dict(), "DATH.pt")

if __name__ == '__main__':
    main()
