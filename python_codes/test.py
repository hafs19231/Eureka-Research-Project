from main import *

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None, cmap='viridis'):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    img = ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
    return img

def main(input_file="infer.wav"):
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    train_dataset = MaestroDatasetHDF5('maestro_spectrograms.hdf5')
    test_dataset = MaestroDatasetHDF5('maestro_spectrograms_test.hdf5')
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=20)
    
    freq_bins = 513
    model = DATM(2 * freq_bins, 2, 1)
    # model = DASH(2, freq_bins)
    model = model.to(device)
    model.load_state_dict(torch.load("DATH.pt"))
    model.eval()
    
    criterion = nn.MSELoss()
    
    model.train()
    running_loss = 0.0
    pbar = tqdm(test_loader, desc=f'Loss: 0.000000000')
    for (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = torch.log10(inputs + 1e-10)
        labels = torch.log10(labels + 1e-10)
        outputs = model(inputs)
        outputs = F.interpolate(outputs, size=(labels.size(-2), labels.size(-1)), mode='bilinear')
        loss = criterion(10 * outputs, 10 * labels)
        running_loss += loss.item()
        pbar.set_description("Loss: {:11.9f}".format(loss.item()))
    # current_lr = scheduler.get_last_lr()[0]
    print(f'Total Loss: {running_loss / len(test_loader)}')
    
if __name__ == '__main__':
    main("opt.wav")