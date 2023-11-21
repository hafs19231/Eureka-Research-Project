from main import *

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None, cmap='viridis'):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    img = ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
    return img

def save_to_text_file(data, filename):
    """
    Save the 2D numpy array data to a text file in the specified format.
    Each row in the text file will contain: x-coordinate, y-coordinate, value
    """
    with open(filename, 'w') as file:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                file.write(f"{i} {j} {data[i, j]}\n")

def main(input_file="infer.wav"):
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    freq_bins = 513
    
    model = DATM(2 * freq_bins, 2, 1)
    # model = DASH(2, freq_bins)
    model = model.to(device)
    model.load_state_dict(torch.load("DATH.pt"))
    model.eval()
    
    os.makedirs('infer', exist_ok=True)
    criterion1 = nn.MSELoss()
    
    waveform_orig, sr = torchaudio.load(input_file, normalize=True)
    print(sr)
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    plot_waveform(waveform_orig, sample_rate=16000, title="Input Waveform", ax=axs)
    plt.savefig(f"infer/wave.png")
    plt.close(fig)
    print(waveform_orig.shape)
    
    input_ = waveform_to_spectrogram(waveform_orig, n_fft=1024, win_length=None, hop_length=256, n_mels=128)
    input_ = torch.log10(input_ + 1e-10).unsqueeze(0)
    input_, label = next(iter(DataLoader(MaestroDatasetHDF5('maestro_spectrograms_test.hdf5'), batch_size=1, shuffle=False, pin_memory=True, num_workers=20)))
    input_ = input_.to(device)
    label = label.to(device)
    output = model(input_)
    # input_ = F.interpolate(input_, size=(output.size(-2), t), mode='bilinear')
    t = round(output.size(-1) / 1.5)
    output = F.interpolate(output, size=(output.size(-2), t), mode='bilinear')
    refer = F.interpolate(input_, size=(output.size(-2), t), mode='bilinear')
    
    input_ = input_[0].cpu().detach()
    output = output[0].cpu().detach()
    refer = refer[0].cpu().detach()
    label = label[0].cpu().detach()
    save_to_text_file(input_[0].numpy(), "input.dat")
    save_to_text_file(output[0].numpy(), "output.dat")
    save_to_text_file(refer[0].numpy(), "refer.dat")
    save_to_text_file(label[0].numpy(), "label.dat")
    
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    img0 = plot_spectrogram(input_[0], title="Input Spectrogram", ax=axs)
    fig.colorbar(img0, ax=axs, format='%+2.0f dB')
    plt.savefig(f"infer/spectrogram_input.png")
    plt.close(fig)
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    img1 = plot_spectrogram(output[0], title="Output Spectrogram", ax=axs)
    fig.colorbar(img1, ax=axs, format='%+2.0f dB')
    plt.savefig(f"infer/spectrogram_output.png")
    plt.close(fig)
    
    input_ = 10 ** input_ + 1e-10
    output = 10 ** output + 1e-10
    refer = 10 ** refer + 1e-10
    
    waveform_input = spectrogram_to_waveform(input_ * 10)
    waveform_output = spectrogram_to_waveform(output * 10)
    waveform_refer = spectrogram_to_waveform(refer * 10)
    print(waveform_input.shape)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plot_waveform(waveform_input, sample_rate=16000, title="Input Waveform", ax=axs[0])
    plot_waveform(waveform_output, sample_rate=16000, title="Output Waveform", ax=axs[1])
    plot_waveform(waveform_refer, sample_rate=16000, title="Reference Waveform", ax=axs[2])
    plt.tight_layout()
    plt.savefig(f"infer/waveform.png")
    plt.close(fig)
    
    torchaudio.save(f"infer/input.wav", waveform_input, sample_rate=16000)
    torchaudio.save(f"infer/output.wav", waveform_output, sample_rate=16000)
    torchaudio.save(f"infer/refer.wav", waveform_refer, sample_rate=16000)
    
if __name__ == '__main__':
    main("opt.wav")