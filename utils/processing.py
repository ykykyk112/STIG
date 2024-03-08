import torch

def img2fft(img) :
    coord_matched_img = torch.fft.fftshift(img, dim = (-1, -2))
    fft = torch.fft.fft2(coord_matched_img)
    fft = torch.fft.fftshift(fft)
    return fft

def fft2polar(fft) :
    magnitude = torch.log1p(torch.abs(fft))
    phase = torch.angle(fft)
    return magnitude, phase

def normalize(tensor) :
    v_max, v_min = tensor.max(), tensor.min()
    tensor_normed = (tensor - v_min) / (v_max - v_min)
    tensor_normed = tensor_normed * 2. - 1.
    return tensor_normed, v_max, v_min

def image_normalize(image) :
    v_max, v_min = image.max(), image.min()
    return (image - v_min) / (v_max - v_min)

def inverse_normalize(tensor, v_max, v_min) :
    tensor_scaled = (tensor + 1.) / 2.
    tensor_scaled = (tensor_scaled * (v_max - v_min)) + v_min
    return tensor_scaled

def polar2img(magnitude, phase) :
    fft = torch.expm1(magnitude) * torch.exp(1j * phase)
    ifft = torch.fft.ifftshift(fft)
    ifft = torch.fft.ifft2(ifft)
    img_mag = torch.abs(ifft)
    #img_mag = ifft.real
    coord_matched_img = torch.fft.fftshift(img_mag, dim = (-1, -2))
    return coord_matched_img

def get_magnitude(img) :
    fft = torch.fft.fft2(img, dim = [-1, -2])
    fft = torch.fft.fftshift(fft, dim = [-1, -2])
    mag = torch.log1p(torch.abs(fft))
    v_min, v_max = torch.amin(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1), torch.amax(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1)
    mag_normed = (mag - v_min) / (v_max - v_min)
    return mag_normed

def get_averaged_magnitude(img) :
    fft = torch.fft.fft2(img)
    fft = torch.fft.fftshift(fft)
    mag = torch.log1p(torch.abs(fft))
    v_min, v_max = torch.amin(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1), torch.amax(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1)
    mag_normed = (mag - v_min) / (v_max - v_min)
    mag_normed = mag_normed * 2. - 1.
    #mag_mean = mag_normed.mean(0, keepdim = True)
    return mag_normed