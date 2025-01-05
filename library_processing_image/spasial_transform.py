import numpy as np
from math_method import bicubic_interpolation, gaussian_filter

def zoom_in(image, zoom_factor):
    """
    Melakukan zoom in pada gambar dengan crop dan bicubic interpolation.
    
    Parameters:
    - image: numpy array, citra input (grayscale atau RGB).
    - zoom_factor: float, faktor zoom (> 1 untuk zoom in).
    
    Returns:
    - zoomed_image: numpy array, citra hasil zoom in.
    """
    h, w = image.shape[:2]
    
    # Hitung ukuran crop berdasarkan zoom factor
    crop_h, crop_w = int(h / zoom_factor), int(w / zoom_factor)
    
    # Hitung koordinat untuk crop bagian tengah
    start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
    end_y, end_x = start_y + crop_h, start_x + crop_w
    
    # Crop bagian tengah citra
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    # Resize hasil crop ke ukuran asli menggunakan bicubic interpolation
    zoomed_image = bicubic_interpolation(cropped_image, h, w)
    
    return zoomed_image


def zoom_out(image, zoom_factor):
    """
    Melakukan zoom out pada gambar dengan bicubic interpolation dan Gaussian filtering.
    
    Parameters:
    - image: numpy array, citra input (grayscale atau RGB).
    - zoom_factor: float, faktor zoom (> 1 untuk zoom out).
    
    Returns:
    - zoomed_image: numpy array, citra hasil zoom out.
    """
    h, w = image.shape[:2]
    
    # Hitung ukuran baru
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    
    # Resize citra menggunakan bicubic interpolation
    zoomed_image = bicubic_interpolation(image, new_h, new_w)
    
    # Gaussian filtering untuk mengurangi noise dan artefak
    filtered_image = gaussian_filter(zoomed_image)
    
    # Tambahkan padding hitam jika ukuran lebih kecil dari ukuran asli
    padded_image = np.zeros_like(image)
    padded_image[:new_h, :new_w] = filtered_image
    
    return padded_image
