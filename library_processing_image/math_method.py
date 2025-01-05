import numpy as np

def cubic_interpolate(p, x):
    """
    Melakukan interpolasi kubik pada empat titik p dengan posisi x (0 <= x <= 1).
    
    Parameters:
    - p: numpy array, empat nilai piksel di sekitar
    - x: float, posisi relatif di antara piksel (0 <= x <= 1)
    
    Returns:
    - interpolated_value: float, hasil interpolasi kubik
    """
    return (
        p[1]
        + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])))
    )



def bicubic_interpolation(image, new_h, new_w):
    """
    Melakukan interpolasi bicubic pada citra.
    
    Parameters:
    - image: numpy array, citra input (grayscale atau RGB)
    - new_h: int, tinggi baru hasil interpolasi
    - new_w: int, lebar baru hasil interpolasi
    
    Returns:
    - interpolated_image: numpy array, citra hasil interpolasi bicubic
    """
    h, w = image.shape[:2]
    if image.ndim == 3:  # Jika RGB, iterasi per channel
        interpolated_image = np.zeros((new_h, new_w, 3), dtype=image.dtype)
        for c in range(3):
            interpolated_image[..., c] = bicubic_interpolation(image[..., c], new_h, new_w)
        return interpolated_image

    # Buat citra hasil dengan ukuran (new_h, new_w)
    interpolated_image = np.zeros((new_h, new_w), dtype=image.dtype)

    # Skala faktor
    scale_x, scale_y = w / new_w, h / new_h

    for i in range(new_h):
        for j in range(new_w):
            x, y = j * scale_x, i * scale_y
            x_int, y_int = int(x), int(y)
            dx, dy = x - x_int, y - y_int

            # Ambil 16 piksel di sekitar (4x4)
            patch = image[max(0, y_int - 1):min(h, y_int + 3), max(0, x_int - 1):min(w, x_int + 3)]
            if patch.shape[0] < 4 or patch.shape[1] < 4:
                interpolated_image[i, j] = image[y_int, x_int]  # Jika di tepi, gunakan nearest neighbor
            else:
                interpolated_image[i, j] = cubic_interpolate(cubic_interpolate(patch.T, dx), dy)

    return interpolated_image


def gaussian_kernel(kernel_size, sigma):
    """
    Membuat kernel Gaussian 2D.
    
    Parameters:
    - kernel_size: int, ukuran kernel (harus ganjil)
    - sigma: float, standar deviasi Gaussian
    
    Returns:
    - kernel: numpy array 2D, kernel Gaussian
    """
    k = kernel_size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalisasi agar total bobot = 1
    return kernel

def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    Mengaplikasikan Gaussian filter pada citra menggunakan NumPy.
    
    Parameters:
    - image: numpy array, citra input (grayscale atau RGB)
    - kernel_size: int, ukuran kernel Gaussian (harus ganjil)
    - sigma: float, standar deviasi Gaussian
    
    Returns:
    - filtered_image: numpy array, citra hasil filtering
    """
    # Buat kernel Gaussian
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Ukuran citra
    h, w = image.shape[:2]
    
    # Padding citra untuk konvolusi
    pad_size = kernel_size // 2
    if image.ndim == 3:  # Jika RGB
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
        filtered_image = np.zeros_like(image)
        for c in range(3):  # Konvolusi per channel
            for i in range(h):
                for j in range(w):
                    region = padded_image[i:i+kernel_size, j:j+kernel_size, c]
                    filtered_image[i, j, c] = np.sum(region * kernel)
    else:  # Jika grayscale
        padded_image = np.pad(image, pad_size, mode='constant')
        filtered_image = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                region = padded_image[i:i+kernel_size, j:j+kernel_size]
                filtered_image[i, j] = np.sum(region * kernel)
    
    return filtered_image.astype(image.dtype)

def normalisasi_image(image):
    """
    Melakukan normalisasi citra RGB per channel.
    
    Parameters:
    - image: numpy array dengan shape (H, W, 3)
    
    Returns:
    - normalized_image: numpy array dengan shape (H, W, 3), hasil normalisasi
    """
    # Hitung mean dan std untuk setiap channel (R, G, B)
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    
    # Normalisasi tiap channel
    normalized_image = (image - mean) / std
    
    return normalized_image