import numpy as np

def rotasi(image, sudut):
    """
    Rotasi gambar menggunakan numpy.
    
    Parameters:
    - image: numpy array, citra input (grayscale atau RGB).
    - sudut: float, sudut rotasi dalam derajat.
    
    Returns:
    - rotated_image: numpy array, citra hasil rotasi.
    """
    
    # Konversi sudut ke radian
    sudut_rad = np.deg2rad(sudut)
    
    # Ukuran gambar
    h, w = image.shape[:2]
    
    # Pusat rotasi
    cx, cy = w // 2, h // 2
    
    # Matriks rotasi (2x2)
    matriks_rotasi = np.array([
        [np.cos(sudut_rad), -np.sin(sudut_rad)],
        [np.sin(sudut_rad), np.cos(sudut_rad)]
    ])
    
    # Siapkan matriks nol untuk citra hasil rotasi
    rotated_image = np.zeros_like(image)
    
    for y in range(h):
        for x in range(w):
            # Hitung koordinat relatif terhadap pusat
            rel_x, rel_y = x - cx, y - cy
            
            # Rotasi koordinat
            koord_new = matriks_rotasi @ np.array([rel_x, rel_y])
            new_x, new_y = koord_new + np.array([cx, cy])
            
            # Boundary check
            if 0 <= new_x < w and 0 <= new_y < h:
                rotated_image[int(new_y), int(new_x)] = image[y, x]
    
    return rotated_image



def flip_horizontal(image):
    """
    Membalik gambar secara horizontal (kiri ke kanan).
    
    Parameters:
    - image: numpy array, citra input (grayscale atau RGB).
    
    Returns:
    - flipped_image: numpy array, citra hasil flipping horizontal.
    """
    flipped_image = image[:, ::-1]  # Membalik array sepanjang sumbu x
    return flipped_image


def flip_vertical(image):
    """
    Membalik gambar secara vertikal (atas ke bawah).
    
    Parameters:
    - image: numpy array, citra input (grayscale atau RGB).
    
    Returns:
    - flipped_image: numpy array, citra hasil flipping vertikal.
    """
    flipped_image = image[::-1, :]  # Membalik array sepanjang sumbu y
    return flipped_image


def flip_horizontal_vertical(image):
    """
    Membalik gambar secara horizontal dan vertikal (180 derajat).
    
    Parameters:
    - image: numpy array, citra input (grayscale atau RGB).
    
    Returns:
    - flipped_image: numpy array, citra hasil flipping horizontal dan vertikal.
    """
    flipped_image = image[::-1, ::-1]  # Membalik array sepanjang kedua sumbu
    return flipped_image
