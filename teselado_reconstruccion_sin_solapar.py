#Aqui se pondrá el código del teselado y reconstrucción de las imágenes que sean más grandes

#Se teselará en imágenes de 1024x1024

import numpy as np

def tile_image(image, tile_shape=(1024, 1024), pad_mode='constant'):
    """
    Divide una imagen de cualquier tamaño en teselas sin solapamiento y sin pérdida.

    Args:
        image (np.ndarray): Imagen de entrada en formato Numpy (H, W) o (H, W, C).
        tile_shape (tuple): Dimensión de la tesela (alto, ancho). Por defecto (1024, 1024).
        pad_mode (str): Modo de padding para bordes ('constant' pone 0s).

    Returns:
        tiles (list): Lista de arrays numpy con cada tesela.
        metadata (dict): Diccionario necesario para reconstruir la imagen original.
    """
    orig_h, orig_w = image.shape[:2]
    tile_h, tile_w = tile_shape

    # 1. Calcular el padding necesario para que sea múltiplo exacto del tile_shape
    pad_h = (tile_h - (orig_h % tile_h)) % tile_h
    pad_w = (tile_w - (orig_w % tile_w)) % tile_w

    # Configurar el padding según si es imagen en escala de grises o RGB
    if image.ndim == 2:
        pad_width = ((0, pad_h), (0, pad_w))
    else:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))

    # Aplicar el relleno (padding)
    padded_image = np.pad(image, pad_width, mode=pad_mode)
    pad_h_total, pad_w_total = padded_image.shape[:2]

    # 2. Extraer las teselas en orden
    tiles = []
    n_tiles_y = pad_h_total // tile_h
    n_tiles_x = pad_w_total // tile_w

    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            y_start = i * tile_h
            y_end = y_start + tile_h
            x_start = j * tile_w
            x_end = x_start + tile_w

            tile = padded_image[y_start:y_end, x_start:x_end]
            tiles.append(tile)

    # 3. Guardar metadatos esenciales para la reconstrucción
    metadata = {
        'orig_shape': image.shape,
        'padded_shape': padded_image.shape,
        'tile_shape': tile_shape,
        'grid_shape': (n_tiles_y, n_tiles_x)
    }

    return tiles, metadata


def reconstruct_image(tiles, metadata):
    """
    Reconstruye la imagen (o mapa de densidad) original a partir de sus teselas.

    Args:
        tiles (list): Lista de teselas (o salidas de la U-Net).
        metadata (dict): Diccionario retornado previa y automáticamente por `tile_image`.

    Returns:
        reconstructed (np.ndarray): Imagen reconstruida idéntica al tamaño original.
    """
    orig_shape = metadata['orig_shape']
    padded_shape = metadata['padded_shape']
    tile_h, tile_w = metadata['tile_shape']
    n_tiles_y, n_tiles_x = metadata['grid_shape']

    # Si las teselas son de 1 canal (mapas de densidad), ajustamos la forma del canvas
    sample_tile = tiles[0]
    if sample_tile.ndim == 2 and len(padded_shape) == 3:
        # Padded shape ajustado a 2D para mapas de densidad
        padded_canvas_shape = (padded_shape[0], padded_shape[1])
    else:
        padded_canvas_shape = padded_shape

    # 1. Crear lienzo vacío con las dimensiones padded
    reconstructed_padded = np.zeros(padded_canvas_shape, dtype=sample_tile.dtype)

    # 2. Colocar cada tesela en su posición correspondiente
    idx = 0
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            y_start = i * tile_h
            y_end = y_start + tile_h
            x_start = j * tile_w
            x_end = x_start + tile_w

            reconstructed_padded[y_start:y_end, x_start:x_end] = tiles[idx]
            idx += 1

    # 3. Recortar exactamente el padding original sin pérdida de píxeles
    orig_h, orig_w = orig_shape[:2]
    if len(padded_canvas_shape) == 2:
        reconstructed = reconstructed_padded[:orig_h, :orig_w]
    else:
        reconstructed = reconstructed_padded[:orig_h, :orig_w, :]

    return reconstructed