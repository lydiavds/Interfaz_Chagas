#%%
from PIL import Image, ImageSequence
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from collections import defaultdict



def read_tiff(path, size_img=None):
    """
    path - Path to the multipage-tiff file
    """
    images = []

    with Image.open(path) as img:
        for im in ImageSequence.Iterator(img):
            if size_img is None:
                images.append(np.array(im))
            else:
                images.append(np.array(im.resize((size_img, size_img))))

    img_temp = np.stack(images, axis=-1)

    return img_temp


def convert_uint8(img):
    if img.dtype == np.uint8:
        return img.copy()

    imin = img.min(axis=(0, 1))
    imax = img.max(axis=(0, 1))

    # Evitar división por cero cuando imax == imin (imagen uniforme)
    denom = imax - imin
    denom = np.where(denom == 0, 1, denom)

    return (255. * (img - imin) / denom).astype(np.uint8)


def convert_gray2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def merge_img_map(img_org, maps):
    merge_img = img_org.copy()
    size_img = merge_img.shape[:2]

    tmp_cellmap = cv2.resize(maps[1] * 22, size_img, interpolation=cv2.INTER_NEAREST)
    mask_tmp = tmp_cellmap > 150
    merge_img[mask_tmp] = [255, 255, 0]

    tmp_cellmap = cv2.resize(maps[2] * 22, size_img, interpolation=cv2.INTER_NEAREST)
    mask_tmp = tmp_cellmap > 150
    merge_img[mask_tmp] = [255, 0, 0]

    merge_img[:, :, 2] = merge_img[:, :, 2] + cv2.resize(maps[0] * 2, size_img, interpolation=cv2.INTER_NEAREST)

    return merge_img


def add_padding(img_org, target_size=(1024,1024)):
    """
    Añade padding a la imagen original para que tenga el tamaño objetivo minimo de 1024x1024.
    """
    h, w = img_org.shape[:2]
    target_h, target_w = target_size

    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if img_org.ndim == 2:
        padded_img = np.pad(img_org, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    else:
        padded_img = np.pad(img_org, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    return padded_img





#A PARTIR DE AQUI SE VAN A AÑADIR LAS FUNCIONES DE CONTABILIZACIÓN DE PARÁSITOS ETC QUE SE TIENEN DEL CÓDIGO DEL TFM

#PARÁSITOS TFM

def contar_parasitos_dbscan(img_original, density_map, threshold_rel=0.35, eps=2, min_samples=2):
    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min())
    threshold = threshold_rel * np.max(density_map)
    y_coords, x_coords = np.where(density_map > threshold)
    coords = np.column_stack((x_coords, y_coords))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}
    centros = [np.mean(coords[labels == label], axis=0) for label in unique_labels]
    return len(centros), centros


#PARÁSITOS TFG
def contar_parasitos_tfg(density_map):
    threshold = max(0.1, density_map.max() * 0.3) #threshold adaptativo
    parasitos_detectados = peak_local_max(density_map, min_distance=5, threshold_abs=threshold)

    return len(parasitos_detectados), parasitos_detectados #Con parasitos_detectados siendo pd[0]=y, pd[1]=x


#NÚCLEOS CELULARES TFM

def detectar_parasitos_maxima_parametrizable(pred, sigma_suavizado=3, min_distance=30, percentil_fondo=80):
    smoothed = gaussian_filter(pred, sigma=sigma_suavizado)
    coordinates = peak_local_max(smoothed, min_distance=min_distance)

    background_threshold = np.percentile(smoothed, percentil_fondo)

    filtered_coords = []
    for y, x in coordinates:
        val_centro = smoothed[y, x]
        vecindario = smoothed[max(y-3,0):y+4, max(x-3,0):x+4]
        val_medio_around = np.mean(np.delete(vecindario.flatten(), vecindario.size // 2))
        if val_centro > background_threshold and val_centro - val_medio_around > 0.05:
            filtered_coords.append((y, x))

    return len(filtered_coords), filtered_coords



#NÚCLEOS CELULARES TFG

#Igual que TFM pero radio del vecindario depende del sigma y tiene threshold absoluto para descartar picos muy bajos
def detectar_nucleos_maxima_parametrizable_tfg(pred, sigma_suavizado=2, min_distance=40, percentil_fondo=30, diferencia_minima=0.001):
    # 1. Suavizado gaussiano
    smoothed = gaussian_filter(pred, sigma=sigma_suavizado)

    # 2. Búsqueda de picos SIN descartar los márgenes de las teselas
    coordinates = peak_local_max(
        smoothed,
        threshold_abs=0.5,
        min_distance=min_distance,
        exclude_border=False  #No ignora los bordes
    )

    background_threshold = np.percentile(smoothed, percentil_fondo)

    # 3. Adaptar el radio del vecindario según el sigma (ej. sigma 15 -> radio 7)
    radio_vecindario = max(3, int(sigma_suavizado / 2))

    filtered_coords = []
    for y, x in coordinates:
        val_centro = smoothed[y, x]

        y_min, y_max = max(y - radio_vecindario, 0), min(y + radio_vecindario + 1, smoothed.shape[0])
        x_min, x_max = max(x - radio_vecindario, 0), min(x + radio_vecindario + 1, smoothed.shape[1])

        vecindario = smoothed[y_min:y_max, x_min:x_max]

        if vecindario.size > 1:
            val_medio_around = np.mean(np.delete(vecindario.flatten(), vecindario.size // 2))
        else:
            val_medio_around = 0

        # Validación del pico
        if val_centro > background_threshold and (val_centro - val_medio_around) >= diferencia_minima:
            filtered_coords.append((y, x))

    return len(filtered_coords), filtered_coords


#ASIGNACIÓN PARÁSITO-NÚCLEO CELULAR

#Esta es la asignación que se acaba usando, ya que proporiciona mejores resultados (mejores que usando solo la distancia minima)
def asociar_por_clustering(coords_ch1, coords_ch4, draw, eps=25, min_samples=1):
    asociaciones = defaultdict(list)
    if len(coords_ch1) == 0 or len(coords_ch4) == 0:
        return asociaciones
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_ch4)
    labels = clustering.labels_
    for label in set(labels):
        if label == -1:
            continue  # ruido
        cluster = coords_ch4[labels == label]
        centroide = np.mean(cluster, axis=0)
        distancias = np.linalg.norm(coords_ch1 - centroide, axis=1)
        idx_min = np.argmin(distancias)
        for par in cluster:
            #Solo se asigna el parásito al núcleo más cercano si está dentro de un cierto umbral de distancia
            if distancias[idx_min] < 200:  # Umbral de distancia grande pero lo suficientemente pequeño para evitar asociaciones erróneas
                asociaciones[idx_min].append(par)
                x1, y1 = par
                x2, y2 = coords_ch1[idx_min]
                draw.line((x1, y1, x2, y2), fill="cyan", width=1)
    return asociaciones


#Calcula celulas sanas, media de parásitos por célula y ahora se ha añadido que calcule el porcentaje de células infectadas
def resumen(asociaciones, total_celulas):
    infectadas = sum(1 for v in asociaciones.values() if len(v) > 0)
    sanas = total_celulas - infectadas
    total_parasitos = sum(len(v) for v in asociaciones.values())
    media_parasitos = total_parasitos / infectadas if infectadas > 0 else 0
    #return infectadas, sanas, media_parasitos

    #Se ha modificado esta funcion del TFM para que además calcule el porcentaje de células infectadas
    pct_infectadas = (infectadas / total_celulas * 100) if total_celulas > 0 else 0
    return infectadas, sanas, media_parasitos, pct_infectadas


