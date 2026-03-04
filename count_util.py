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





#A PARTIR DE AQUI SE VAN A AÑADIR LAS FUNCIONES DE CONTABILIZACIÓN DE PARÁSITOS ETC QUE SE TIENEN DEL CÓDIGO DEL TFM

#PARÁSITOS

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


#NÚCLEOS CELULARES

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


#ASIGNACIÓN PARÁSITO-NÚCLEO CELULAR

#Esta es la asignación que se acaba usando, ya que proporiciona mejores resultados (mejores que usando solo la distancia minima)
def asociar_por_clustering(coords_ch1, coords_ch4, draw, eps=25, min_samples=1):
    asociaciones = defaultdict(list)
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
