import numpy as np
import torch
import cv2
from pytorch_models import UNet

#Basicamente, ignorar un poco lo que ya esta aqui, y obretodo usar el que esta en el codigo de la fase3

# Configurar dispositivo (GPU si está disponible, sino CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

#%%


"""
COUNT CELLS DE FICRON


def count_cells(img_org, return_map=False):

    size_img = np.array(img_org.shape[:2])

    size_img -= (size_img % 32)
    img_org = cv2.resize(img_org, (size_img[1], size_img[0]), interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(img_org, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
    img1 = img1 / 65535.
    resul_predict = np.zeros((3,))

    #Prediccion parasitos
    with torch.no_grad():
        img_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device)
        mask_p = model_p(img_tensor).squeeze().cpu().numpy()
    
    mask_p = mask_p / 3400.
    resul_predict[0] = mask_p.sum()
    mask_p = cv2.resize(mask_p, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA) * 4
    mask_p = mask_p * 3400.

    img1 = cv2.resize(img1, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)

    #Prediccion macrofagos
    with torch.no_grad():
        img_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device)
        mask_c = model_c(img_tensor).squeeze().cpu().numpy()
    
    resul_predict[1] = mask_c.sum() / 1031.

    img_aux = np.concatenate((np.expand_dims(img1[:, :, 0], axis=-1), np.expand_dims(mask_p, axis=-1),
                              np.expand_dims(mask_c, axis=-1)), axis=-1)

    #Prediccion infectados con concatenacion de parasitos y macrofagos
    with torch.no_grad():
        img_tensor = torch.from_numpy(img_aux).permute(2, 0, 1).unsqueeze(0).float().to(device)
        mask_i = model_i(img_tensor).squeeze().cpu().numpy()
    
    thresh_inf = 0.8
    mask_i[mask_i >= thresh_inf] = 1
    mask_i[mask_i < (1 - thresh_inf)] = 0
    mask_i = mask_i * mask_c

    resul_predict[2] = mask_i.sum() / 1031.

    if return_map:
        return [mask_p, mask_c, mask_i], resul_predict
    else:
        return resul_predict
"""



#Solo se va a procesar una img a la vez, como en ficron
def count_cells(img_org, channel=None, return_map=False):
    """
    Procesa la imagen y realiza predicciones según el canal especificado.
    
    Args:
        img_org: Imagen de entrada
        channel: Canal a procesar (0 para CH1/núcleos, 2 para CH3/parásitos, None para ambos) (Aunque no deberia hacerse nunca lo de ambos)
        return_map: Si True, devuelve los mapas de densidad
    
    Returns:
        Si return_map=True: ([mask_p, mask_c, mask_i], resul_predict)
        Si return_map=False: resul_predict
    """
    #PROCESADO IMG COMO EN TFM
    #Se asume que img está cargada
    if img_org is None:
        raise ValueError("img_org no puede ser None")

    #Convertir a numpy y a rgb si es posible
    #Transpose para que el canal esté en la primera dimensión (C, H, W) como espera PyTorch
    if hasattr(img_org, 'convert'):
        img_rgb = img_org.convert("RGB")
        image_np = np.array(img_rgb).transpose(2, 0, 1)
    else:
        image_np = np.array(img_org).transpose(2, 0, 1)

    #Se pasa a tensor
    img_tensor = torch.FloatTensor(image_np).unsqueeze(0)

    # Inicializar máscaras vacías
    mask_p = None
    mask_c = None

    # Validar canal si se especifica (parásitos o núcleos (CH1, CH3))
    if channel is not None and channel not in [0, 2]:
        raise ValueError("channel debe ser 0, 2 o None")

    # Predicción según el canal (parásitos o núcleos)
    if channel is None or channel == 2:
        # Canal CH3 (parásitos)
        with torch.no_grad():
            mask_p = model_p(img_tensor).squeeze().detach().cpu().numpy()
    
    if channel is None or channel == 0:
        # Canal CH1 (núcleos)
        with torch.no_grad():
            mask_c = model_c(img_tensor).squeeze().detach().cpu().numpy()

    # Crear máscaras vacías para los canales no procesados
    if mask_p is None:
        h, w = image_np.shape[1], image_np.shape[2]
        mask_p = np.zeros((h, w), dtype=np.float32)
    if mask_c is None:
        h, w = image_np.shape[1], image_np.shape[2]
        mask_c = np.zeros((h, w), dtype=np.float32)

    # Máscara de infectados vacía porque se esperan 3 valores de mascara
    mask_i = np.zeros_like(mask_c, dtype=np.float32)

    resul_predict = np.zeros((3,), dtype=float)
    resul_predict[0] = float(mask_p.sum())
    resul_predict[1] = float(mask_c.sum())
    resul_predict[2] = 0.0

    if return_map:
        return [mask_p, mask_c, mask_i], resul_predict

    return resul_predict

#Para cargar el modelo pytorch
def load_model(path, model_class):
    """
    Carga un modelo PyTorch desde un archivo .pth
    
    Args:
        path: Ruta al archivo del modelo .pth
        model_class: Clase del modelo PyTorch
    
    Returns:
        Modelo cargado en modo evaluación
    """
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modelo cargado: {path}")
    return model


#La fase 3 no tiene modelo propio, si no que asigna los parasitos encontrados en la fase 1 con las celulas de la fase 2

# Rutas de los modelos PyTorch
path_model_p = 'Models/fase1_modelo8.pth' 
path_model_cel = 'Models/fase2_modelo2.pth'


# Cargar modelos PyTorch
# NOTA: Ajusta las clases de modelos según la arquitectura de cada uno
model_p = load_model(path_model_p, model_class=UNet)
model_c = load_model(path_model_cel, model_class=UNet)

