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
def count_cells(img_org, return_map=False):


    """
    if img_org is None:
        raise ValueError("img_org no puede ser None")

    img_np = np.asarray(img_org)


    
    #Cosas de formato de la imagen -> resolver dudas 
    if img_np.ndim == 2:
        img_rgb = np.stack([img_np, img_np, img_np], axis=-1)
    elif img_np.ndim == 3 and img_np.shape[2] >= 2:
        ch1 = img_np[:, :, 0]
        ch4 = img_np[:, :, 1]
        img_rgb = np.stack([ch1, ch4, ch1], axis=-1)
    elif img_np.ndim == 3 and img_np.shape[2] == 1:
        img_rgb = np.repeat(img_np, 3, axis=2)
    else:
        raise ValueError(f"Formato de imagen no soportado: {img_np.shape}")

    img_rgb = img_rgb.astype(np.float32)
    max_val = float(img_rgb.max()) if img_rgb.size else 0.0
    if max_val > 1.0:
        scale = 65535.0 if max_val > 255.0 else 255.0
        img_rgb = img_rgb / scale

    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    """

    #PROCESADO IMG COMO EN TFM
    #Se asume que img está cargada
    if img_org is None:
        raise ValueError("img_org no puede ser None")

    #Convertir a numpy si es PIL, si no se asume que ya es numpy
    if hasattr(img_org, 'convert'):
        img_rgb = img_org.convert("RGB")
        image_np = np.array(img_rgb).transpose(2, 0, 1)
    else:
        image_np = np.array(img_org).transpose(2, 0, 1)

    #Se pasa a tensor
    img_tensor = torch.FloatTensor(image_np).unsqueeze(0)



    #Mascara de parásitos 
    with torch.no_grad():
        mask_p = model_p(img_tensor).squeeze().detach().cpu().numpy()

    #Mascara de núcleos 
    with torch.no_grad():
        mask_c = model_c(img_tensor).squeeze().detach().cpu().numpy()

    #Como las otras funciones esperan una salida de 3 mapas, se devuelve esto vacio por ahora
    mask_i = np.zeros_like(mask_c, dtype=np.float32)

    resul_predict = np.zeros((3,), dtype=float)
    resul_predict[0] = float(mask_p.sum())
    resul_predict[1] = float(mask_c.sum())
    resul_predict[2] = 0.0

    if return_map:
        return [mask_p, mask_c, mask_i], resul_predict

    return resul_predict

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

