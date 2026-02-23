#%%
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
import numpy as np

#%%
algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=2).initialize()

def decv_img(img_orig, kernel):
    data_1 = np.squeeze(img_orig[0, ...])
    data_2 = np.squeeze(img_orig[1, ...])
    kernel_1 = np.squeeze(kernel[0, ...])
    kernel_2 = np.squeeze(kernel[1, ...])

    if data_1.ndim != 2 or data_2.ndim != 2:
        raise ValueError(f"Cada canal de imagen debe ser 2D. Recibido: {data_1.shape} y {data_2.shape}")
    if kernel_1.ndim != 2 or kernel_2.ndim != 2:
        raise ValueError(f"Cada kernel debe ser 2D. Recibido: {kernel_1.shape} y {kernel_2.shape}")

    res_1 = algo.run(fd_data.Acquisition(data=data_1, kernel=kernel_1), niter=30).data
    res_2 = algo.run(fd_data.Acquisition(data=data_2, kernel=kernel_2), niter=30).data
    res_otro = np.stack([res_1, res_2], axis=-1)

    return res_otro

