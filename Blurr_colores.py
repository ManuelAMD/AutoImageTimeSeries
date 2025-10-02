from PIL import Image
import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd
from pandas import DataFrame

# 1. Colores hex (añadimos '000000' al inicio, 'ffffff' al final)
hex_colors = [
    '000000', '730000', 'e60000', 'ffaa00', 'fcd37f',
    'ffff00', 'aaff55', '00ffff', '00aaff', '0000ff',
    '0000aa', 'ffffff'
]

# 2. Grayscale values asignados manualmente
grayscale_values = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 240, 255]

# Convert hex a RGB
def hex_to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
reference_colors = [hex_to_rgb(h) for h in hex_colors]

# 3. Procesar imagen
def procesar_imagen(path_entrada, path_gris_salida, path_rgb_salida= None):
    # Cargar imagen y convertir a RGB
    img = Image.open(path_entrada).convert("RGB")
    img_np = np.array(img).reshape(-1, 3)

    # Buscar color más cercano
    tree = KDTree(reference_colors)
    _, indices = tree.query(img_np, k=1)
    indices = indices.flatten()

    # Crear imagen RGB limpia
    img_rgb = np.array([reference_colors[i] for i in indices], dtype=np.uint8).reshape(img.size[1], img.size[0], 3)

    # Crear imagen en escala de grises
    img_gris = np.array([grayscale_values[i] for i in indices], dtype=np.uint8).reshape(img.size[1], img.size[0])

    # Guardar resultados
    if path_rgb_salida is not None:
        Image.fromarray(img_rgb).save(path_rgb_salida)
        print(f"Imagen RGB guardada en: {path_rgb_salida}")
    Image.fromarray(img_gris).save(path_gris_salida)
    #print(f"Imagen en escala de grises guardada en: {path_gris_salida}")

# 4. Ejecutar la función con tus archivos
def get_names(file: str):
    names = DataFrame(pd.read_csv(file, header= None))
    #Get the first column
    return names[0]

names = get_names("NamesSPIDataset.csv")
for name in names:
    procesar_imagen(
        '{}/{}{}'.format("app/datasets/SPIReescaleFull", name, ""),
        '{}/{}{}'.format("app/datasets/SPIFullNoBlurr", name, ""),      
    )
    #img = image.load_img('{}/{}{}'.format("app/datasets/SPIReescaleFull", name, ".png"), target_size= (260, 640, 3))
    #img = np.array(img)
    

procesar_imagen(
    "20000105.png",
    "imagen_rgb_limpia.png",
    "imagen_mascara_gris.png"
)
