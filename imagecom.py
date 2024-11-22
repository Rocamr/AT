import os
import pandas as pd
import cv2

# Clases a incluir y su índice para YOLO
CLASSES = {
    "Handgun": 0,
    "Rifle": 1,
    "Knife": 2,
    "Bomb": 3
}

# Función para convertir anotaciones
def convert_annotations(csv_path, images_dir, labels_dir):
    os.makedirs(labels_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        class_name = row['LabelName']
        if class_name not in CLASSES:
            continue

        class_id = CLASSES[class_name]
        image_id = row['ImageID']
        x_min, x_max = row['XMin'], row['XMax']
        y_min, y_max = row['YMin'], row['YMax']

        # Cargar la imagen para obtener las dimensiones
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Advertencia: No se pudo cargar la imagen {image_path}")
            continue

        img_height, img_width, _ = image.shape

        # Convertir a formato YOLO (coordenadas normalizadas)
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # Guardar anotaciones en .txt
        label_file = os.path.join(labels_dir, f"{image_id}.txt")
        with open(label_file, 'a') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Convertir para el conjunto de entrenamiento
convert_annotations('D:/Asitec/OIDv4_ToolKit/OID/csv_folder/train-annotations-bbox.csv',
                    'D:/Asitec/OIDv4_ToolKit/OID/Dataset/train',
                    'D:/Asitec/OIDv4_ToolKit/OID/Dataset/train/Handgun_Rifle_Knife_Bomb/Label')
