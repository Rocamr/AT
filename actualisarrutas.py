import os

# Mapeo de nombres de clases a índices
CLASSES = {
    "Handgun": 0,
    "Rifle": 1,
    "Knife": 2,
    "Bomb": 3
}

label_dir = 'D:/Asitec/OIDv4_ToolKit/OID/Dataset/train/Handgun_Rifle_Knife_Bomb/Label'

for label_file in os.listdir(label_dir):
    label_path = os.path.join(label_dir, label_file)

    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.split()
        if parts[0] in CLASSES:
            class_id = CLASSES[parts[0]]
            new_line = f"{class_id} {' '.join(parts[1:])}\n"
            new_lines.append(new_line)
        else:
            print(f"Clase desconocida encontrada en {label_path}: {parts[0]}")

    # Guardar las etiquetas corregidas
    with open(label_path, 'w') as f:
        f.writelines(new_lines)
