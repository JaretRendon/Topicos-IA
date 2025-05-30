import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np

# Definir dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Clases
classes = ['Abutilon', 'Acacia', 'Alchemilla', 'Allium', 'Amaryllis', 'BalloonFlower', 'Ballota', 'Bellflower', 'Borage', 'ButterflyBush',
           'Camellia', 'Carnation', 'CoralBells', 'Coreopsis', 'Cyclamen', 'Diascia', 'DutchIris', 'Euphorbia', 'Foxglove', 'Gazania',
           'Hawthorn', 'Hosta', 'Hyacinth', 'Iris', 'Larkspur', 'LemonVerbena', 'Magnolia', 'MargueriteDaisy', 'MorningGlory', 'Oleander',
           'Olearia', 'Pansy', 'PeaceLily', 'Peony', 'Petunia', 'Photinia', 'Pinks', 'Poppy', 'Quince', 'Sedum', 'Silene', 'Snapdragon',
           'Tuberose', 'WinterJasmine', 'YellowBell', 'Zinnia', 'daisy', 'dandelion', 'lilly', 'lotus', 'orchid', 'rose', 'sunflower', 'tulip']


# Cargar modelo preentrenado
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

# Reemplazar la última capa para nro de clases
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

# Cargar tus pesos entrenados
ruta_modelo = "C:/Users/Adrian Jaret/Desktop/Programas Py/IA/T2U4/modelo_flores.pth"
model.load_state_dict(torch.load(ruta_modelo, map_location=device))
model.to(device)
model.eval()

# Transformación para la cámara (igual que para el dataset)
base_transform = weights.transforms()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    base_transform,
])

# Abrir cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer frame")
        break

    # Convertir BGR a RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Aplicar transformaciones
    input_tensor = transform(img_rgb)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # Añadir batch dimension

    # Inferencia
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        label = classes[pred.item()]

    # Mostrar resultado en la imagen
    cv2.putText(frame, f"Flor: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Clasificador de Flores', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
