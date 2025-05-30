# Importaciones (sin cambios)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# Comprobar si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Ruta a tu carpeta 'train'
data_dir = 'C:/Users/Adrian Jaret/Desktop/train'

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset completo
print("Cargando dataset...")
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
print(f"Dataset creado con {len(full_dataset)} imágenes.")

# Verifica lectura de primeras imágenes
print("🔍 Probando carga de imágenes:")
for i in range(min(5, len(full_dataset))):
    image, label = full_dataset[i]
    print(f"Imagen {i + 1} leída correctamente, clase: {label}")

# Clases detectadas
class_names = full_dataset.classes
print("Clases detectadas:", class_names)

# Dividir en entrenamiento y validación (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"Train: {train_size}, Validación: {val_size}")

# DataLoaders
print("Creando DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
print("DataLoaders listos.")

# Verifica si carga el primer lote correctamente
print("Verificando primer lote del train_loader...")
for i, (inputs, labels) in enumerate(train_loader):
    print(f"Lote {i + 1} cargado con {len(inputs)} imágenes.")
    break

# Modelo preentrenado
print("Cargando modelo...")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)
print("Modelo listo.")

# Pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
print("Iniciando entrenamiento...")
epochs = 5
for epoch in range(epochs):
    print(f"\nÉpoca {epoch+1}/{epochs}")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 10 == 0:
            print(f"Lote {i + 1} procesado")

    train_acc = 100 * correct / total
    print(f"Época {epoch+1} completada - Pérdida: {running_loss:.4f} - Precisión entrenamiento: {train_acc:.2f}%")

# Validación
print("\nIniciando validación...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_acc = 100 * correct / total
print(f"Precisión en validación: {val_acc:.2f}%")

# Guardar modelo
ruta_guardado = r"C:\Users\Adrian Jaret\Desktop\Programas Py\IA\T2U4\modelo_flores.pth"
print("Entrenamiento terminado, guardando modelo...")
torch.save(model.state_dict(), ruta_guardado)
print(f"Modelo guardado en '{ruta_guardado}'")
