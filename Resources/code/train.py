import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pathlib import Path
from model import build_model

from dataset import get_dataloaders, CLASSES

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_SAVE_DIR = BASE_DIR / "Resources" / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True) 
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "mobilenetv3_freshness.pth"

EPOCHS = 30 
LEARNING_RATE = 0.0001

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение запущено на: {device}")

    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(batch_size=16)
    
    model = build_model(len(CLASSES)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    best_val_loss = 10.0

    print("Начинаем обучение...")
    for epoch in range(EPOCHS):
        # --- ФАЗА ОБУЧЕНИЯ ---
        model.train() 
        running_loss = 0.0
        correct_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct_train.double() / len(train_dataset)

        # --- ФАЗА ВАЛИДАЦИИ ---
        model.eval() 
        val_loss = 0.0
        correct_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct_val.double() / len(val_dataset)

        print(f"Эпоха {epoch+1}/{EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")


        if epoch_val_acc >= best_val_acc and epoch_val_loss <= best_val_loss:
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" -> Сохранена новая лучшая модель!")

    print(f"Обучение завершено. Итоговая модель лежит в {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
