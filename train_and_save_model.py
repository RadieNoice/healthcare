import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import heart_model

def train_and_save():
    print("Initializing LightweightCNN_SE model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = heart_model.LightweightCNN_SE(num_classes=5).to(device)
    
    # Generate synthetic training signals representing 5 AAMI workout classes
    np.random.seed(42)
    X_train_list = []
    y_train_list = []
    
    samples_per_class = 200
    for class_id in range(5):
        for _ in range(samples_per_class):
            sig = heart_model.generate_sample_signal(target_class=class_id, noise_std=0.04)
            X_train_list.append(sig)
            y_train_list.append(class_id)
            
    X_train = torch.tensor(np.array(X_train_list), dtype=torch.float32).unsqueeze(1).to(device)
    y_train = torch.tensor(np.array(y_train_list), dtype=torch.long).to(device)
    
    print(f"Dataset shape: {X_train.shape}, Labels shape: {y_train.shape}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    model.train()
    epochs = 35
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y_train).float().mean().item()
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Accuracy: {acc*100:.2f}%")
            
    # Save trained checkpoint
    save_path = "heart_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Successfully trained and saved model weights to '{save_path}'!")

if __name__ == "__main__":
    train_and_save()
