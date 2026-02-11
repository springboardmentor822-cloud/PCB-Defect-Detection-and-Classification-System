import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json
from pathlib import Path

def plot_training_history(history_path, results_dir):
    """
    Plots training and validation loss and accuracy from history JSON.
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['test_acc'], 'r-', label='Test Acc')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / "learning_curves.png")
    plt.close()
    print(f"Learning curves saved to {results_dir / 'learning_curves.png'}")

def evaluate_model(data_dir, model_path, results_dir):
    """
    Evaluates the trained model and generates performance visualizations.
    """
    data_dir = Path(data_dir)
    test_dir = data_dir / "test"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Transforms for testing
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    class_names = test_dataset.classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model architecture
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("--- Starting Evaluation ---")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate Accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save Accuracy Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(report)
    print("Classification report saved.")

    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Test Acc: {accuracy:.4f})')
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved to {results_dir / 'confusion_matrix.png'}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_PATH = BASE_DIR / "models" / "classifier_best.pt"
    RESULTS_DIR = BASE_DIR / "results" / "milestone2"
    HISTORY_PATH = RESULTS_DIR / "training_history.json"

    if MODEL_PATH.exists():
        evaluate_model(DATA_DIR, MODEL_PATH, RESULTS_DIR)
        if HISTORY_PATH.exists():
            plot_training_history(HISTORY_PATH, RESULTS_DIR)
    else:
        print(f"Model not found at {MODEL_PATH}. Please train the model first.")
