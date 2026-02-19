from dataset import get_dataloaders

train_loader, val_loader, classes = get_dataloaders()

print("Classes:", classes)
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

images, labels = next(iter(train_loader))
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
