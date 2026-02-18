from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PCBDataset(Dataset):
    def __init__(self, root_dir):
        self.root = Path(root_dir)

        
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # image paths
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for img_path in list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")):
                self.samples.append((img_path, self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        # Data augmentation 
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label
