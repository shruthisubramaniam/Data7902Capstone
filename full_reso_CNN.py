import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os
import zipfile
current_dir = os.getcwd()


# with zipfile.ZipFile(current_dir + "\\train_jpg_scans.zip", 'r') as zip_ref:
#     zip_ref.extractall("train_jpg_scans")

# with zipfile.ZipFile(current_dir + "\\test_jpg_scans.zip", 'r') as zip_ref:
#     zip_ref.extractall("test_jpg_scans")

# with zipfile.ZipFile(current_dir + "\\train_masks.zip", 'r') as zip_ref:
#     zip_ref.extractall("train_masks")

# with zipfile.ZipFile(current_dir + "\\test_masks.zip", 'r') as zip_ref:
#     zip_ref.extractall("test_masks")

# Implement later
# with zipfile.ZipFile(current_dir + "/new_images.zip", 'r') as zip_ref:
#     zip_ref.extractall("new_images")

# Getting the paths for training and test set
# train_img_dir  = "/content/train/jpg_scans" # Training jpg scans
# train_mask_dir = "/content/train/masks" # Training masks
# test_img_dir   = "/content/test/jpg_scans" # Testing jpg scans
# test_mask_dir  = "/content/test/masks" # Testing masks

train_img_dir  = current_dir + "/train_jpg_scans/jpg_scans" # Training jpg scans
train_mask_dir = current_dir + "/train_masks/masks" # Training masks
test_img_dir   = current_dir + "/test_jpg_scans/jpg_scans" # Testing jpg scans
test_mask_dir  = current_dir + "/test_masks/masks" # Testing masks

model_path   = current_dir + "/best_large_image_model.pth"
new_img_dir  = current_dir + "/new_images"
out_mask_dir = current_dir + "/predicted_masks"

os.makedirs(new_img_dir, exist_ok=True)  # Creating the folder. 

class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels): # initialising the parent nn.model 
        super(DoubleConv, self).__init__()
        # Define two convolutional layers in sequence
        self.conv = nn.Sequential(
            # First conv layer (3x3). changes channel count to out channels
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            # Helps training run smoother
            nn.BatchNorm2d(out_channels),
            # Activation so it's not just linear
            nn.ReLU(inplace=True),
            # Second conv layer (same idea as above, but stays at out_channels)
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Just run the input through the conv block
        return self.conv(x)

class SimplifiedFullResolutionCNN(nn.Module): 
    def __init__(self, n_channels=3, n_classes=1):
        super(SimplifiedFullResolutionCNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder. starting small to save memory
        self.inc = DoubleConv(n_channels, 16) # first conv block
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Middle part, most compressed features
        self.bottleneck = DoubleConv(256, 512)
        
        # # Decoder. bringing features back up while combining with skips
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)  # 256 + 256 from skip connection
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)  # 128 + 128 from skip connection
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)   # 64 + 64 from skip connection
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, 32)    # 32 + 32 from skip connection
        
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(32, 16)    # 16 + 16 from skip connection
        
        # Final 1x1 conv to get required output channels
        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Keeping original size to resize at the end
        input_size = x.shape[2:]
        
        # Encoder. going down and save feature maps for skips
        x1 = self.inc(x)      # 16 channels
        x2 = self.down1(x1)   # 32 channels
        x3 = self.down2(x2)   # 64 channels
        x4 = self.down3(x3)   # 128 channels
        x5 = self.down4(x4)   # 256 channels
        
        # Bottleneck
        x = self.bottleneck(x5)  # 512 channels
        
        # Decoder. going back up and combine with earlier features
        x = self.up1(x)  # 256 channels
        if x.shape[2:] != x5.shape[2:]:
            x = F.interpolate(x, size=x5.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x5, x], dim=1)  # 256 + 256 = 512 channels
        x = self.conv1(x)  # 256 channels
        
        x = self.up2(x)  # 128 channels
        if x.shape[2:] != x4.shape[2:]:
            x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x4, x], dim=1)  # 128 + 128 = 256 channels
        x = self.conv2(x)  # 128 channels
        
        x = self.up3(x)  # 64 channels
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x3, x], dim=1)  # 64 + 64 = 128 channels
        x = self.conv3(x)  # 64 channels
        
        x = self.up4(x)  # 32 channels
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x], dim=1)  # 32 + 32 = 64 channels
        x = self.conv4(x)  # 32 channels
        
        x = self.up5(x)  # 16 channels
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x], dim=1)  # 16 + 16 = 32 channels
        x = self.conv5(x)  # 16 channels
        
        # Output layer
        logits = self.outc(x)  # 1 channel
        
        # Ensuring output matches input size exactly
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        return torch.sigmoid(logits)

class LargeImageDataset(Dataset):
     # Dataset that can handle really big images without too much extra stuff
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        # Just returning how many images we have
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Opening image and mask
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        # Applying transforms
        # If transforms were given, use them, else just turn into tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Converting mask to binary tensor
        # Making mask binary (0 or 1)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

def create_optimized_data_loaders(image_dir, mask_dir, batch_size=1, val_split=0.2):
    # Creating train/val dataloaders that won't blow up RAM for big images
    
    # Getting sorted file lists
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Full paths
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]
    
    # Splitting into train/val
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=val_split, random_state=42
    )
    
    # Very minimal transforms to keep memory low
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Creating datasets
    train_dataset = LargeImageDataset(train_imgs, train_masks, transform=train_transform)
    val_dataset = LargeImageDataset(val_imgs, val_masks, transform=val_transform)
    
    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def simple_dice_loss(pred, target, smooth=1e-6):
    # Basic dice loss (good for segmentation tasks)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice

def train_large_image_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-3):
    # Training model in a way that's safer for big images
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device")
    
    model.to(device)
    
    # using SCD on CPU as it is less memory. Using Adam on GPU
    if device.type == 'cpu':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            try:
                images, masks = images.to(device), masks.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                
                # Simple loss. Doing this for efficiency
                loss = simple_dice_loss(outputs, masks) + F.binary_cross_entropy(outputs, masks)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Clearing cache frequently for large images
                del outputs, loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                print(f"  Batch {batch_idx+1}/{len(train_loader)} completed")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Out of memory at batch {batch_idx+1}. Skipping...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                try:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = simple_dice_loss(outputs, masks) + F.binary_cross_entropy(outputs, masks)
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Clearing cache
                    del outputs, loss
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  Validation OOM at batch {batch_idx+1}. Skipping...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        # Calculating averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Saving the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_large_image_model_20.pth')
            print(f"✓ Best model saved! Val Loss: {best_val_loss:.4f}")
        
        scheduler.step()
    
    return train_losses, val_losses

def predict_single_image(model, image_path, output_path=None, device='cuda', threshold=0.5):
    # Predicting mask for one image
    print(f"Processing: {image_path}")
    
    model.eval()
    
    # Loading image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"Image size: {original_size}")
    
    # Transforming (same as training)
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        try:
            # Getting prediction
            prediction = model(image_tensor)
            prediction = prediction.squeeze().cpu().numpy()
            
            print(f"Prediction shape: {prediction.shape}")
            
            # Converting to binary mask
            binary_mask = (prediction > threshold).astype(np.uint8) * 255
            
            # Saving if output path provided
            if output_path:
                mask_image = Image.fromarray(binary_mask, mode='L')
                mask_image.save(output_path)
                print(f"Mask saved to: {output_path}")
            
            # Returning both probability and binary mask
            return binary_mask, prediction
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Image too large for memory. Try using CPU or reducing image size.")
                return None, None
            else:
                raise e

def predict_batch_images(model, input_dir, output_dir, device='cuda', threshold=0.5):
    # Predicting masks for every image in a folder
    
    # Creating output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Getting all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    
    for i, image_file in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_file}")
        
        input_path = os.path.join(input_dir, image_file)
        
        # Creating output filename (same name but .png extension)
        base_name = os.path.splitext(image_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}_mask.png")
        
        # Predicting
        binary_mask, prob_mask = predict_single_image(
            model, input_path, output_path, device, threshold
        )
        
        if binary_mask is not None:
            # Calculating some stats
            root_pixels = np.sum(binary_mask > 0)
            total_pixels = binary_mask.size
            root_percentage = (root_pixels / total_pixels) * 100
            
            results.append({
                'image': image_file,
                'root_pixels': root_pixels,
                'total_pixels': total_pixels,
                'root_percentage': root_percentage,
                'output_path': output_path
            })
        
        # Clearing memory
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    return results

import multiprocessing
multiprocessing.set_start_method('spawn', force=True) # helping avoid weird PyTorch issues 

print("Initializing simplified model for large images...")
model = SimplifiedFullResolutionCNN(n_channels=3, n_classes=1) # RGB input, 1 mask output

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params:,} parameters")

print("\nCreating data loaders...")
train_loader, val_loader = create_optimized_data_loaders(
    image_dir=train_img_dir, # folder with training images
    mask_dir=train_mask_dir, # folder with training masks
    batch_size=1, # keep it low for huge images
    val_split=0.2 # 20% for validation
)

print("\nStarting training...")
print("Note: For 3307x2339 images, this will require significant memory!")

# Training with fewer epochs for large images
train_losses, val_losses = train_large_image_model(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=20,  # Reducing for large images
    learning_rate=1e-3
)

# Plotting results (using this to check error over epochs)
if train_losses and val_losses:
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Large Image Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

print("\nTraining completed!") # helps me to kno that training is done and no error has popped up

