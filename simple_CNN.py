import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RootSegmentationDataset(Dataset):
    """
    Asked chatgpt to help build a class that retrieves the images from a local 
    directory. 
    """
    def __init__(self, img_dir, mask_dir, transform=None):
        img_patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG'] # List holding file extensions 
        self.img_paths = [] # Empty variable to store paths 
        for pat in img_patterns:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, pat))) # Saves all the file directory paths of the images with their extensions.   
        self.img_paths = sorted(self.img_paths) # Arranges the list 
        
        # This is done similarly for masks with added logic to detect whether each mask has a corresponding image. 
        self.mask_paths = []
        for img_path in self.img_paths:
            base = os.path.splitext(os.path.basename(img_path))[0] # Splits it to only focus on name of the file. 
            mask_patterns = [f"{base}_ann.png", f"{base}_ann.PNG"]
            found = None
            for mpat in mask_patterns:
                candidate = os.path.join(mask_dir, mpat)
                if os.path.isfile(candidate):
                    found = candidate
                    break
            if not found:
                raise FileNotFoundError(f"Mask not found for image {img_path}: looked for {mask_patterns}")
            self.mask_paths.append(found)

        self.transform = transform # Saves transform object that is used to convert the images. 

    # Helper functions. 
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()
        return img, mask

# Building the CNN 
# choose SimpleSegNet
class SimpleSegNet(nn.Module):
    def __init__(self):
        super(SimpleSegNet, self).__init__()
        
        # Encoder 
        # Doing first convolution
        # Chat GPT recommended layer numbers.  
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # RGB input = 3, Kernel size = 3 (3x3 filter), learn 16 different features
            nn.ReLU(inplace=True), # Relu to set negative activations to 0
            # In = 3 channels, out = 16 
            # Second convolution 
            nn.Conv2d(16, 16, kernel_size=3, padding=1), # Takes the previous 16 feature maps and produce 16 new feature maps. 
            # In = 16 feature maps, out = new 16. 
            nn.ReLU(inplace=True) # Relu to set negative activations to 0
        )
        self.pool1 = nn.MaxPool2d(2) # downside from 256 to 128 

        # Doing the same thing as above 
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # In = 16 feature maps, out = 32 feature maps 
            nn.ReLU(inplace=True), # Relu to set negative activations to 0
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # Second convolution to refine those 32 feature maps 
            # In = 32, out = new 32 
            nn.ReLU(inplace=True) # Relu to set negative activations to 0
        )
        self.pool2 = nn.MaxPool2d(2)   # downside from 128 to 64 

        # Decoder 
        # Go up from 64x64 to 128x128
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) # Take 32 feature maps and bring back to 16, kernel size = 2x2. Move by 2 pixels 
        # Now take the 16 feature maps and refine them with conv + ReLu 
        # Conv + ReLu recommended by Chat GPT 
        self.dec2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Go from 128x128 to 256x256 
        # channels = 16 
        self.up1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        # Do same refining as before (conv + ReLu) 
        self.dec1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 1x1 convolution 
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        # Extractive the low level features 
        x1 = self.enc1(x)      
        p1 = self.pool1(x1)    

        # Extracting high level feature 
        x2 = self.enc2(p1)     
        p2 = self.pool2(x2)    

        # Decoder
        # Refining 
        u2 = self.up2(p2)      
        d2 = self.dec2(u2)     

        # Bringing back to original 
        u1 = self.up1(d2)   
        d1 = self.dec1(u1)     

        out = self.final_conv(d1) # Linear projection 
        # Sigmoid used to squash logits into [0,1]. This gives the per pixel probability. 
        return torch.sigmoid(out)


def main():
    # Getting the paths for training and test set 
    train_img_dir  = "/content/train/jpg_scans" # Training jpg scans 
    train_mask_dir = "/content/train/masks" # Training masks 
    test_img_dir   = "/content/test/jpg_scans" # Testing jpg scans 
    test_mask_dir  = "/content/test/masks" # Testing masks 

    # defining constants 
    batch_size = 8 # Batch size # Amount in each iteration 
    lr = 1e-3 # learning rate 
    epochs = 1 # epochs (number of iterations)

    # Resizing all images and masks to become 256x256 
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Putting all directories in a dataset 
    train_ds = RootSegmentationDataset(train_img_dir, train_mask_dir, transform)
    test_ds  = RootSegmentationDataset(test_img_dir,  test_mask_dir,  transform)

    # Used loader here to handle batching and shuffling 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # ChatGPT recommendation on how to store efficently 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # ChatGPT recommendation on how to store efficently
    # I.e. use gpu if not use cpu 
    
    # Producing and calling the network 
    model = SimpleSegNet().to(device)
    criterion = nn.BCELoss() # Using binary cross entropy (recommended via journal article) for segmentation
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Using Adam optimizer (recommended via journal article)

    # Training the model and calculating losses 
    for epoch in range(1, epochs + 1):
        model.train() # Setting the model to train 
        train_loss = 0.0 # Initial loss
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device) # Recommended by ChatGPT to move batch to GPU/CPU 
            preds = model(imgs) # Forward (compute predictions)
            loss = criterion(preds, masks) # Calculating the loss between the predicted binary image and real true binary image. 
            optimizer.zero_grad() # remove old gradients 
            loss.backward() # Calculate new gradients 
            optimizer.step() # Update model parameters (contants that change as defined before)
            train_loss += loss.item() # adding up loss value with each iteration 
        train_loss /= len(train_loader) # Calculating average loss for batches 

        model.eval() # Setting the model to evaluate 
        val_loss = 0.0 # Initiall validation loss 
        with torch.no_grad(): 
            for imgs, masks in test_loader:
                imgs, masks = imgs.to(device), masks.to(device) # Recommended by ChatGPT to move batch to GPU/CPU 
                val_loss += criterion(model(imgs), masks).item()
        val_loss /= len(test_loader) # Calculating the average loss for validation. 

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}") # printing the number of epochs being done, the training loss and validation 
        # loss to get an idea of how well the model is preforming or to see if the model is overfitting or underfitting. 

    # Saving the new learned parameters after training 
    torch.save(model.state_dict(), 'simple_segnet.pth')
    print("Training complete. Model saved to simple_segnet.pth")

def run_inference(model_path, new_img_dir, out_mask_dir, device='cpu'):
    os.makedirs(out_mask_dir, exist_ok=True) # Creating a folder on my desktop for saving the output binary images produced. 

    # Making sure the training tranforms match. Converting to tensor after resizing 256x256
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load model and weights
    model = SimpleSegNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Gathering all the jpgs in the directory of the new images 
    img_paths = sorted(glob.glob(os.path.join(new_img_dir, '*.jpg')))
    for img_path in img_paths:
        # Recommended by ChatGPT to load and preprocess the image
        img = Image.open(img_path).convert('RGB')
        inp = transform(img).unsqueeze(0).to(device)

        # Doing the forward pass 
        # No gradients 
        with torch.no_grad():
            pred = model(inp)                       
        # Creating the binary mask
        mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255 # If greater than per pixel prob is > 0.5 then make white (root)

        # saving onto desktop 
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_mask_dir, f"{base}_pred.png")
        Image.fromarray(mask).save(out_path)

if __name__ == '__main__':

    main()

    model_path   = "simple_segnet.pth"
    new_img_dir  = "/content/new_images"
    out_mask_dir = "/content/predicted_masks"
    device       = "cuda" if torch.cuda.is_available() else "cpu"

    run_inference(model_path, new_img_dir, out_mask_dir, device)