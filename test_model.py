"""
    To run the test_model.py, you can run the program using the following command:

        python test_model.py --model_path /path/to/pretrained/model.pth 
                                --image_folder_path /path/to/folder/with/images 
                                --output_file_path /path/to/output.txt

"""

import os
from PIL import Image
from torchvision import transforms
import torch
import torchvision.transforms as transforms
import torchvision  
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 
label_map = {'bmw serie 1': 0,
'chevrolet spark': 1,
'chevroulet aveo': 2,
'clio': 3,
'duster': 4,
'golf': 5,
'hyundai i10': 6,
'hyundai tucson': 7,
'logan': 8,
'megane': 9,
'mercedes class a': 10,
'nemo citroen': 11,
'octavia': 12,
'picanto': 13,
'polo': 14,
'sandero': 15,
'seat ibiza': 16,
'symbol': 17,
'toyota corolla': 18,
'volkswagen tiguan': 19}

# Define the command line arguments
parser = argparse.ArgumentParser(description='Testing a pre-trained PyTorch model on a folder of images')
parser.add_argument('--model_path', type=str, help='path to the pre-trained PyTorch model')
parser.add_argument('--image_folder_path', type=str, help='path to the folder with images to be tested')
parser.add_argument('--output_file_path', type=str, help='path to the output text file')
args = parser.parse_args()

class ResNet50(ResNet):
    def __init__(self, num_classes=1000, pretrained=True):
        super(ResNet50, self).__init__(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
        if pretrained:
            self.load_state_dict(torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls['resnet50'], progress=True, check_hash=True))

model = ResNet50(num_classes=1000, pretrained=True)

# Load the model from the .pth file
model = torch.load(args.model_path, map_location=torch.device('cpu'))

# Define the transformations to be applied to the test images
transform = transforms.Compose([    
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Set the directory containing the test images
test_dir = args.image_folder_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a dictionary to hold the results
results = {}

# Iterate through the test images and make predictions using the model
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg'):
        # Load the image
        img_path = os.path.join(test_dir, filename)
        img = Image.open(img_path)

        # Apply the transformations to the image
        img = transform(img)
        img = img.to(device)
        # Add a batch dimension to the tensor
        img = img.unsqueeze(0)

        # Make a prediction using the model
        with torch.no_grad():
            output = model(img)
            prediction = torch.argmax(output, dim=1)

        # Store the prediction result in the dictionary
        label = list(label_map.keys())[list(label_map.values()).index(prediction.item())]
        if label in results:
            results[label].append(filename)
        else:
            results[label] = [filename]

# Initialize an empty dictionary to store the results
results = {}

# Iterate through the test images and make predictions using the model
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg'):
        # Load the image
        img_path = os.path.join(test_dir, filename)
        img = Image.open(img_path)

        # Apply the transformations to the image
        img = transform(img)
        img = img.to(device)
        # Add a batch dimension to the tensor
        img = img.unsqueeze(0)

        # Make a prediction using the model
        with torch.no_grad():
            output = model(img)
            prediction = torch.argmax(output, dim=1)

        # Store the prediction result in the dictionary
        label = list(label_map.keys())[list(label_map.values()).index(prediction.item())]
        if label in results:
            results[label].append(filename)
        else:
            results[label] = [filename]

# Print the results
for label, filenames in results.items():
    count = len(filenames)
    print(f"{label}: {count} - {filenames}")

# Save the results to a text file
with open(args.output_file_path, 'w') as f:
    for label, filenames in results.items():
        count = len(filenames)
        f.write(f"{label}: {count} - {str(filenames)}\n")