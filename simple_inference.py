import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from models import networks

def get_args():
    parser = argparse.ArgumentParser(description="Simple Inference for Asymmetric Pix2PixHD")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to folder containing input sonar images")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to your specific .pth weights file (e.g., ./checkpoints/experiment/latest_net_G.pth)")
    parser.add_argument("--output_folder", type=str, default="./inference_outputs", help="Where to save results")
    return parser.parse_args()

def load_generator(weights_path):
    print(f"Loading weights from: {weights_path}")
    
    # Instantiate your custom asymmetric GlobalGenerator structure
    # input_nc=3 handles raw images from disk, output_nc=1 emits your 48x96 layer matrix
    netG = networks.GlobalGenerator(input_nc=3, output_nc=1, ngf=64, n_blocks=9)
    
    # Load the state dictionary matrix safely
    state_dict = torch.load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Handle pix2pixHD Multi-GPU 'module.' naming wrappers if present
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
        
    netG.load_state_dict(new_state_dict)
    netG.eval()
    if torch.cuda.is_available():
        netG = netG.cuda()
    return netG

def preprocess_image(img_path):
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        return None
        
    # Enforce your custom training dimension boundaries precisely
    img = cv2.resize(img, (96, 512)) # Width=96, Height=512
    
    # Convert BGR array data to floating-point and normalize to [-1, 1] range matching Tanh layers
    img = (img.astype(np.float32) / 127.5) - 1.0
    
    # HWC -> CHW layout sequence mapping
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    return img_tensor

def postprocess_tensor(tensor):
    # Extract out of CUDA memory tracking back into basic CPU arrays
    img = tensor.detach().cpu().float().numpy()[0]
    
    # Rescale back from [-1, 1] values to standard [0, 255] unsigned pixel boundaries
    img = (np.transpose(img, (1, 2, 0)) + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def run_inference():
    args = get_args()
    
    input_dir = Path(args.input_folder)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize your modified asymmetric network layout
    netG = load_generator(args.weights_path)
    
    # Grab all image extensions
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    img_files = []
    for ext in valid_extensions:
        img_files.extend(input_dir.glob(ext))
    img_files = sorted(img_files)
    
    print(f"Found {len(img_files)} files to evaluate. Commencing inference loop...")
    
    with torch.no_grad():
        for img_p in img_files:
            # Prepare tensor dimensions
            input_tensor = preprocess_image(img_p)
            if input_tensor is None:
                continue
                
            # Execute cross-modal generation pass
            output_tensor = netG(input_tensor)
            
            # Reconstruct back to viewable image map parameters
            out_img = postprocess_tensor(output_tensor)
            
            # Save array directly to target destination folder
            out_name = output_dir / f"generated_{img_p.name}"
            cv2.imwrite(str(out_name), out_img)
            
    print(f"Inference run finished. All generated arrays saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    run_inference()