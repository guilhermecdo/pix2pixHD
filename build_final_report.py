import cv2
import os
import argparse
import shutil
from pathlib import Path
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Path to test_latest/images")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset (holoocean-didson-128)")
    return parser.parse_args()

def build_report(results_dir, dataset_dir):
    res_path = Path(results_dir).resolve()
    ds_path = Path(dataset_dir).resolve()
    
    test_a_path = ds_path / "test_A"
    test_b_path = ds_path / "test_B"
    
    jet_path = res_path / "jet_maps"
    jet_path.mkdir(exist_ok=True)
    
    fakes = sorted(list(res_path.glob("*_synthesized_image.jpg")))
    
    html_content = """
    <html>
    <head>
        <title>Sonar Translation Report - HoloOcean</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 40px; }
            .row-group { background: #1a1a1a; padding: 20px; border-radius: 8px; border: 1px solid #333; margin-bottom: 30px; }
            .image-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center; }
            img { width: 100%; max-width: 300px; border: 1px solid #444; background: black; }
            .label { font-weight: bold; font-size: 12px; color: #4CAF50; text-transform: uppercase; margin-bottom: 10px; display: block;}
        </style>
    </head>
    <body>
        <h1>HoloOcean Acoustic Translation Report</h1>
        <div style="color: #888; margin-bottom: 20px;">Source Dataset: """ + ds_path.name + """</div>
        <div class="container">
    """

    for fake_p in fakes:
        base_filename = fake_p.name.replace("_synthesized_image.jpg", "")
        
        oculus_src = test_a_path / f"{base_filename}.jpg"
        didson_src = test_b_path / f"{base_filename}.jpg"
        
        # Naming explicitly to match what curate_results_jet.py expects
        oculus_dst = res_path / f"{base_filename}_src_oculus.jpg"
        didson_dst = res_path / f"{base_filename}_gt_didson.jpg"
        gen_img    = fake_p.name

        if oculus_src.exists() and didson_src.exists():
            shutil.copy(oculus_src, oculus_dst)
            shutil.copy(didson_src, didson_dst)
        else:
            print(f"Warning: Missing pairs for file: {base_filename}.jpg")
            continue

        for img_name in [oculus_dst.name, gen_img, didson_dst.name]:
            img_path = res_path / img_name
            img = cv2.imread(str(img_path))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                jet = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                cv2.imwrite(str(jet_path / img_name), jet)

        html_content += f"""
        <div class="row-group">
            <div style="font-size: 11px; color: #555; margin-bottom: 10px;">Frame Reference: {base_filename}</div>
            <div class="image-grid">
                <div><span class="label">Oculus Input</span><img src="./{oculus_dst.name}"></div>
                <div><span class="label">Synthesized Didson</span><img src="./{gen_img}"></div>
                <div><span class="label">Target Ground Truth</span><img src="./{didson_dst.name}"></div>
            </div>
            <div class="image-grid" style="margin-top: 15px;">
                <div><img src="./jet_maps/{oculus_dst.name}"></div>
                <div><img src="./jet_maps/{gen_img}"></div>
                <div><img src="./jet_maps/{didson_dst.name}"></div>
            </div>
        </div>
        """

    html_content += "</div></body></html>"
    with open(res_path / "index.html", "w") as f:
        f.write(html_content)
    print(f"Clean report successfully written to: {res_path / 'index.html'}")

if __name__ == "__main__":
    args = get_args()
    build_report(args.results_dir, args.dataset_dir)