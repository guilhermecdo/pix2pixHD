import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import shutil
import numpy as np

class ResultCurator:
    def __init__(self, root, base_results_dir):
        self.root = root
        self.root.title("Sonar Result Curator (JET Edition)")
        self.base_dir = Path(base_results_dir)
        self.curated_dir = Path("./Curated_Final_Report")
        self.curated_dir.mkdir(exist_ok=True)

        self.experiments = sorted([d.name for d in self.base_dir.iterdir() if d.is_dir()])
        if not self.experiments:
            messagebox.showerror("Error", "No experiment folders found.")
            self.root.destroy()
            return

        self.setup_selection_ui()

    def setup_selection_ui(self):
        self.select_frame = tk.Frame(self.root, padx=20, pady=20)
        self.select_frame.pack()
        tk.Label(self.select_frame, text="Select Experiment:", font=("Arial", 12)).pack(pady=10)
        self.exp_var = tk.StringVar()
        self.combo = ttk.Combobox(self.select_frame, textvariable=self.exp_var, values=self.experiments, width=40)
        self.combo.pack(pady=10)
        self.combo.current(0)
        tk.Button(self.select_frame, text="Start Curation", command=self.start_curation, bg="#4444ff", fg="white").pack(pady=20)

    def start_curation(self):
        selected_exp = self.exp_var.get()
        self.current_exp_dir = self.base_dir / selected_exp / "test_latest" / "images"
        self.image_list = sorted(list(self.current_exp_dir.glob("*_synthesized_image.jpg")))
        
        if not self.image_list:
            messagebox.showwarning("Warning", "No images found in this experiment. Run build_final_report.py first!")
            return

        self.exp_report_dir = self.curated_dir / selected_exp
        self.exp_report_images = self.exp_report_dir / "images"
        self.exp_report_images.mkdir(parents=True, exist_ok=True)

        self.select_frame.destroy()
        self.index = 0
        self.good_results = []
        self.setup_main_ui()
        self.load_image()

    def setup_main_ui(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        self.label_info = tk.Label(self.main_frame, text="", font=("Arial", 10, "bold"))
        self.label_info.pack(pady=5)

        self.canvas = tk.Label(self.main_frame)
        self.canvas.pack(pady=10)

        btn_frame = tk.Frame(self.main_frame)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="REJECT", command=self.next_image, bg="#ff4444", fg="white", width=15).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="APPROVE", command=self.save_good, bg="#44ff44", width=15).pack(side=tk.LEFT, padx=10)

    def load_image(self):
        if self.index >= len(self.image_list):
            self.finish_curation()
            return

        img_p = self.image_list[self.index]
        self.label_info.config(text=f"Current: {img_p.name} ({self.index+1}/{len(self.image_list)})")

        base = str(img_p).replace("_synthesized_image.jpg", "")
        # Matches exactly what the report builder copies over
        paths = [Path(base + "_src_oculus.jpg"), img_p, Path(base + "_gt_didson.jpg")]
        
        row_gray = []
        row_jet = []

        for p in paths:
            if p.exists():
                img = cv2.imread(str(p))
                img = cv2.resize(img, (250, 250))
                row_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                gray_v = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                jet_v = cv2.applyColorMap(gray_v, cv2.COLORMAP_JET)
                row_jet.append(cv2.cvtColor(jet_v, cv2.COLOR_BGR2RGB))
            else:
                # Fallback visual cue if file wasn't created yet
                blank = np.zeros((250, 250, 3), dtype=np.uint8)
                cv2.putText(blank, "Missing File", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                row_gray.append(blank)
                row_jet.append(blank)

        top_row = np.hstack(row_gray)
        bottom_row = np.hstack(row_jet)
        full_display = np.vstack((top_row, bottom_row))
        
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(full_display))
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk

    def save_good(self):
        img_p = self.image_list[self.index]
        prefix = img_p.name.replace("_synthesized_image.jpg", "")
        
        jet_folder = self.exp_report_images / "jet_maps"
        jet_folder.mkdir(exist_ok=True)
        
        suffixes = ["_synthesized_image.jpg", "_src_oculus.jpg", "_gt_didson.jpg"]
        for s in suffixes:
            src = self.current_exp_dir / (prefix + s)
            if src.exists():
                shutil.copy(src, self.exp_report_images / (prefix + s))
                img = cv2.imread(str(src))
                jet = cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
                cv2.imwrite(str(jet_folder / (prefix + s)), jet)
        
        self.good_results.append(prefix)
        self.next_image()

    def next_image(self):
        self.index += 1
        self.load_image()

    def finish_curation(self):
        exp_name = self.exp_var.get()
        html = f"<html><body style='background:#0a0a0a; color:white; font-family:sans-serif; padding:40px;'>"
        html += f"<h1>Curated Highlights: {exp_name}</h1><hr>"
        
        for prefix in self.good_results:
            html += f"""
            <div style='margin-bottom:60px; background:#1a1a1a; padding:20px; border-radius:10px;'>
                <div style='display:flex; gap:10px; justify-content:center;'>
                    <img src='./images/{prefix}_src_oculus.jpg' width='300'>
                    <img src='./images/{prefix}_synthesized_image.jpg' width='300' style='border:2px solid #4CAF50;'>
                    <img src='./images/{prefix}_gt_didson.jpg' width='300'>
                </div>
                <div style='display:flex; gap:10px; justify-content:center; margin-top:10px;'>
                    <img src='./images/jet_maps/{prefix}_src_oculus.jpg' width='300'>
                    <img src='./images/jet_maps/{prefix}_synthesized_image.jpg' width='300'>
                    <img src='./images/jet_maps/{prefix}_gt_didson.jpg' width='300'>
                </div>
                <div style='text-align:center; color:#555; margin-top:10px;'>{prefix}</div>
            </div>"""
        
        with open(self.exp_report_dir / "curated_jet_report.html", "w") as f:
            f.write(html)
        
        messagebox.showinfo("Done", f"Curated {len(self.good_results)} pairs for {exp_name}")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ResultCurator(root, "./results")
    root.mainloop()