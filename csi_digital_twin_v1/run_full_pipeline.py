"""Unified pipeline: Generate Isaac Sim digital twin images and run Roboflow inference."""
import os
import cv2
import glob
import subprocess
import argparse
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

ap = argparse.ArgumentParser()
ap.add_argument("--out-prefix", default="test_batch_002", help="Prefix for generated images")
ap.add_argument("--out-dir", default="inference_results", help="Directory for annotated images")
a = ap.parse_args()

print("--- STEP 1: GENERATING DIGITAL TWIN IMAGES ---")
sim_cmd = [
    "/root/IsaacLab/_isaac_sim/python.sh", 
    "tools/render_example.py", 
    "--camera", "CapSealCam", 
    "--stage", "stations/tray_inspection/tray_inspection.usdc", 
    "--out", f"{a.out_prefix}.png", 
    "--subframes", "32"
]
subprocess.run(sim_cmd, check=True)

print("\n--- STEP 2: RUNNING ROBOFLOW INFERENCE ---")
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="5PSJzCRgWlxxoaIaYDJk"
).configure(InferenceConfiguration(api_key_transport="header"))

os.makedirs(a.out_dir, exist_ok=True)
image_paths = glob.glob(f"{a.out_prefix}_*.png")

if not image_paths:
    print(f"No images generated with prefix {a.out_prefix}.")
    exit()

def extract_boxes(data):
    """Recursively search the JSON payload for Roboflow bounding box dictionaries."""
    boxes = []
    if isinstance(data, dict):
        # Identify a bounding box payload by its standard coordinate keys
        if 'class' in data and 'x' in data and 'y' in data:
            boxes.append(data)
        else:
            for v in data.values():
                boxes.extend(extract_boxes(v))
    elif isinstance(data, list):
        for item in data:
            boxes.extend(extract_boxes(item))
    return boxes

for img_path in image_paths:
    filename = os.path.basename(img_path)
    print(f"Processing {filename}...")
    
    result = client.run_workflow(
        workspace_name="tray-inspection",
        workflow_id="tray-inspection-2-rfdetr-small-t1-vtray-inspection-2-rfdetr-small-t1-logic",
        images={"image": img_path},
        use_cache=True
    )
    
    predictions = extract_boxes(result)
    img = cv2.imread(img_path)
    
    if not predictions:
        print(f"  No defects/vials detected in {filename}.")
    
    for bbox in predictions:
        w = bbox.get('width', bbox.get('w', 0))
        h = bbox.get('height', bbox.get('h', 0))
        x = bbox.get('x', 0)
        y = bbox.get('y', 0)
        class_name = bbox.get('class', bbox.get('class_name', 'Unknown'))
        conf = bbox.get('confidence', 0.0)
        
        # Calculate top-left and bottom-right box coordinates
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        
        # Green for pass/good, Red for any defect
        color = (0, 255, 0) if class_name.lower() == "pass" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    out_path = os.path.join(a.out_dir, filename)
    cv2.imwrite(out_path, img)
    print(f"  Saved annotated image to {out_path}")

print(f"\nPipeline complete. Annotated images are in the '{a.out_dir}' directory.")