import os
import cv2
import glob
import subprocess
import argparse
import json
import re
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

ap = argparse.ArgumentParser()
ap.add_argument("--out-prefix", default="test_batch_005", help="Prefix for generated images")
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

for img_path in image_paths:
    filename = os.path.basename(img_path)
    print(f"Processing {filename}...")
    
    try:
        result = client.run_workflow(
            workspace_name="tray-inspection",
            workflow_id="tray-inspection-2-rfdetr-small-t1-vtray-inspection-2-rfdetr-small-t1-logic",
            images={"image": img_path},
            use_cache=True
        )
    except Exception as e:
        print(f"  API Error on {filename}: {e}")
        continue
    
    try:
        if hasattr(result, 'model_dump_json'):
            raw_str = result.model_dump_json()
        elif hasattr(result, 'to_json'):
            raw_str = result.to_json()
        else:
            raw_str = json.dumps(result, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
    except Exception as e:
        raw_str = str(result)
        
    predictions = []
    
    try:
        data = json.loads(raw_str)
        
        def find_predictions(obj):
            boxes = []
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and 'x' in item and 'y' in item and ('width' in item or 'w' in item):
                        boxes.append(item)
                    else:
                        boxes.extend(find_predictions(item))
            elif isinstance(obj, dict):
                if 'x' in obj and 'y' in obj and ('width' in obj or 'w' in obj):
                     boxes.append(obj)
                else:
                    for k, v in obj.items():
                        boxes.extend(find_predictions(v))
            return boxes
            
        predictions = find_predictions(data)
    except Exception as e:
        print(f"JSON Parsing failed: {e}")

    img = cv2.imread(img_path)
    
    if not predictions:
        print(f"  NO BOUNDING BOXES DETECTED BY MODEL.")
        continue
        
    print(f"  Found {len(predictions)} objects.")
    
    for bbox in predictions:
        w = float(bbox.get('width', bbox.get('w', 0)))
        h = float(bbox.get('height', bbox.get('h', 0)))
        x = float(bbox.get('x', 0))
        y = float(bbox.get('y', 0))
        
        if w == 0 or h == 0: continue
            
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        
        class_name = bbox.get('class', bbox.get('class_name', bbox.get('label', 'Unknown')))
        if isinstance(class_name, dict): class_name = str(class_name)
        label = str(class_name)
        
        color = (0, 255, 0) if label.lower() == "pass" else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw filled background rectangle for text readability
        font_scale = 0.4
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw + 2, y1), color, -1)
        
        # Draw black text over the colored background
        cv2.putText(img, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
    out_path = os.path.join(a.out_dir, filename)
    cv2.imwrite(out_path, img)
    print(f"  Saved annotated image to {out_path}")

print(f"\nPipeline complete. Annotated images are in the '{a.out_dir}' directory.")
