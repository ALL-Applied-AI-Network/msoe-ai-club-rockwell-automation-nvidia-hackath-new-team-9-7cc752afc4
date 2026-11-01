"""Inspect a rendered pallet image using the Roboflow workflow API."""
import os
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# 1. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="5PSJzCRgWlxxoaIaYDJk"
).configure(InferenceConfiguration(
    api_key_transport="header"
))

image_path = "cap_seal_001_2.png"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Could not find image to inspect at: {image_path}")

print(f"Sending {image_path} to Roboflow workflow for inspection...")

# 2. Run your workflow on the image
result = client.run_workflow(
    workspace_name="tray-inspection",
    workflow_id="tray-inspection-1-rfdetr-small-t1-vtray-inspection-1-rfdetr-small-t1-logic",
    images={
        "image": image_path
    },
    use_cache=True
)

# 3. Parse workflow results and apply Pass/Fail logic
predictions = []
try:
    # Safely extract predictions from Roboflow workflow JSON structure
    if isinstance(result, list) and len(result) > 0:
        step_output = result[0]
        if isinstance(step_output, dict) and "predictions" in step_output:
            inner = step_output["predictions"]
            if isinstance(inner, dict) and "predictions" in inner:
                predictions = inner["predictions"]

    print(f"\n--- INSPECTION RESULTS FOR: {image_path} ---")
    
    defects_found = []
    for pred in predictions:
        class_name = pred.get("class", "unknown")
        confidence = pred.get("confidence", 0.0)
        
        # Flag anything that isn't a normal passing cap/vial
        if class_name in ["Missing_Cap", "Crooked_Cap", "defective", "missing"]:
            defects_found.append(f"{class_name} (confidence: {confidence:.2f})")

    if len(defects_found) > 0:
        print("❌ PALLET STATUS: FAILED (BAD)")
        print("Detected Defects:")
        for defect in defects_found:
            print(f"  - {defect}")
    else:
        print(print(f"✅ PALLET STATUS: PASSED (GOOD) — Detected {len(predictions)} valid items."))

except Exception as e:
    print(f"Error parsing results: {e}")
    print("Raw output was:", result)