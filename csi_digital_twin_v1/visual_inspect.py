"""Inspect a rendered pallet image, draw bounding boxes, and save a visual result."""
import os
import cv2
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# 1. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="5PSJzCRgWlxxoaIaYDJk"
).configure(InferenceConfiguration(
    api_key_transport="header"
))

# Test variation 1 (which contains defects)
image_path = "cap_seal_001_1.png"
output_path = "inspected_result_1.png"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Could not find image to inspect at: {image_path}")

print(f"Running visual inspection on {image_path}...")

# 2. Run workflow
result = client.run_workflow(
    workspace_name="tray-inspection",
    workflow_id="tray-inspection-1-rfdetr-small-t1-vtray-inspection-1-rfdetr-small-t1-logic",
    images={"image": image_path},
    use_cache=True
)

# 3. Load image with OpenCV for drawing
img = cv2.imread(image_path)
predictions = []

try:
    if isinstance(result, list) and len(result) > 0:
        step_output = result[0]
        if isinstance(step_output, dict) and "predictions" in step_output:
            inner = step_output["predictions"]
            if isinstance(inner, dict) and "predictions" in inner:
                predictions = inner["predictions"]

    defects_found = []
    
    # 4. Draw boxes on the image
    for pred in predictions:
        class_name = pred.get("class", "unknown")
        confidence = pred.get("confidence", 0.0)
        
        # Roboflow coordinates are center-based (x, y, width, height)
        x = int(pred.get("x") - pred.get("width") / 2)
        y = int(pred.get("y") - pred.get("height") / 2)
        w = int(pred.get("width"))
        h = int(pred.get("height"))

        # Choose box color: Green for pass, Red/Orange for defects
        color = (0, 255, 0) if class_name == "pass" else (0, 0, 255)
        
        if class_name != "pass":
            defects_found.append(f"{class_name} ({confidence:.2f})")
            color = (0, 0, 255) # Red for errors

        # Draw rectangle and label on image
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{class_name} {confidence:.2f}", (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Determine final Pallet Status
    status_text = "PASSED (GOOD)" if len(defects_found) == 0 else "FAILED (BAD)"
    status_color = (0, 255, 0) if len(defects_found) == 0 else (0, 0, 255)
    
    # Draw big banner status at the top of the image
    cv2.putText(img, f"PALLET: {status_text}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Save visual output
    cv2.imwrite(output_path, img)
    print(f"Saved visual inspection image to: {output_path}")
    print(f"Status: {status_text} | Defects caught: {len(defects_found)}")

except Exception as e:
    print(f"Error processing visual output: {e}")