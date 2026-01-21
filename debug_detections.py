import cv2
import numpy as np
import fitz
import sys
from formula_detector import FormulaDetector

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "test_formulas.pdf" # get pdf path from command line or use default sample
page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0
doc = fitz.open(pdf_path)
page = doc[page_num] # pick target page

zoom = 300 / 72
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)) # render page at 300 dpi for consistent detection quality
img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
if pix.n == 4:
    page_image = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
elif pix.n == 3:
    page_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)

print(f"{pdf_path} - Page {page_num + 1}")
print(f"Page size: {page_image.shape[1]}x{page_image.shape[0]}")

detector = FormulaDetector() # use the shared formula detector pipeline
formulas = detector.detect_formulas(page_image) # detect formulas on the rendered page
debug_image = page_image.copy() # visualize detections

for i, (x, y, w, h) in enumerate(formulas):
    # draw green rectangle for each detected formula region
    cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(debug_image, f"{i+1}", (x, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

doc.close()

output_file = f"debug_page{page_num+1}_filtered.png"
cv2.imwrite(output_file, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

print(f"Detected formulas (after filtering): {len(formulas)}")
print(f"\nVisualization saved: {output_file}")

if formulas:
    print("\nDetected formula regions:")
    for i, (x, y, w, h) in enumerate(formulas, 1):
        area = w * h
        print(f"  {i}. position ({x}, {y}), size {w}x{h}, area {area}")