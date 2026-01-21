import os
import io
import logging
import tempfile
import base64
from typing import Dict, List, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
import numpy as np
from formula_detector import FormulaDetector

# this logger captures backend events (model load, pdf conversion, box processing)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Math formula LaTeX-OCR service", version="2.0")

# add cors middleware to allow the browser frontend (index.html/ results.html)
# to call the fastapi endpoints via fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# models for request/ response exchanged between frontend javascript and backend
# the BoundingBox coordinates are in canvas pixel space on the client and are used to crop the uploaded page image before passing it to pix2tex
class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    id: str

class BoxExtractionRequest(BaseModel):
    boxes: List[BoundingBox]
    imageData: str

class AutoDetectRequest(BaseModel):
    imageData: str
    pageNum: int = 1

def setup_latex_ocr_model(): # initializes the pix2tex LatexOCR model
    try:
        from pix2tex.cli import LatexOCR
        model = LatexOCR()
        logger.info("LaTeX-OCR model loaded successfully")
        return model
    except ImportError as e:
        logger.error(f"Failed to import LaTeX-OCR, install with: conda/ pip install pix2tex")
        raise
    except Exception as e:
        logger.error(f"Failed to load LaTeX-OCR model: {e}")
        raise

# initialize the model once at process start so all requests reuse the same instance
try:
    MODEL = setup_latex_ocr_model()
    MODEL_LOADED = True
except:
    MODEL = None
    MODEL_LOADED = False
    logger.warning("latex-ocr model failed to load")

def convert_pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]: # converts a binary pdf payload to a list of PIL.Image pages using pdf2image
    try:
        from pdf2image import convert_from_bytes
        
        images = convert_from_bytes(pdf_bytes, dpi=300)
        logger.info(f"Converted pdf to {len(images)} page images")
        return images
    except ImportError:
        logger.error("pdf2image not installed")
        return []
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        return []

@app.post("/api/simple-extract")
async def simple_extract(file: UploadFile = File(...)): # simplified endpoint that uses FormulaDetector to automatically detect and extract formulas from pdf pages
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="LaTeX-OCR model not loaded."
        )
    
    try:
        contents = await file.read()
        filename = file.filename.lower()
        
        results = []
        
        if filename.endswith('.pdf'):
            page_images = convert_pdf_to_images(contents)
            
            for page_num, page_image in enumerate(page_images):
                try:
                    # convert PIL image to numpy array for the detector
                    image_array = np.array(page_image)
                    if len(image_array.shape) == 2: # grayscale
                        image_array = np.stack([image_array] * 3, axis=-1)
                    elif image_array.shape[2] == 4: # rgba
                        image_array = image_array[:, :, :3]
                    
                    detector = FormulaDetector()
                    formulas = detector.detect_formulas(image_array)
                    
                    # process each detected formula region
                    for region_idx, (x, y, w, h) in enumerate(formulas):
                        try:
                            cropped = page_image.crop((x, y, x + w, y + h)) # crop the detected region
                            if cropped.mode != 'L':
                                cropped = cropped.convert('L') # convert to grayscale for pix2tex
                            cropped = ImageOps.expand(cropped, border=20, fill='white')
                            
                            latex_code = MODEL(cropped) # extract latex from the detected region
                            
                            results.append({
                                "page": page_num + 1,
                                "region_index": region_idx,
                                "latex": latex_code.strip(),
                                "region": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                            })
                        except Exception as e:
                            logger.error(f"Error processing formula region on page {page_num + 1}: {e}")
                            results.append({
                                "page": page_num + 1,
                                "region_index": region_idx,
                                "latex": "",
                                "region": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                            })
                    
                    logger.info(f"Detected and processed {len(formulas)} formulas on page {page_num + 1}")
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
        else: # process single image
            pil_image = Image.open(io.BytesIO(contents))
            
            image_array = np.array(pil_image)
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            detector = FormulaDetector()
            formulas = detector.detect_formulas(image_array)
            
            for region_idx, (x, y, w, h) in enumerate(formulas):
                try:
                    cropped = pil_image.crop((x, y, x + w, y + h))
                    if cropped.mode != 'L':
                        cropped = cropped.convert('L')
                    cropped = ImageOps.expand(cropped, border=20, fill='white')
                    
                    latex_code = MODEL(cropped)
                    
                    results.append({
                        "page": 1,
                        "region_index": region_idx,
                        "latex": latex_code.strip(),
                        "region": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                    })
                except Exception as e:
                    logger.error(f"Error processing formula region: {e}")
                    results.append({
                        "page": 1,
                        "region_index": region_idx,
                        "latex": "",
                        "region": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                    })
        
        return JSONResponse({
            "filename": file.filename,
            "total_formulas_found": len(results),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in simple-extract: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/auto-detect-formulas")
async def auto_detect_formulas(request: AutoDetectRequest): # uses the FormulaDetector algorithm to automatically detect formula regions on a page image, returns a list of bounding boxes that the user can accept or reject
    try:
        image_data = request.imageData.split(',')[1] if ',' in request.imageData else request.imageData
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        image_array = np.array(image)
        if len(image_array.shape) == 2:  # grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        
        detector = FormulaDetector()
        formulas = detector.detect_formulas(image_array)
        
        detected_boxes = []
        for idx, (x, y, w, h) in enumerate(formulas):
            detected_boxes.append({
                "id": f"auto_box_{request.pageNum}_{idx}",
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h),
                "isAutoDetected": True,
                "confidence": 0.85 # confidence score from detection
            })
        
        logger.info(f"Auto detected {len(detected_boxes)} formula regions on page {request.pageNum}")
        
        return JSONResponse({
            "page": request.pageNum,
            "detectedBoxes": detected_boxes,
            "totalDetected": len(detected_boxes)
        })
        
    except Exception as e:
        logger.error(f"Error in auto detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/api/extract-boxes")
async def extract_boxes(request: BoxExtractionRequest): # extracts latex from user defined bounding boxes on a page image
    # the frontend posts a json body with imageData (base64 dataurl) and boxes (canvas-space coordinates)
    # the backend decodes the base64 payload and crops per box
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="LaTeX-OCR model not loaded"
        )
    
    try:
        image_data = request.imageData.split(',')[1] if ',' in request.imageData else request.imageData
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        results = []
        
        for box in request.boxes: # calculate pixel coordinates from client sent floats
            x = int(box.x)
            y = int(box.y)
            w = int(box.width)
            h = int(box.height)
            
            try:
                cropped = image.crop((x, y, x + w, y + h))
                if cropped.mode != 'L':
                    cropped = cropped.convert('L')
                cropped = ImageOps.expand(cropped, border=20, fill='white')
                
                latex_code = MODEL(cropped)
                
                results.append({
                    "box_id": box.id,
                    "latex": latex_code.strip(),
                    "region": {"x": x, "y": y, "width": w, "height": h}
                })
            except Exception as e:
                logger.error(f"error processing box {box.id}: {e}")
                results.append({
                    "box_id": box.id,
                    "latex": "",
                    "region": {"x": x, "y": y, "width": w, "height": h}
                })
        
        return JSONResponse({
            "total_boxes": len(request.boxes),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in box extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

@app.get("/results", response_class=HTMLResponse)
async def serve_results():
    try:
        with open("results.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>results.html not found</h1>", status_code=404)

@app.get("/floating-formulas-html", response_class=HTMLResponse)
async def floating_formulas_html(): # returns server rendered floating formulas as html that mathjax can process
    try:
        with open("floating-formulas.txt", "r", encoding="utf-8") as f:
            content = f.read()
        import re
        formula_matches = re.findall(r'\\\[(.*?)\\\]', content, re.DOTALL) # split by \[ and \] to extract all formulas
        
        logger.info(f"total formulas extracted: {len(formula_matches)}")
        for i, formula in enumerate(formula_matches, 1):
            logger.info(f"formula {i}: {formula[:50]}...")
        
        formulas_html = ""
        import random
        
        for idx, formula_content in enumerate(formula_matches, 1):
            # generate random position
            top = random.randint(10, 80)
            left = random.randint(10, 80)
            
            anim_num = (idx - 1) % 4 + 1
            duration = 20 + (idx - 1) * 2
            logger.info(f"floating formula {idx} positioned at {top}% top, {left}% left")
            formulas_html += f"""<div class="floating-formula" style="top: {top}%; left: {left}%; animation: float{anim_num} {duration}s ease-in-out infinite;">\\[{formula_content}\\]</div>
"""
        
        logger.info(f"Generated {len(formula_matches)} floating formulas")
        return formulas_html
    except Exception as e:
        logger.error(f"Error generating floating formulas html: {e}")
        return ""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)