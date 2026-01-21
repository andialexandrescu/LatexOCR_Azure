import cv2
import numpy as np
import fitz
import os
import sys
import re

class FormulaDetector: # used to detect and crop mathematical formulas from pdf pages using cv2
    def __init__(self):
        self.pdf_doc = None
    def pdf_to_images(self, pdf_path, dpi=300):
        print(f"Converting pdf to images at {dpi} dpi")
        zoom = dpi / 72  # convert dpi to zoom factor used for rendering
        images = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n) # convert pixmap bytes to numpy array
                
                # meant to normalize channel order to rgb for opencv pipeline
                if pix.n == 4:  # handle rgba pages
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
                elif pix.n == 3:  # handle rgb pages from pdf
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                
                images.append(img_data)
            doc.close()
        except Exception as e:
            print(f"Error reading pdf: {e}")
            return []
        
        return images
    
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert to grayscale to simplify downstream thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # apply adaptive thresholding to separate foreground strokes
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21) # denoise to remove speckle noise before contour detection
        
        return denoised
    
    def detect_formulas(self, image):
        processed = self.preprocess_image(image)
        inverted = cv2.bitwise_not(processed)
        
        # use moderate kernels to connect fragmented formula components without merging distinct formulas
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 16))
        
        dilated_h = cv2.dilate(inverted, kernel_horizontal, iterations=2)
        dilated = cv2.dilate(dilated_h, kernel_vertical, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours that bound candidate formula regions
        
        formulas = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if (area > 800 and area < 500000 and h > 25 and w > 35 and aspect_ratio < 50): # keep regions within plausible size and aspect thresholds for formulas
                formulas.append((x, y, w, h))
        
        formulas = self.merge_nearby_formulas(formulas, horizontal_gap=18, vertical_gap=10)
        formulas.sort(key=lambda r: (r[1], r[0]))
        
        return formulas
    
    def filter_text_regions(self, formulas, page_obj, img_shape): # filter out regions that are just regular text, not formulas
        filtered = []
        zoom = 300 / 72 # dpi conversion matching rendering scale
        
        for region in formulas:
            x, y, w, h = region
            # convert pixel coordinates to pdf coordinates
            pdf_x = x / zoom
            pdf_y = y / zoom
            pdf_w = w / zoom
            pdf_h = h / zoom
            rect = fitz.Rect(pdf_x, pdf_y, pdf_x + pdf_w, pdf_y + pdf_h) # extract text within the region to decide if it's math or regular text
            text = page_obj.get_text("text", clip=rect).strip()
            
            if not text:
                filtered.append(region)
                continue
            
            is_formula = self.is_formula_text(text, w, h)
            
            if is_formula:
                filtered.append(region)
        
        return filtered
    
    def is_formula_text(self, text, width, height): # determine if extracted text is likely a formula vs regular text
        math_score = 0 # count mathematical indicators to score formula likelihood
        text_stripped = text.strip()
        
        # descriptive keywords that indicate captions/explanations rather than formulas
        descriptive_keywords = ['limit of', 'approaches', 'relationship between', 'property', 'as x approaches', 'time domain', 'function of', 'where', 'theorem', 'definition', 'proof', 'such that', 'given by', 'function', 'relationship', 'formula', 'equation', 'between', 'series', 'transform', 'lemma', 'corollary', 'domain', 'range']
        
        # check for operators while distinguishing hyphens from minus signs
        has_equals = '=' in text
        has_plus = '+' in text
        has_mult = '*' in text or '×' in text
        has_div = '/' in text or '÷' in text
        # only count minus if separated by spaces or boundaries to avoid hyphenated words
        has_minus = bool(re.search(r'\s-\s|^-\s|\s-$|^-$', text))
        
        # reject pure labels that lack operators or math symbols
        has_operators = has_equals or has_plus or has_mult or has_div or has_minus
        has_math_symbols = any(sym in text for sym in ['∫', '∑', '∏', '√', '∂', '∇', '∞', '→', '←', '↔', 'lim', '≈', '≠', '≤', '≥'])
        
        if not has_operators and not has_math_symbols: # pure text with no math notation is likely a caption
            return False
        
        lines = text.split('\n')
        descriptive_lines = 0
        formula_lines = 0
        
        for line in lines: # check if region contains descriptive sentences mixed with formulas using pattern cues
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            if len(line_stripped) < 3:
                continue
            
            # long lines with mostly alphabetic characters indicate descriptive text
            alpha_count = sum(c.isalpha() for c in line_stripped)
            total_chars = len(line_stripped.replace(' ', ''))
            alpha_ratio = alpha_count / max(total_chars, 1)
            
            looks_like_sentence = (line_stripped[0].isupper() if line_stripped else False) and len(line_stripped) > 20 # starts with capital, ends with no punctuation
            
            math_char_count = sum(line.count(c) for c in ['=', '+', '∫', '∑', '√', '∂', '∇', '÷', '×'])
            has_minimal_math = math_char_count <= 1 and len(line_stripped) > 15 # very few math symbols relative to length
            
            word_count = len(line_lower.split())
            is_wordy = word_count >= 5 and alpha_ratio > 0.7 # high word count with spaces
            
            # check for descriptive keywords
            has_descriptive_phrase = any(pattern in line_lower for pattern in descriptive_keywords)
            
            is_descriptive = (
                (alpha_ratio > 0.75 and has_minimal_math) or # mostly letters, few math symbols
                (looks_like_sentence and is_wordy) or # sentence structure
                (has_descriptive_phrase and math_char_count <= 2) or # known pattern with little math
                (word_count >= 6 and alpha_ratio > 0.65) # many words, mostly alphabetic
            ) # either descriptive text or formula
            
            has_formula_indicators = any(sym in line for sym in ['=', 'lim', '∫', '∑', '√', '∂'])
            
            if is_descriptive:
                descriptive_lines += 1
            elif has_formula_indicators:
                formula_lines += 1
        
        if descriptive_lines >= 2: # if we have multiple descriptive lines, this is mixed content, need to reject it
            return False
        if descriptive_lines >= 1 and len(text) > 100: # if we have one descriptive line and it's long text, also reject
            return False
        
        math_symbols = ['=', '≈', '≠', '≤', '≥', '±', '×', '÷', '∫', '∑', '∏', '√', '∂', '∇', '∞', '→', '←', '↔', 'α', 'β', 'γ', 'θ', 'λ', 'π', 'σ', 'lim']
        for symbol in math_symbols: # check for math symbols and boost score
            if symbol in text:
                math_score += 3
        
        # strong indicators for formulas
        if text_stripped.startswith('(') or text_stripped.startswith('['): # starts with parenthesis
            math_score += 4
        if text_stripped.startswith(('+', '×', '÷', '*', '/')) or re.match(r'^-\s', text_stripped): # starts with operator
            math_score += 5
        
        if re.search(r'[a-zA-Z]\s*=', text): # variable assignment
            math_score += 5
        if re.search(r'\d+\s*[+\-*/÷×]\s*\d+', text): # arithmetic
            math_score += 3
        if re.search(r'[a-zA-Z]\^', text) or '^' in text: # exponents
            math_score += 4
        if re.search(r'[a-zA-Z]_', text) or '_' in text: # subscripts
            math_score += 3
        if '/' in text and len(text) < 50: # fractions
            math_score += 2
        if re.search(r'\([^)]+\)', text): # parentheses with content
            math_score += 2
        
        # count operator density (avoid hyphens in compound words)
        operator_count = (text.count('=') + text.count('+') + text.count('*') + text.count('/') + len(re.findall(r'\s-\s|^-\s|\s-$', text)))
        if operator_count >= 2: # multiple operators
            math_score += 3
        elif operator_count == 1: # single operator
            math_score += 1
        
        text_lower = text.lower() # penalty for regular text patterns
        
        # check if it is mostly descriptive text
        words = text_lower.split()
        if len(words) >= 2:
            # if it has descriptive keywords and low operator count, likely just a label
            if any(keyword in text_lower for keyword in descriptive_keywords):
                if operator_count <= 1 and not has_math_symbols:
                    return False
                # even with operators, if it's very wordy, probably a description
                if len(words) >= 4 and operator_count <= 1:
                    return False
        
        word_count_total = len(text.split())
        if word_count_total > 15:
            math_score -= 5 # long sentences are usually not formulas
        common_words = ['the', 'and', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'this', 'that', 'where', 'when', 'as']
        word_count = sum(1 for word in common_words if word in text_lower.split())
        if word_count >= 3: # only penalize if multiple common words
            math_score -= word_count * 2
        aspect_ratio = width / float(height) if height > 0 else 0
        if aspect_ratio > 10 and len(text) > 30:
            math_score -= 5 # high aspect ratio with lots of text suggests a paragraph
        if len(text) < 50 and (has_operators or has_math_symbols):
            math_score += 2 # compact region with symbols is likely a formula block
        
        return math_score >= 2 # decision threshold tuned low to catch more formulas
    
    def merge_nearby_formulas(self, formulas, horizontal_gap=40, vertical_gap=15):
        if not formulas:
            return []
        
        formulas = sorted(formulas, key=lambda r: (r[1], r[0]))
        
        merged = []
        current = list(formulas[0])
        for formula in formulas[1:]:
            x, y, w, h = formula
            cx, cy, cw, ch = current

            horizontal_close = (x <= cx + cw + horizontal_gap) and (x + w >= cx - horizontal_gap) # check if formulas are close enough to merge
            vertical_close = abs(y - (cy + ch)) < vertical_gap # vertical proximity (an example is a stacked numerator/ denominator)
            
            if horizontal_close and vertical_close: # merge formulas into a combined bounding box
                new_x = min(cx, x)
                new_y = min(cy, y)
                new_w = max(cx + cw, x + w) - new_x
                new_h = max(cy + ch, y + h) - new_y
                current = [new_x, new_y, new_w, new_h]
            else:
                merged.append(tuple(current))
                current = list(formula)
        
        merged.append(tuple(current))
        return merged
    
    def crop_formula(self, image, region, padding=10):
        x, y, w, h = region
        # add padding so crops include breathing room around symbols
        x = max(0, x - padding)
        y = max(0, y - padding)
        x_end = min(image.shape[1], x + w + 2 * padding)
        y_end = min(image.shape[0], y + h + 2 * padding)
        
        return image[y:y_end, x:x_end], (x, y, x_end - x, y_end - y)
    
    def process_pdf(self, pdf_path, output_dir="extracted_formulas"):
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        page_images = self.pdf_to_images(pdf_path) # convert pdf pages to images for processing
        
        for page_num, page_image in enumerate(page_images, 1): # process each page image
            print(f"\nPage {page_num}:")
            
            formulas = self.detect_formulas(page_image) # detect formulas on the current page
            print(f"\tFound {len(formulas)} formula regions")
            
            for idx, region in enumerate(formulas, 1): # crop and save each detected formula region
                cropped, new_region = self.crop_formula(page_image, region)
                
                filename = f"page{page_num}_formula{idx}.png"
                filepath = os.path.join(output_dir, filename)
                
                cv2.imwrite(
                    filepath,
                    cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                )
                
                h, w = cropped.shape[:2]
                print(f"\t\tFormula {idx}: {w}x{h}px -> {filename}")
                
                results.append({
                    'page': page_num,
                    'formula_num': idx,
                    'filename': filename,
                    'filepath': filepath,
                    'size': (w, h),
                    'region': region
                })
        
        return results

def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "simple_test.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return
    
    detector = FormulaDetector()
    
    print(f"Processing: {pdf_path}\n")
    
    results = detector.process_pdf(pdf_path)
    
    print(f"\nExtraction complete")
    print(f"Total formulas found: {len(results)}")
    
    if results:
        print("\nExtracted formulas:")
        for result in results:
            print(f"\t{result['filename']:30s} {result['size'][0]:4d}x{result['size'][1]:4d}px")

if __name__ == "__main__":
    main()
