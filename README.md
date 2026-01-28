# LatexOCR_Azure

## Demo video
https://youtu.be/MDS35h-oiCY

## Installation

```bash
python app.py

# open browser
http://localhost:8000
```

## Demo

Running the full pipeline:
```bash
curl.exe -X POST http://localhost:8000/api/simple-extract `
  -F "file=@test_descriptive_text.pdf" `
  -H "accept: application/json"
```

## Docker + Azure Container App


```bash
docker build -t latexocr:latest .

# one time testing command since the image is going to be public
docker run -p 8000:8000 -e DOCUMENT_INTELLIGENCE_ENDPOINT="" -e DOCUMENT_INTELLIGENCE_SUBSCRIPTION_KEY="" latexocr:latest

# access at http://localhost:8000
```

```bash
docker tag latexocr:latest andialexandrescu/latexocr:latest
docker push andialexandrescu/latexocr:latest
```

```bash
docker pull andialexandrescu/latexocr:latest
docker run -p 8000:8000 andialexandrescu/latexocr:latest
```

When creating the container app, add environment variables:

DOCUMENT_INTELLIGENCE_ENDPOINT = ...
DOCUMENT_INTELLIGENCE_SUBSCRIPTION_KEY = ...

Then enable external ingress on port 8000

Updating docker image + container app
```bash
 $TAG="2026-01-23-1"
docker build -t andialexandrescu/latexocr:$TAG .

docker push andialexandrescu/latexocr:$TAG

$RESOURCE_GROUP="latex-ocr"
$CONTAINER_APP_NAME="latex-ocr-container-app"
az containerapp update `
  --name $CONTAINER_APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --image "andialexandrescu/latexocr:$TAG"
```
