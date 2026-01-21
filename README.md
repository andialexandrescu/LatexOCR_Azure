# LatexOCR_Azure

## Installation

```bash
winget install --exact --id Microsoft.AzureCLI
winget install microsoft.azd
az login
```

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