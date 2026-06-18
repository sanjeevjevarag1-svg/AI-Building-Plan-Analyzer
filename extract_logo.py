import fitz  # PyMuPDF
from PIL import Image
import io

print("Converting PDF to image...")

# Open PDF
pdf_document = fitz.open('Screenshot 2026-06-01 172412.pdf')

# Get first page
page = pdf_document[0]

# Render page to image (high quality)
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

# Save as PNG
pix.save('kle_logo.png')

print("✅ Logo saved as kle_logo.png")

pdf_document.close()