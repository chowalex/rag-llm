import os
import requests
from io import BytesIO
from PyPDF2 import PdfReader

class PDFParser:
  def __init__(self, url):
    self.url = url

    response = requests.get(self.url)
    response.raise_for_status()
    self.pdf = PdfReader(BytesIO(response.content))

  def parse(self):
    """Returns the full text of the PDF as a single string."""
    contents = []
    for page in self.pdf.pages:
      contents.append(page.extract_text())
    
    return " ".join(contents)

  def metadata(self):
    """Extract metadata from the PDF."""
    if not self.pdf:
      self.pdf = PdfReader(content_bytes)
    
    return self.pdf.metadata