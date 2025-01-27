from PyPDF2 import PdfReader

class PDFParser:
  def __init__(self, path):
    self.path = path

  def parse(self):
    contents = []
    with open(self.path,'rb') as f:
      pdf = PdfReader(f)
      for page in pdf.pages:
        contents.append(page.extract_text())
    return " ".join(contents)


parser = PDFParser("pdf/tiktok.pdf")
text = parser.parse()
print(text)
