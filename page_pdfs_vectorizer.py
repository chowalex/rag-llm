import os
import re
import requests
import urllib.parse
from bs4 import BeautifulSoup
from pdf_vectorizer import PDFVectorizer
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
import pandas as pd

class Link:
  def __init__(self, base_url, link, remove_fragments=True):
    self.text = link.text
    self.href = link.get('href')
    self.title = link.get('title')
    
    parsed_href = urllib.parse.urlparse(self.href)
    if parsed_href.scheme and parsed_href.netloc:
      self.full_url = self.href
    else:      
      parsed_base = urllib.parse.urlparse(base_url)
      self.full_url = urllib.parse.urlunparse((parsed_base.scheme, parsed_base.netloc, self.href, "", "", ""))
      if remove_fragments:
        reparsed = urllib.parse.urlparse(self.full_url)
        self.full_url = reparsed._replace(fragment="").geturl()

class PageLinksParser:
  def __init__(self, url):
    self.url = url
    response = requests.get(url)
    self.soup = BeautifulSoup(response.text, 'html.parser')
    self.processed = set()

  def get_links(self, link_re = '.*'):
    """Returns a list of links with hrefs that match the given regular expression"""
    links = []
    for l in self.soup.find_all('a', href=True):
      link = Link(self.url, l)
      if re.match(link_re, link.href):
        links.append(link)
      
    return links

class PdfsVectorizer:
  def __init__(self, pdf_links):
    self.pdf_links = pdf_links
    self.processed = set()
    
  def vectorize(self, persist_path, serializer="parquet", limit=200, dry_run=False):    
    api_key = os.getenv("OPENAI_API_KEY")
    if not dry_run:
      vectorstore = SKLearnVectorStore(
            embedding=OpenAIEmbeddings(openai_api_key=api_key),
            persist_path=persist_path,
            serializer=serializer)

    done = 0
    for pdf_link in self.pdf_links:
      if done >= limit:
        break
            
      metadata = {
        "title": pdf_link.title,
        "url": pdf_link.full_url,
        "text": pdf_link.text
      }
      done += 1

      if pdf_link.full_url in self.processed:
        continue
      self.processed.add(pdf_link.full_url)
            
      print("Processing: ", pdf_link.full_url)
           
      if not dry_run:
        pdf_vec = PDFVectorizer(pdf_link.full_url, metadata)
        pdf_vec.vectorize(vectorstore)

    if not dry_run:
      vectorstore.persist()      
