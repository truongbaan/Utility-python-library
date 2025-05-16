from collections import Counter
from haystack import Document
from haystack.components.readers import ExtractiveReader
from typing import List, Optional
import os
from .pdf_docx_reader import PDF_DOCX_Reader

#smaller model: deepset/roberta-base-squad2

class DocumentFilter:
    def __init__(self, model_name="deepset/tinyroberta-squad2", path : Optional[str] = None, threshold : float = 0.4, max_per_doc : int = 2, top_answer : int = 4):
        self.reader = ExtractiveReader(model=model_name)
        self.reader.warm_up()
        self.threshold = threshold
        self.max_per_doc = max_per_doc
        self.top_k = top_answer
        self._documents: List[Document] = [] #init var to hold document
        if path is None:
            path = os.path.dirname(os.path.abspath(__file__))
        self.__init_documents(path)
    
    @property
    def documents(self):
        return self._documents
       
    def search_document(self, prompt : str = None) -> List:
        result = self.reader.run(query=prompt, documents=self._documents, top_k=self.top_k)

        seen_texts = set()
        counts = Counter()
        filtered_answers = []

        for ans in result["answers"]:
            if ans.score < self.threshold:
                continue
            if ans.document is None:
                continue
            text = ans.document.content
            doc_id = ans.document.id

            if counts[doc_id] >= self.max_per_doc:
                continue
            if text in seen_texts:
                continue

            filtered_answers.append((text, ans.score, doc_id))
            seen_texts.add(text)
            counts[doc_id] += 1

        #contain text, score, and doc.id
        return filtered_answers #return list of document with ranking score that > threshold

    def __init_documents(self, directory : str = ""):
        #get from here
        self._documents.clear()
        pdf_urls, docx_urls =self.__collect_file_paths(directory)
        
        reader = PDF_DOCX_Reader() #init reader
        
        #extract from docx
        for doc in docx_urls:
            text = reader.extract_ordered_text(doc)
            self._documents.append(Document(content=text))
            
        #extract from pdf
        for pdf in pdf_urls:
            text = reader.extract_ordered_text(pdf)
            self._documents.append(Document(content=text))
    
    def __collect_file_paths(self, directory):
        pdf_urls = []
        docx_urls = []

        # Walk through all directories and files
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file).replace('\\', '/')  # normalize path
                if file.lower().endswith('.pdf'):
                    pdf_urls.append(file_path)
                elif file.lower().endswith('.docx'):
                    docx_urls.append(file_path)

        return pdf_urls, docx_urls