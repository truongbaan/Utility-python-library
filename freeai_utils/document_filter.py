from collections import Counter
from haystack import Document
from haystack.components.readers import ExtractiveReader
from haystack.utils.device import ComponentDevice
from typing import List, Optional
import os
from .pdf_docx_reader import PDF_DOCX_Reader
from freeai_utils.log_set_up import setup_logging

#smaller model: deepset/roberta-base-squad2
class DocumentFilter:
    
    __slots__ = ("_reader", "threshold", "max_per_doc", "top_k", "_documents", "_initialized", "logger")
    
    _initialized: bool
    _reader: ExtractiveReader
    threshold: float
    max_per_doc: int
    top_k: int
    _documents: List[Document]
    
    def __init__(self, model_name="deepset/tinyroberta-squad2", path : Optional[str] = None, threshold : float = 0.4, max_per_doc : int = 2, top_answer : int = 4, device: str = "cuda", auto_init : bool = True) -> None:
        #check type first
        self.__enforce_type(threshold, float, "threshold")
        self.__enforce_type(max_per_doc, int, "max_per_doc")
        self.__enforce_type(top_answer, int, "top_answer")
        self.__enforce_type(auto_init, bool, "auto_init")
        self.logger = setup_logging(self.__class__.__name__)
        self.logger.propagate = False  # Prevent propagation to the root logger
        
        # init not lock
        super().__setattr__("_initialized", False)
        # set core reader
        try:
            device_obj = ComponentDevice.from_str(device)
            super().__setattr__("_reader", ExtractiveReader(model=model_name, device=device_obj))
            self._reader.model.eval()
            self._reader.warm_up()
            self.logger.info(f"Successfully runs on {device}")
        except Exception as e:
            self.logger.error(f"Fail to run on {device}")
            if device != "cpu":
                self.logger.info("Trying on cpu instead")
                cpu_device_obj = ComponentDevice.from_str("cpu")
                super().__setattr__("_reader", ExtractiveReader(model=model_name, device=cpu_device_obj))
                self._reader.warm_up()
                self.logger.info(f"Successfully runs on {device}")
                
        # lock down, reader can't be modified
        super().__setattr__("_initialized", True)
    
        self.threshold = threshold
        self.max_per_doc = max_per_doc
        self.top_k = top_answer
        self._documents: List[Document] = [] #init var to hold document
        if path is None:
            path = os.getcwd()
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path '{path}' does not exist.")
        if auto_init:
            self.__init_documents(path) #init documents from the path
        self.logger.info(f"Initialize successfully at path {path}")
    
    @property
    def reader(self):
        return self._reader
    
    @property
    def documents(self):
        return self._documents
       
    def __setattr__(self, name, value):
        # once initialized, prevent changing core internals
        if getattr(self, "_initialized", False) and name in ("_reader"):
            raise AttributeError(f"Cannot reassign '{name}' after initialization")
        super().__setattr__(name, value)
    
    def search_document(self, prompt : str = None) -> List:
        self.__enforce_type(prompt, str, "prompt") #check type before start
        
        result = self._reader.run(query=prompt, documents=self._documents, top_k=self.top_k)

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

    def __init_documents(self, directory : str = "") -> None:
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
    
    def __enforce_type(self, value, expected_type, arg_name):
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_type.__name__}, but received {type(value).__name__}")