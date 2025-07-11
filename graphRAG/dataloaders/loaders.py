import os
import json
import pymupdf
from loguru import logger
from typing import Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import WikipediaLoader

class DataLoader:
    def __init__(self):
        self.output_dir = "graphRAG/data/format_data"

    def load(self, file_path: Optional[str] = None, save_to: bool = False) -> list:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if file_path is None:
            logger.info("Not found file, crawl from Wikipedia")
            query_wiki = input("Enter the query to search on Wikipedia: ")
            logger.info(f"Searching Wikipedia for: {query_wiki}")
            
            loader = WikipediaLoader(query=query_wiki, load_max_docs=10)
            documents = loader.load()
            
            if save_to:
                query_key = query_wiki.replace(" ", "_").replace("-", "_").lower()
                docs = [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ]
                with open(f"{self.output_dir}/{query_key}.json", "w", encoding="utf-8") as f:
                    json.dump(docs, f, ensure_ascii=False, indent=4)
                logger.info(f"Saved Wikipedia data")

            return documents

        if file_path.endswith(".pdf"):
            logger.info(f"Loading PDF file: {file_path}")
            documents = []
            with pymupdf.open(file_path, filetype="pdf") as pdf:
                for page_num, page in enumerate(pdf, 1):
                    text = page.get_text("text")
                    if text.strip():
                        metadata = {
                            "source": file_path,
                            "page": page_num,
                            "title": pdf.metadata.get("title", "Unknown") if pdf.metadata else "Unknown"
                        }
                        documents.append(Document(page_content=text, metadata=metadata))
            
            if save_to:
                file_name = file_path.split("/")[-1].replace(".pdf", "").lower()
                docs = [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ]
                with open(f"{self.output_dir}/{file_name}.json", "w", encoding="utf-8") as f:
                    json.dump(docs, f, ensure_ascii=False, indent=4)
                logger.info(f"Saved PDF data")

            return documents
        
        elif file_path.endswith('.json'):
            logger.info(f"Loading JSON file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                documents = [
                    Document(page_content=data["page_content"], metadata=data["metadata"]) for data in json_data
                    ]
                
                logger.info(f"JSON documents loaded")
                return documents
        
        else:
            logger.error(f"Unsupported file type: {file_path}")
            raise ValueError(f"Unsupported file type: {file_path}")
        

if __name__ == "__main__":
    loader = DataLoader()
    file_path = "graphRAG/data/raw_data/sample.pdf"
    raw_docs = loader.load(save_to=False)
    print(f"Loaded {len(raw_docs)} documents")
    for doc in raw_docs:
        print(doc.page_content)
