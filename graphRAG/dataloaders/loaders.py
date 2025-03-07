import json
import pymupdf
from loguru import logger
from typing import Literal
from langchain_core.documents import Document
from langchain_community.document_loaders import WikipediaLoader

class DataLoader:
    def __init__(self, format_type: Literal["wiki", "pdf", "json"]):
        self.format_type = format_type

    def load(self, file_path: str = None, save_to: bool = False) -> list:
        if self.format_type == "wiki":
            query_wiki = input("Enter the query to search on Wikipedia: ")
            logger.info(f"Searching Wikipedia for: {query_wiki}")
            
            loader = WikipediaLoader(query=query_wiki)
            documents = loader.load()
            
            if save_to:
                query_key = query_wiki.replace(" ", "_").lower()
                docs = [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ]
                with open(f"data/format_data/{query_key}.json", "w", encoding="utf-8") as f:
                    json.dump(docs, f, ensure_ascii=False, indent=4)
                logger.info(f"Saved Wikipedia data")

            return documents
        
        elif self.format_type == "pdf":
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
                with open(f"data/format_data/{file_name}.json", "w", encoding="utf-8") as f:
                    json.dump(docs, f, ensure_ascii=False, indent=4)
                logger.info(f"Saved PDF data")

            return documents
        
        elif self.format_type == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                documents = [
                    Document(page_content=data["page_content"], metadata=data["metadata"]) for data in json_data
                    ]
                
                logger.info(f"JSON documents loaded")
                return documents

        else:
            raise ValueError(f"Unsupported this format type: {self.format_type}")
        
    
if __name__ == "__main__":
    loader = DataLoader(format_type="pdf")
    raw_docs = loader.load(file_path="data/raw_data/sample.pdf", save_to=False)
    print(f"Loaded {len(raw_docs)} documents")
    for doc in raw_docs:
        print(doc.page_content)
