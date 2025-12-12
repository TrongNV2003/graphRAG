import json
from loguru import logger
from langchain_community.document_loaders import WikipediaLoader


class DataLoader:
    def __call__(self, keyword: str = "Elizabeth I", load_max_docs: int = 10) -> list:
        return self.load(keyword, load_max_docs)
    
    def load(self, keyword: str = "Elizabeth I", load_max_docs: int = 10) -> list:
        """
        Load documents from Wikipedia based on the query.
        
        Args:
            keyword (str): The keyword to search query in Wikipedia.
            load_max_docs (int): Maximum number of documents to load.
        Returns:
            list: A list of documents with their content.
        """
        
        if keyword is not None:
            logger.info(f"Searching Wikipedia for: {keyword}")
            
            loader = WikipediaLoader(query=keyword, load_max_docs=load_max_docs)
            keyword = loader.load()
            docs = [
                {
                    "id": i,
                    "content": doc.page_content,
                }
                for i, doc in enumerate(keyword)
            ]
            
            return docs
