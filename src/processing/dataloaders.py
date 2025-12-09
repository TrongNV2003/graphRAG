import json
from loguru import logger
from langchain_community.document_loaders import WikipediaLoader


class DataLoader:
    def __call__(self, documents: str = "Elizabeth I", load_max_docs: int = 10) -> list:
        return self.load(documents, load_max_docs)
    
    def load(self, documents: str = "Elizabeth I", load_max_docs: int = 10) -> list:
        """
        Load documents from Wikipedia based on the query.
        
        Args:
            documents (str): The search query for Wikipedia.
            load_max_docs (int): Maximum number of documents to load.
        Returns:
            list: A list of documents with their content.
        """
        
        if documents is not None:
            logger.info(f"Searching Wikipedia for: {documents}")
            
            loader = WikipediaLoader(query=documents, load_max_docs=load_max_docs)
            documents = loader.load()
            docs = [
                {
                    "id": i,
                    "content": doc.page_content,
                }
                for i, doc in enumerate(documents)
            ]
            
            return docs


if __name__ == "__main__":
    loader = DataLoader()
    documents = loader(wiki_query="Elizabeth I", load_max_docs=10)
    
    with open("dump/elizabeth_i.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)
