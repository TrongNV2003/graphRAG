import json
from loguru import logger
from langchain_community.document_loaders import WikipediaLoader


class DataLoader:
    def __call__(self, wiki_query: str = "Elizabeth I", load_max_docs: int = 10) -> list:
        return self.load(wiki_query, load_max_docs)
    
    def load(self, wiki_query: str = "Elizabeth I", load_max_docs: int = 10) -> list:
        if wiki_query is not None:
            logger.info(f"Searching Wikipedia for: {wiki_query}")
            
            loader = WikipediaLoader(query=wiki_query, load_max_docs=load_max_docs)
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

