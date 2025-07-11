from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextPreprocessor:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunking(self, documents: list):
        return self.splitter.split_documents(documents)