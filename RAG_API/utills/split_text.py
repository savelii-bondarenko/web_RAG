from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_text(text: str,
               chunk_size: int = 512,
               chunk_overlap: int = 100) ->  list[Document]:
    """Split text into chunks using Markdown structure + Recursive fallback.

    Args:
        text (str): Raw text to split.
        chunk_size (int): Max size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list[Document]: List of LangChain Documents with metadata.
    """
    headers_on_split = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
    markdown_splitter  = MarkdownHeaderTextSplitter(headers_on_split, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(md_header_splits)

