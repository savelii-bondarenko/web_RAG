import pandas as pd
from magic import from_buffer
from mammoth import convert_to_markdown
from pymupdf4llm import to_markdown
import logging

logger = logging.getLogger(__name__)

def read_data(file_path: str) -> str:
    """Read data from files.

    Args:
        file_path (str): Path to the input file.
        Supported formats: txt, docx, pdf, xlsx.

    Returns:
        str: Extracted text from the file.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is not supported.
    """
    file_format = _detect_file_format(file_path)
    readers = {
        "txt": _read_TXT,
        "docx": _read_DOCX,
        "pdf": _read_PDF,
        "xlsx": _read_XLSX,
    }
    try:
        logger.info("Reading data completed")
        return readers[file_format](file_path)
    except KeyError:
        logger.error(f"File format {file_format} not supported.")
        raise ValueError("Unsupported file type")

def _read_PDF(file_path: str) -> str:
    """ Read PDF files.

    Args:
        file_path (str): Path to the input file.
        Supported format: pdf.

    Returns:
        str: Extracted text from the file.
    """
    return to_markdown(file_path, write_images=False)

def _read_DOCX(file_path: str) -> str:
    """ Read docx files.

    Args:
        file_path (str): Path to the input file.
        Supported format: docx.

    Returns:
        str: Extracted text from the file.
    """

    with open(file_path, "rb") as file:
        return convert_to_markdown(file).value

def _read_TXT(file_path: str) -> str:
    """ Read txt files.

    Args:
        file_path (str): Path to the input file.
        Supported format: txt.

    Returns:
        str: Extracted text from the file.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def _read_XLSX(file_path: str) -> str:
    """ Read xlsx files.

    Args:
        file_path (str): Path to the input file.
        Supported format: xlsx.

    Returns:
        str: Extracted text from the file.
    """
    df = pd.read_excel(file_path)
    return df.to_markdown(index=False)

def _detect_file_format(file_path: str) -> str:
    """Detects file format.

    Args:
        file_path (str): Path to the input file.
        Supported formats: txt, doc, pdf, xlsx.

    Returns:
        str: Extracted text from the file.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    try:
        with open(file_path, "rb") as f:
            mime = from_buffer(f.read(2048), mime=True)
        if "pdf" in mime:
            return "pdf"
        elif "vnd.openxmlformats-officedocument.wordprocessingml.document" in mime:
            return "docx"
        elif "officedocument.spreadsheetml.sheet" in mime:
            return "xlsx"
        elif "text" in mime:
            return "txt"
        else:
            return "unknown"
    except FileNotFoundError:
        raise FileNotFoundError("File does not exist")



