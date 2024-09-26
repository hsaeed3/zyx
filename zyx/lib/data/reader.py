from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from loguru import logger
from pathlib import Path
import mimetypes
import json
from typing import Dict, List, Optional, Literal, Union
from contextlib import suppress
import xml.etree.ElementTree as ET
import PyPDF2
import docx
import openpyxl
import csv


from ..types.document import Document


OutputType = Literal["markdown", "text", "json"]


def read(
    path: Union[str, Path],
    target: OutputType = "text",
    verbose: bool = False
) -> Union[Document, List[Document]]:
    """
    Reads either a file or a directory and returns the content.
    """
    path = Path(path)
    if path.is_file():
        return _read_single_file(path, target, verbose)
    elif path.is_dir():
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(_read_single_file, file, target, verbose)
                       for file in path.glob('*') if file.is_file()]
            results = [future.result() for future in futures]
        return [result for result in results if result is not None]
    else:
        raise ValueError(f"Invalid path: {path}")

def _read_single_file(
    path: Union[str, Path],
    target: OutputType = "text",
    verbose: bool = False
) -> Optional[Document]:
    """
    Reads a single file and returns its content based on the target format.
    """
    path = Path(path)
    mime_type, _ = mimetypes.guess_type(str(path))

    try:
        content = None
        match mime_type:
            case 'application/pdf':
                content = _read_pdf(path)
            case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                content = _read_docx(path)
            case 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                content = _read_xlsx(path)
            case 'text/csv':
                content = _read_csv(path)
            case _ if mime_type and mime_type.startswith('text/'):
                content = _read_text(path)
            case 'application/xml':
                content = _read_xml(path)
            case _:
                content = _read_binary(path)

        if verbose:
            logger.info(f"Read {path.name} as {mime_type or 'BINARY'}")

        if target == "markdown":
            content = _to_markdown(content)
        elif target == "json":
            content = json.dumps(content)

        metadata = {
            "file_name": path.name,
            "file_type": mime_type or "unknown",
            "file_size": path.stat().st_size
        }

        return Document(content=content, metadata=metadata)

    except Exception as e:
        if verbose:
            logger.error(f"Error reading file {path}: {str(e)}")
        return None

def _read_pdf(path: Path) -> str:
    """
    Extracts text from a PDF, including proper formatting for tables and paragraphs.
    """
    try:
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted_text = page.extract_text() or ""
                # Additional handling for simple table formatting
                # We'll assume columns are separated by enough space to infer table-like structures
                text += _format_pdf_text(extracted_text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF {path}: {str(e)}")
        return ""

def _format_pdf_text(extracted_text: str) -> str:
    """
    Post-process PDF extracted text to ensure proper formatting for tables, paragraphs, etc.
    """
    # Insert custom logic to handle common cases like columns and tables, for example:
    # - Replace multiple spaces with a ' | ' for column-separated data in tables
    # - Handle paragraph breaks or other structures

    # This is a placeholder and can be adjusted based on the specifics of the PDF content
    lines = extracted_text.split("\n")
    formatted_lines = []
    for line in lines:
        if "  " in line:  # Simple heuristic for columns in tables
            formatted_lines.append(" | ".join(line.split()))
        else:
            formatted_lines.append(line)
    return "\n".join(formatted_lines)

def _read_docx(path: Path) -> str:
    """
    Reads text from a DOCX file, including tables.
    """
    try:
        doc = docx.Document(path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        
        # Handle tables in the DOCX file
        for table in doc.tables:
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                full_text.append(" | ".join(row_data))  # Simple table formatting with '|'

        return "\n".join(full_text)
    except Exception as e:
        logger.error(f"Error reading DOCX {path}: {str(e)}")
        return ""

def _read_xlsx(path: Path) -> List[List[str]]:
    """
    Reads an XLSX file and returns its data, including table formatting.
    """
    try:
        with openpyxl.load_workbook(path, read_only=True) as workbook:
            sheet = workbook.active
            data = []
            for row in sheet.iter_rows():
                row_data = [str(cell.value) if cell.value is not None else "" for cell in row]
                data.append(row_data)
            return data
    except Exception as e:
        logger.error(f"Error reading XLSX {path}: {str(e)}")
        return []

def _read_csv(path: Path) -> List[List[str]]:
    """
    Reads CSV data.
    """
    try:
        with open(path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            return list(reader)
    except Exception as e:
        logger.error(f"Error reading CSV {path}: {str(e)}")
        return []

def _read_text(path: Path) -> str:
    """
    Reads plain text files.
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading text file {path}: {str(e)}")
        return ""

def _read_xml(path: Path) -> Dict:
    """
    Reads XML files and converts them to a dictionary.
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        return _element_to_dict(root)
    except Exception as e:
        logger.error(f"Error reading XML {path}: {str(e)}")
        return {}

def _read_binary(path: Path) -> str:
    return f"Binary file: {path.name}"

def _element_to_dict(element):
    """
    Recursively converts XML elements to a dictionary.
    """
    result = {}
    for child in element:
        if len(child) == 0:
            result[child.tag] = child.text
        else:
            result[child.tag] = _element_to_dict(child)
    return result

def _to_markdown(content: Union[str, List, Dict]) -> str:
    """
    Converts the content into markdown format.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join([" | ".join(row) for row in content])
    elif isinstance(content, dict):
        return "\n".join([f"**{k}**: {v}" for k, v in content.items()])
    else:
        return str(content)
