import os
import re
from io import BytesIO

import PyPDF2
import litellm
import pymupdf


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        return "".join(page.extract_text() for page in reader.pages)


def get_pdf_title(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        return reader.metadata.get("/Title", "")


def get_text_of_pages(pdf_path, start_page, end_page, tag=True):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        texts = []
        for page_idx in range(start_page - 1, end_page):
            page_text = reader.pages[page_idx].extract_text()
            if tag:
                texts.append(f"<physical_index_{page_idx + 1}>\n{page_text}\n<physical_index_{page_idx + 1}>\n")
            else:
                texts.append(page_text)
        return "".join(texts)


def get_first_start_page_from_text(text):
    match = re.search(r"<physical_index_(\d+)>", text)
    return int(match.group(1)) if match else None


def get_last_start_page_from_text(text):
    matches = re.findall(r"<physical_index_(\d+)>", text)
    return int(matches[-1]) if matches else None


def sanitize_filename(filename, replacement="-"):
    return re.sub(r'[<>:"/\\|?*]', replacement, filename)


def get_pdf_name(pdf_path):
    if isinstance(pdf_path, BytesIO):
        return "bytesio_pdf"
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    if os.name == "nt":
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


def get_page_tokens(pdf_path, model=None, pdf_parser="PyPDF2"):
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            token_length = litellm.token_counter(model=model, text=page_text)
            page_list.append((page_text, token_length))
        return page_list
    if pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            doc = pymupdf.open(stream=pdf_path, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        else:
            raise ValueError("Unsupported PDF input for PyMuPDF parser")
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = litellm.token_counter(model=model, text=page_text)
            page_list.append((page_text, token_length))
        return page_list
    raise ValueError(f"Unsupported PDF parser: {pdf_parser}")


def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page - 1, end_page):
        text += pdf_pages[page_num][0]
    return text


def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page - 1, end_page):
        text += f"<physical_index_{page_num + 1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num + 1}>\n"
    return text


def get_number_of_pages(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    return len(pdf_reader.pages)


__all__ = [
    "BytesIO",
    "extract_text_from_pdf",
    "get_pdf_title",
    "get_text_of_pages",
    "get_first_start_page_from_text",
    "get_last_start_page_from_text",
    "sanitize_filename",
    "get_pdf_name",
    "get_page_tokens",
    "get_text_of_pdf_pages",
    "get_text_of_pdf_pages_with_labels",
    "get_number_of_pages",
]
