import os
import json
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from dotenv import load_dotenv
load_dotenv()


def partition_doc(docs_path):
    if not os.path.exists(docs_path):
        raise FileNotFoundError("The path doesn't contain any files")
    elements=partition_pdf(
        filename=docs_path,
        extract_images_in_pdf=True,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_to_payload=True,
    )
    return elements
def chunk_doc(elements):
    if len(elements)==0:
        raise FileNotFoundError("No elements detected in the file you have provided")
    chunks=chunk_by_title(
        elements,
        multipage_sections=True,
        combine_text_under_n_chars=500,
        new_after_n_chars=2400,
        max_characters=3000,
        overlap=200
    )
    return chunks
def separate_contents(chunks):
    content_data={
        "text":[],
        "tables":[],
        "images":[],
        "types":[] # Tells what kind of data is present in the chunk
    }
    for chunk in chunks:
        # Ensures only text is present and not empty string
        if hasattr(chunk,'text') and chunk.text:
            content_data["text"].append(chunk.text)
            # The orig_elements attribute is a Python list kept inside the chunk's metadata that remembers exactly which individual pieces were put into that box.
        if hasattr(chunk,'metadata') and hasattr(chunk.metadata,'orig_elements'):
            for element in chunk.metadata.orig_elements:
                #type(element).__name__ simply gets the name of the item as a string (e.g., "Table", "Image")
                element_type=type(element).__name__

                if element_type=="table":
                    content_data["types"].append('table')
                    # getattr(object, attribute, default_value): It tries get the text_as_html. If it doesn't exist, don't crash—just give me the standard element.text instead."
                    html_table=getattr(chunk.metadata,'table_as_html',element.text)
                    content_data["tables"].append(html_table)
                elif element_type=="image":
                    if hasattr(element,'metadata') and hasattr(element.metadata,'image_base64'):
                        content_data["types"].append('image')
                        content_data["images"].append(element.metadata.image_base64)

        content_data["types"]=list(set(content_data['types']))
    return content_data

