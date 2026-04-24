import os
import json
from typing import List
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from dotenv import load_dotenv
import unstructured_pytesseract
unstructured_pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)
chat_history=[]

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
def separate_contents(chunk):
    content_data={
        "text":[],
        "tables":[],
        "images":[],
        "types":[] # Tells what kind of data is present in the chunk
    }
    
        # Ensures only text is present and not empty string
    if hasattr(chunk,'text') and chunk.text:
            content_data["text"].append(chunk.text)
            # The orig_elements attribute is a Python list kept inside the chunk's metadata that remembers exactly which individual pieces were put into that box.
    if hasattr(chunk,'metadata') and hasattr(chunk.metadata,'orig_elements'):
            for element in chunk.metadata.orig_elements:
                #type(element).__name__ simply gets the name of the item as a string (e.g., "Table", "Image")
                element_type=type(element).__name__

                if element_type=="Table":
                    content_data["types"].append('tables')
                    # getattr(object, attribute, default_value): It tries get the text_as_html. If it doesn't exist, don't crash—just give me the standard element.text instead."
                    # Why llm understands html better so if we do this we can senf it directly to llm
                    html_table=getattr(chunk.metadata,'table_as_html',element.text)
                    content_data["tables"].append(html_table)
                elif element_type=="Image":
                    if hasattr(element,'metadata') and hasattr(element.metadata,'image_base64'):
                        content_data["types"].append('images')
                        content_data["images"].append(element.metadata.image_base64)

    content_data["types"]=list(set(content_data['types']))
    return content_data
# This is for embedding model
def create_summary(text:str,tables:List[str],images:List[str])->str:
    try:

        prompt=f"""You are an expert technical assistant. Analyze the following content 
        and generate a highly detailed, searchable summary.
        Include key facts, metrics, and describe any visual anomalies.
        
        TEXT:
        {text}
        
        TABLES:
        {tables}

        """
        message_content = [
            {"type": "text", "text": prompt}
        ]
        # Llm talks to json not binary files 
        for img_base64 in images:
            message_content.append({
                "type":"image_url",
                "image_url":{"url":f"data:image/jpeg;base64,{img_base64}"}
            })
        message=HumanMessage(content=message_content)
        response=model.invoke([message])
        return response.content
    except Exception as e:
        return (f"Summary failed due to {e}")
# This is for vectordb as it requires langchain document 
def langdoc(chunks):
    langchain_documents=[]
    if len(chunks)==0:
        raise FileNotFoundError("Can't convert chunks to Langchain document")
    for chunk in chunks:
        content_data=separate_contents(chunk)
        if 'tables' in content_data['types'] or 'images' in content_data['types']:
            raw_text=" ".join(content_data['text'])
            data=create_summary(
                text=raw_text,
                tables=content_data['tables'],
                images=content_data['images']
            )
        else:
            data=" ".join(content_data['text'])

        LangDocument=Document(
            page_content=data,
            metadata={
                # Helpful to give images or table directly during retrieval
            "original_content":json.dumps({
                'raw_text':content_data['text'],
                'table_as_html':content_data['tables'],
                'base_64_image':content_data['images'],
            }
        )
        })
        langchain_documents.append(LangDocument)
    return langchain_documents


def embed(summary,persist_directory="db/chroma_db"):
    model_name="BAAI/bge-m3"
    model_kwargs={'device': 'cpu'}
    encode_kwargs={'normalize_embeddings': True}
    embedding_model= HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )
    vector_db=Chroma.from_documents(
        documents=summary,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_db
def export_chunks_to_json(chunks, filename="chunks_export.json"):
    # Saving in case or any error in db
    export_data = []
    
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print("file exported")
    return export_data
def memory(query):
    
    if chat_history:
        messages=[
            SystemMessage(content="Given chat history,rewrite the question to be standalone and just return the rewritten question")]+ chat_history+[HumanMessage(content=f"Question:{query}")]
        result=model.invoke(messages)
        question=result.content.strip()
    else:
        question=query
    return question
def chat(db):
    retriever=db.as_retriever(search_kwargs={"k":7})
    while(True):
        ques=input("Ask query")
        if 'quit' in ques.lower():
            break
        asked_ques=memory(ques)
        docs=retriever.invoke(asked_ques)
        relevant_content="\n\n".join([doc.page_content for doc in docs])
        llm_prompt=f"""You are an expert technical assistant, Your job is to provide accurate, professional answers based STRICTLY on the provided document context.

        ### INSTRUCTIONS:
        1. Review the provided context thoroughly.
        2. Answer the user's query using ONLY the information found in the context.
        3. If the context contains tables or image summaries, use them to enrich your answer.
        4. If the answer cannot be found in the context, you must output exactly: "I don't have enough information." Do not attempt to guess or use outside knowledge.

        ### CONTEXT:
        {relevant_content}

        ### USER QUERY:
        {asked_ques}
        """
        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and the previous chat history"),
        ] + chat_history + [
            HumanMessage(content=llm_prompt)
        ]
            
        result = model.invoke(messages)
        final_answer = result.content
        print(final_answer)
        chat_history.append(HumanMessage(content=ques))
        chat_history.append(AIMessage(content=final_answer))

def main():
    
    docs_path=r"C:\Users\Tharun R Gowda\Desktop\Multirag\docs\NIPS-2017-attention-is-all-you-need-Paper.pdf"
    elements=partition_doc(docs_path)
    chunks=chunk_doc(elements)
    response=langdoc(chunks)
    export_chunks_to_json(response)
    db=embed(response)
    chat(db)
        
if __name__=="__main__":
    main()
