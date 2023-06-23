from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import time

tic = time.perf_counter() # Start timer

llm = OpenAI(temperature=0.1, model='text-davinci-003')

text_splitter = CharacterTextSplitter(        
    separator = " ",
    chunk_size = 2000,
    chunk_overlap  = 100,
)

with open("Transcript.txt") as f:
    transcript = f.read()

texts = text_splitter.split_text(transcript)
docs = text_splitter.create_documents(texts=texts)
num_docs = len(docs)
num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)
print (f"Prepared {num_docs} document(s) and the first one has {num_tokens_first_doc} tokens")

map_prompt_template = """Crea Notas formales del siguiente fragmento de transcripcion de una junta directiva. Actua como un Secretario de la organizacion, pero no te introduzcas ni te despidas en tu respuesta. El fragmento es el siguiente:


{text}


TU RESPUESTA:
"""
MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """Crea un Acta de Reunion de los siguientes puntos tomados de Juntas anteriores:

{text}


TU ACTA DE REUNION:
"""
COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
chain = load_summarize_chain(
    llm, 
    chain_type="map_reduce", 
    return_intermediate_steps=False, 
    map_prompt=MAP_PROMPT, 
    verbose=True,
    combine_prompt=COMBINE_PROMPT)
output = chain.run(docs)

toc = time.perf_counter() # Capture ending time

print('-----')
print(f"Ended sumarization job in {toc - tic:0.4f} seconds")
print('-----')

print(output)