# Python 3.11.4

# llamaindex

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext

#
import os
import openai

os.environ["OPENAI_API_KEY"]=""
openai.api_key = os.environ["OPENAI_API_KEY"]  # embedded in code

school_name = 'Radley College'
chunk_size = 512

print(f"Building Index: {school_name}")

#
filename_fn = lambda filename: {'file_name': filename}
# builds an index over the documents in the assigned folder
documents = SimpleDirectoryReader(f"../3_chainlit_app/{school_name}_data",
                                  file_metadata=filename_fn).load_data()

service_context = ServiceContext.from_defaults(chunk_size=chunk_size, chunk_overlap=50)
print(f"Chunk Size: {chunk_size}")
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

index.storage_context.persist(persist_dir=f"../3_chainlit_app/{school_name}_persist_{chunk_size}")

print("Completed")
