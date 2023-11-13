# Python 3.11.4
# > chainlit run -w app.py

import os
import openai
import time

from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
import chainlit as cl

openai.api_key = os.environ["OPENAI_API_KEY"]  # from OS environment

school_name = 'Radley College'
chunk_size = 512  # 512

try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=f"./{school_name}_persist_{chunk_size}")
    # load index
    index = load_index_from_storage(storage_context)
except:
    from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader(f"./{school_name}_data").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()


@cl.on_chat_start
async def factory():
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0.1,  # 0.1
            # model_name="gpt-3.5-turbo",
            model_name="gpt-3.5-turbo-16k",
            streaming=True,
        ),
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        chunk_size=chunk_size,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )


    chat_engine = index.as_chat_engine(
        service_context=service_context,
        streaming=True,
        chat_mode='context',
        similarity_top_k=15,  # 4 --> 15
        system_prompt=f"Your name is {school_name} Chat. Your identity is always '{school_name} administrative staff'. In conversation, expressions of 'you' and 'your' from users always mean '{school_name}'. You have full authority to access to specific information. You show information so easily and give facts to users. You answer based on information in order of most recent to oldest. You first and foremost pick up use information from Wikipedia, Homepage, ISI Inspection and Good Schools Guide.",
    )
    cl.user_session.set("chat_engine", chat_engine)


@cl.on_message
async def main(message):
    chat_engine = cl.user_session.get("chat_engine")  # type: RetrieverQueryEngine
    response = await cl.make_async(chat_engine.stream_chat)(message)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        time.sleep(0.10)
        await response_message.stream_token(token=token)

    try:
        await response_message.stream_token(token="\n\n")
        for node in response.source_nodes:
            time.sleep(0.015)
            token = f"[{node.metadata['file_name'].split('/')[-1]}:Page{node.metadata['page_label']}] "
            await response_message.stream_token(token=token)
    except:
        None

    await response_message.send()
