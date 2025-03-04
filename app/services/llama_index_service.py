# os module
import os
import platform
import shutil
import contextlib
import logging

import yaml

# datetime
from datetime import datetime

from llama_index.core.instrumentation.events import rerank
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.anthropic import Anthropic

from llama_index.llms.deepseek import DeepSeek

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)


from llama_index.core.utils import get_cache_dir

from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes
)
from llama_index.core import SimpleDirectoryReader, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
# alias name
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.settings import Settings as llamaSettings

from llama_index.core.schema import QueryBundle

from llama_index.llms.ollama import Ollama
from app.model.knowledge import File

from pathlib import Path

import uuid

from app.services.client_sqlite_service import db_transaction
from app.model.base_config import BaseConfig

from app.model.knowledge import Knowledge
from app.common.LlamaEnum import (
    SystemTypeDiffVectorUrl,
    SystemTypeDiffConfigUrl,
    SystemTypeDiff,
    SystemTypeDiffTempFileUrl,
    SystemTypeDiffLlmUrl,
    SystemTypeDiffModelType,
    SystemTiktokenUrl,
    SystemEmbedUrl
)

from sqlalchemy import select

from app.model.klee_settings import Settings as KleeSettings

from llama_index.core.retrievers import QueryFusionRetriever
from typing import List

from llama_index.core.agent import AgentRunner
from llama_index.core.tools.query_engine import QueryEngineTool

from app.model.note import Note
from app.model.global_settings import GlobalSettings

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LlamaIndexService:
    def __init__(self):
        # user home path
        self.user_home = os.path.expanduser("~")
        # embed model path
        self.embed_model = os.path.join(self.user_home)
        self.chunk_sizes = [2048, 512, 128]

    async def init_config(self):
        """
        init basic config
        """
        os_type = self.judge_system_type()
        KleeSettings.os_type = os_type
        KleeSettings.un_load = True
        data = {
            "embedModelList": None,
            "llmModelList": None,
            "uid": None,
            "openai_api_key": "",
            "embed_model": None,
            "llm": None,
            "http_proxy": ""
        }

        if os_type == SystemTypeDiff.WIN.value:
            if not os.path.exists(SystemTypeDiffLlmUrl.WIN_PATH.value):
                os.makedirs(SystemTypeDiffLlmUrl.WIN_PATH.value, exist_ok=True)
            if not os.path.exists(SystemTypeDiffConfigUrl.WIN_OS_path.value):
                os.makedirs(SystemTypeDiffConfigUrl.WIN_OS_path.value, exist_ok=True)
            if not os.path.exists(SystemTypeDiffTempFileUrl.WIN_PATH.value):
                os.makedirs(SystemTypeDiffTempFileUrl.WIN_PATH.value)
            if not os.path.exists(f"{SystemTypeDiffTempFileUrl.WIN_PATH.value}default"):
                os.makedirs(f"{SystemTypeDiffTempFileUrl.WIN_PATH.value}default", exist_ok=True)
                with open(f"{SystemTypeDiffTempFileUrl.WIN_PATH.value}default/default.txt", "w") as default_file:
                    default_file.write("")
            if not os.path.exists(f"{SystemEmbedUrl.WIN_PATH.value}"):
                os.makedirs(f"{SystemEmbedUrl.WIN_PATH.value}", exist_ok=True)
            with open(SystemTypeDiffConfigUrl.WIN_OS.value, 'w') as file:
                yaml.dump(data, file, default_flow_style=False)

        elif os_type == SystemTypeDiff.MAC.value:
            if not os.path.exists(SystemTypeDiffLlmUrl.MAC_PATH.value):
                os.makedirs(SystemTypeDiffLlmUrl.MAC_PATH.value, exist_ok=True)
            if not os.path.exists(SystemTypeDiffConfigUrl.MAC_OS_path.value):
                os.makedirs(SystemTypeDiffConfigUrl.MAC_OS_path.value, exist_ok=True)
            if not os.path.exists(SystemTypeDiffTempFileUrl.MAC_PATH.value):
                os.makedirs(SystemTypeDiffTempFileUrl.MAC_PATH.value, exist_ok=True)
            if not os.path.exists(f"{SystemTypeDiffTempFileUrl.MAC_PATH.value}default"):
                os.makedirs(f"{SystemTypeDiffTempFileUrl.MAC_PATH.value}default", exist_ok=True)
            if not os.path.exists(f"{SystemTiktokenUrl.MAC_PATH.value}"):
                os.makedirs(f"{SystemTiktokenUrl.MAC_PATH.value}", exist_ok=True)
            if not os.path.exists(f"{SystemEmbedUrl.MAC_PATH.value}"):
                os.makedirs(f"{SystemEmbedUrl.MAC_PATH.value}", exist_ok=True)
            os.environ["TIKTOKEN_CACHE_DIR"] = f"{SystemTiktokenUrl.MAC_PATH.value}"
            with open(f"{SystemTypeDiffTempFileUrl.MAC_PATH.value}default/default.txt", "w") as default_file:
                default_file.write("")
            with open(SystemTypeDiffConfigUrl.MAC_OS.value, 'w') as file:
                yaml.dump(data, file, default_flow_style=False)
        await self.load_config(os_type=os_type)

    def judge_system_type(self) -> str:
        """
            Judge system type includes: Windows, Linux or macOS, then set settings var
        """
        if os.name == 'nt':
            return "Windows"
        elif os.name == "posix":
            if 'darwin' in platform.system().lower():
                return "Linux/macOS"
        else:
            raise Exception(
                "Unknown system type, klee may be not support to this system"
            )

    async def load_config(
            self,
            os_type: str = None
    ):
        """
        load config
        :param: os_type: str   windows or linux or macOS
        return: None
        """
        if os_type == SystemTypeDiff.WIN.value:
            KleeSettings.config_url = SystemTypeDiffConfigUrl.WIN_OS.value
            KleeSettings.vector_url = SystemTypeDiffVectorUrl.WIN_OS.value
            KleeSettings.temp_file_url = SystemTypeDiffTempFileUrl.WIN_PATH.value
            KleeSettings.llm_path = SystemTypeDiffLlmUrl.WIN_PATH.value

            if os.path.exists("./all-MiniLM-L6-v2") and not os.path.exists(
                    f"{KleeSettings.embed_model_path}all-MiniLM-L6-v2"):
                pass

            if os.path.exists("./tiktoken_encode") and not os.path.exists(SystemTiktokenUrl.WIN_PATH.value):
                absolute_path_tiktoken = os.path.abspath("./tiktoken_encode")
                path_arr_tiktoken = absolute_path_tiktoken.split("\\")
                absolute_path_tiktoken_real = ""
                for i in range(len(path_arr_tiktoken)):
                    if i == len(path_arr_tiktoken) - 2:
                        # absolute_path_tiktoken_real += "klee-kernel/"
                        absolute_path_tiktoken_real += path_arr_tiktoken[i] + "/"
                    elif i == len(path_arr_tiktoken) - 1:
                        # absolute_path_tiktoken_real += "main/tiktoken_encode"
                        absolute_path_tiktoken_real += "tiktoken_encode"
                    else:
                        absolute_path_tiktoken_real += path_arr_tiktoken[i] + "/"
                logger.info(f"absolute_path_real:{absolute_path_tiktoken_real}")

                os.environ["TIKTOKEN_CACHE_DIR"] = absolute_path_tiktoken_real

            absolute_path = os.path.abspath("./all-MiniLM-L6-v2")
            path_arr = absolute_path.split("\\")

            absolute_path_real = ""
            for i in range(len(path_arr)):
                if i == len(path_arr) - 2:
                    # absolute_path_real += "klee-kernel/"
                    absolute_path_real += path_arr[i] + "/"
                elif i == len(path_arr) - 1:
                    # absolute_path_real += "main/all-MiniLM-L6-v2"
                    absolute_path_real += "all-MiniLM-L6-v2"
                else:
                    absolute_path_real += path_arr[i] + "/"
            logger.info(f"absolute_path_real:{absolute_path_real}")

            llamaSettings.embed_model = f"local:{absolute_path_real}"
            KleeSettings.embed_model_path = absolute_path_real
        elif os_type == SystemTypeDiff.MAC.value:
            KleeSettings.config_url = SystemTypeDiffConfigUrl.MAC_OS.value
            KleeSettings.vector_url = SystemTypeDiffVectorUrl.MAC_OS.value
            KleeSettings.temp_file_url = SystemTypeDiffTempFileUrl.MAC_PATH.value
            KleeSettings.llm_path = SystemTypeDiffLlmUrl.MAC_PATH.value
            KleeSettings.embed_model_path = SystemEmbedUrl.MAC_PATH.value

            if os.path.exists("./all-MiniLM-L6-v2") and not os.path.exists(
                    f"{KleeSettings.embed_model_path}all-MiniLM-L6-v2"):
                os.chmod("./all-MiniLM-L6-v2", 0o777)
                shutil.move("./all-MiniLM-L6-v2", f"{KleeSettings.embed_model_path}all-MiniLM-L6-v2")

            if os.path.exists("./tiktoken_encode") and not os.path.exists(SystemTiktokenUrl.MAC_PATH.value):
                os.chmod("./tiktoken_encode", 0o777)
                shutil.move("./tiktoken_encode", f"{SystemTiktokenUrl.MAC_PATH.value}")
                os.environ["TIKTOKEN_CACHE_DIR"] = f"{SystemTiktokenUrl.MAC_PATH.value}"

            llamaSettings.embed_model = f"local:{SystemEmbedUrl.MAC_PATH.value}all-MiniLM-L6-v2"

        with open(KleeSettings.config_url, 'r') as file:
            KleeSettings.data = yaml.safe_load(file)
            if KleeSettings.data.get('openai_api_key') is not None:
                os.environ["OPENAI_API_KEY"] = KleeSettings.data.get("openai_api_key")
        KleeSettings.center_url = f"https://xltwffswqvowersvchkj.supabase.co/"

    def load_text_document(
            self,
            source: str
    ):
        """
        load text document from file or folder
        """
        # Get data from a folder
        documents = SimpleDirectoryReader(source).load_data()

        return documents

    @contextlib.contextmanager
    def release_memory(self) -> None:
        """
        release memory
        """
        try:
            yield
        finally:
            import gc
            gc.collect()

    async def load_llm(
            self,
            provider_id: str = None,
            model_name: str = None,
            api_type: str = None,
            api_base_url: str = "https://api.deepseek.com",
    ):
        """
        load llm
        Args:
            provider_id: provider id
            model_name: model name
            api_type: api type
            api_base_url: api base url
        Returns: None
        """
        try:
            with self.release_memory():
                llamaSettings.llm = None
                self.release_memory()

            if KleeSettings.local_mode is False:
                if provider_id != SystemTypeDiffModelType.OPENAI.value and provider_id != SystemTypeDiffModelType.CLAUDE.value:
                    if api_type == SystemTypeDiffModelType.OPENAI.value:
                        llamaSettings.llm = OpenAI(model=model_name, temperature=0.5)
                    elif api_type == SystemTypeDiffModelType.CLAUDE.value:
                        llamaSettings.llm = Anthropic(model=model_name, temperature=0.5)
                    elif api_type == SystemTypeDiffModelType.DEEPSEEK.value:
                        if api_base_url is not None and api_base_url.find("luchentech") != -1:
                            api_base_url = "https://cloud.luchentech.com/api/maas"
                            llamaSettings.llm = DeepSeek(
                                model=model_name,
                                temperature=0.5,
                                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                                api_base=api_base_url
                            )
                        else:
                            llamaSettings.llm = DeepSeek(model=model_name, temperature=0.5)
                    KleeSettings.un_load = False
                else:
                    KleeSettings.un_load = False
            else:
                if provider_id == SystemTypeDiffModelType.OLLAMA.value:
                    llamaSettings.llm = Ollama(
                        model=model_name,
                        request_timeout=60.0
                    )
                    KleeSettings.un_load = False

        except Exception as e:
            raise Exception(f"Load llm failed, provider_id:{provider_id},  model_id:{model_name}")

    async def persist_file_to_disk_2(
            self,
            path: str,
            store_dir: str,
            chunk_sizes=None,
    ) -> None:
        try:
            documents = self.load_text_document(path)

            chunk_size = chunk_sizes or self.chunk_sizes
            node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_size)
            nodes = node_parser.get_nodes_from_documents(documents)
            leaf_nodes = get_leaf_nodes(nodes)
            doc_store = SimpleDocumentStore()
            doc_store.add_documents(nodes)

            store_context = StorageContext.from_defaults(docstore=doc_store)
            auto_merging_index = VectorStoreIndex(
                leaf_nodes, storage_context=store_context
            )

            auto_merging_index.storage_context.persist(persist_dir=store_dir)
        except Exception as e:
            raise Exception(e)


    def build_auto_merging_index(
            self,
            documents,
            save_dir="F:/auto_merge_data",
            chunk_sizes=None
    ) -> VectorStoreIndex:
        """
        Build auto merging index
        Args:
            documents: documents
            save_dir: save dir: the path to save the index
            chunk_sizes: chunk sizes
        Returns: auto merging index
        """
        if not os.path.exists(save_dir):
            chunk_size = chunk_sizes or self.chunk_sizes
            node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_size)
            nodes = node_parser.get_nodes_from_documents(documents)
            leaf_nodes = get_leaf_nodes(nodes)
            doc_store = SimpleDocumentStore()
            doc_store.add_documents(nodes)

            store_context = StorageContext.from_defaults(docstore=doc_store)
            auto_merging_index = VectorStoreIndex(
                leaf_nodes, storage_context=store_context
            )
            auto_merging_index.storage_context.persist(persist_dir=save_dir)
        else:
            store_context_from_disk = StorageContext.from_defaults(persist_dir=save_dir)

            auto_merging_index = load_index_from_storage(
                store_context_from_disk
            )

        return auto_merging_index

    def get_auto_merging_query_engine(
            self,
            index: VectorStoreIndex,
            similarity_top_k=12,
            rerank_top_n=6
    ) -> RetrieverQueryEngine:
        """
        Get auto merging query engine
        Args:
            index: index
            similarity_top_k: similarity top k
            rerank_top_n: rerank top n
        Returns: query engine
        """
        base_retriever = index.as_retriever(
            # streaming=True,
            similarity_top_k=similarity_top_k
        )
        retriever = AutoMergingRetriever(
            base_retriever,
            index.storage_context,
            simple_ratio_thresh=0.5,
            verbose=False,
        )
        auto_merging_engine = RetrieverQueryEngine.from_args(
            retriever,
            # response_mode=ResponseMode.GENERATION,
            streaming=True,
            rerank_top_n=rerank_top_n,
            # text_qa_template=PromptTemplate(sim_template)
            # node_postprocessors=[rerank]
        )
        return auto_merging_engine

    async def get_retrieve_notes_content(
            self,
            query_engine: RetrieverQueryEngine,
            question: str
    ):
        """
        Get retrieve notes content
        Args:
            query_engine: query engine
            question: question
        Returns: content
        """
        score_list = query_engine.retrieve(query_bundle=QueryBundle(query_str=question))
        content = ""
        for node in score_list:
            content = f"{content}{node.text}\n\n"
        return content

    async def choose_which_embed_model(
            self,
            embed_model_path: str,
    ) -> None:
        """
            Choose which embed model of llama settings for global
        Args:
            embed_model_path: path of embed model
        Returns: None
        """
        if embed_model_path is not None:
            cache_folder = os.path.join(get_cache_dir(), "models")
            os.makedirs(cache_folder, exist_ok=True)
            embed_model = HuggingFaceEmbedding(
                model_name="local:".join(embed_model_path), cache_folder=cache_folder
            )
            llamaSettings.embed_model = embed_model
            KleeSettings.embed_model_path = "local:".join(embed_model_path)

    async def import_exist_dir(
            self,
            knowledge_id: str,
            dir_path,
            session
    ):
        """
        import exist dir to database
        Args:
            knowledge_id: knowledge id
            dir_path: dir path
            session: session
        Returns: None
        """
        try:
            directory_path = Path(dir_path)
            path_list = []

            stmt = select(Knowledge).where(Knowledge.id == knowledge_id)
            result = await session.execute(stmt)
            knowledge_data = result.scalars().first()

            for file in directory_path.rglob('*'):
                if file.is_file():
                    file_path = str(file)
                    title = ""
                    if KleeSettings.os_type == SystemTypeDiff.WIN.value:
                        title = file_path.split("\\")[-1]
                    elif KleeSettings.os_type == SystemTypeDiff.MAC.value:
                        title = file_path.split("/")[-1]
                    file_id = str(uuid.uuid4())

                    # if file_path
                    if file_path.find("DS_Store") != -1:
                        continue
                    knowledge = File(
                        id=file_id,
                        path=file_path,
                        name=title,
                        size=os.path.getsize(file_path),
                        knowledgeId=knowledge_data.id,
                        os_mtime=datetime.now().timestamp(),
                        create_at=datetime.now().timestamp(),
                        update_at=datetime.now().timestamp()
                    )
                    path_list.append(knowledge)

                    if not os.path.exists(f"{KleeSettings.temp_file_url}{file_id}"):
                        os.makedirs(f"{KleeSettings.temp_file_url}{file_id}")
                    shutil.copy(file_path, f"{KleeSettings.temp_file_url}{file_id}")
                    await self.persist_file_to_disk_2(path=f"{KleeSettings.temp_file_url}{file_id}",
                                                 store_dir=f"{KleeSettings.vector_url}{file_id}")

            session.add_all(path_list)
        except Exception:
            raise Exception(
                "error"
            )

    async def combine_query(
            self,
            knowledge_ids: List[str] = None,
            note_ids: List[str] = None,
            file_infos: dict = None,
            streaming: bool = True
    ):
        """
        Use query engine to combine query from knowledge, note and file
        Args:
            knowledge_ids: list of knowledge ids
            note_ids: list of note ids
            file_infos: dict of file infos
            streaming: bool
        Returns: query engine
        """
        retrievers = []
        if knowledge_ids is not None and len(knowledge_ids) > 0:
            for s in knowledge_ids:
                documents = self.load_text_document(f"{KleeSettings.temp_file_url}{s}")
                index = self.build_auto_merging_index(documents, save_dir=f"{KleeSettings.vector_url}{s}")
                base_retriever = index.as_retriever(
                    # streaming=True,
                    similarity_top_k=6
                )
                retriever = AutoMergingRetriever(
                    base_retriever,
                    index.storage_context,
                    simple_ratio_thresh=0.5,
                    verbose=False,
                )
                retrievers.append(retriever)

        if note_ids is not None and len(note_ids) > 0:
            for n in note_ids:
                documents = self.load_text_document(f"{KleeSettings.temp_file_url}{n}")
                index = self.build_auto_merging_index(documents, save_dir=f"{KleeSettings.vector_url}{n}")
                base_retriever = index.as_retriever(
                    similarity_top_k=12
                )
                retriever = AutoMergingRetriever(
                    base_retriever,
                    index.storage_context,
                    simple_ratio_thresh=0.2,
                    verbose=True,
                )

                retrievers.append(retriever)

        if file_infos is not None:
            for key in file_infos:
                knowledge_id = key
                files = file_infos.get(knowledge_id)
                for file in files:
                    documents = self.load_text_document(source=f"{KleeSettings.temp_file_url}{file.id}")
                    index = self.build_auto_merging_index(documents, save_dir=f"{KleeSettings.vector_url}{file.id}")
                    base_retriever = index.as_retriever(
                        similarity_top_k=6
                    )
                    retriever = AutoMergingRetriever(
                        base_retriever,
                        index.storage_context,
                        simple_ratio_thresh=0.2,
                        verbose=False,
                    )
                    retrievers.append(retriever)

        text_qa_prompt = """
               "Context information is below.\n"
               "---------------------\n"
               "{context_str}\n"
               "---------------------\n"
               "Given the context information and not prior knowledge, "
               "answer the query.\n"
               "Query: {query_str}\n"
               "Answer: "
           """

        refine_prompt = """
               "The original query is as follows: {query_str}\n"
               "We have provided an existing answer: {existing_answer}\n"
               "We have the opportunity to refine the existing answer "
               "(only if needed) with some more context below.\n"
               "------------\n"
               "{context_msg}\n"
               "------------\n"
               "Given the new context, refine the original answer to better "
               "answer the query. "
               "If the context isn't useful, return the original answer.\n"
               "Refined Answer: "
           """

        summary_prompt = """
               "Context information from multiple sources is below.\n"
               "---------------------\n"
               "{context_str}\n"
               "---------------------\n"
               "Given the information from multiple sources and not prior knowledge, "
               "answer the query.\n"
               "Query: {query_str}\n"
               "Answer: "
           """

        if len(retrievers) == 0:
            documents = self.load_text_document(f"{KleeSettings.temp_file_url}default")
            index = self.build_auto_merging_index(documents=documents, save_dir=f"{KleeSettings.vector_url}default")
            base_retriever = index.as_retriever(
                # streaming=True,
                similarity_top_k=12
            )
            retriever = AutoMergingRetriever(
                base_retriever,
                index.storage_context,
                simple_ratio_thresh=0.2,
                verbose=False,
            )
            retrievers.append(retriever)

            text_qa_prompt = """
                "if the quoted content is empty or unrelated to the question, there is no need to answer based on the context of the quoted content. \n"
                "answer the query.\n"
                "Query: {query_str}\n"
            """

            summary_prompt = """
                "if the quoted content is empty or unrelated to the question, there is no need to answer based on the context of the quoted content. \n"
                "answer the query.\n"
                "Query: {query_str}\n"
            """

        QUERY_GEN_PROMPT = (
            "You are a helpful assistant that generates multiple search queries based on a "
            "single input query. Generate {num_queries} search queries, one on each line, "
            "related to the following input query:\n"
            "Query: {query}\n"
            "Queries:\n"
        )

        qf_retriever = QueryFusionRetriever(
            retrievers,
            similarity_top_k=12,
            num_queries=4,  # Set to 1 for now
            use_async=True,
            query_gen_prompt=QUERY_GEN_PROMPT,
        )

        auto_merging_engine = RetrieverQueryEngine.from_args(
            qf_retriever,
            streaming=streaming,
            text_qa_template=PromptTemplate(text_qa_prompt),
            refine_template=PromptTemplate(refine_prompt),
            summary_template=PromptTemplate(summary_prompt),
            use_async = True,
            response_mode=ResponseMode.COMPACT
        )

        return auto_merging_engine

    async def has_files(
            self,
            path: str = None
    ):
        """
            Check if the folder has files
            Args:
                path: folder path
            Returns: True if the folder has files, False otherwise
        """
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                return True
        return False

    async def get_chat_engine(
            self,
            knowledge_ids: List[str] = None,
            knowledge_list: List[Knowledge] = None,
            note_ids: List[str] = None,
            note_list: List[Note] = None,
            file_infos: dict = None,
            chat_history=None,
            language: str = None
    ):
        """
            Get chat engine
        """
        agent_tools = []

        if chat_history is None:
            chat_history = []
        ids = []
        if knowledge_ids is not None:
            for knowledge_id in knowledge_ids:
                file_url = f"{KleeSettings.temp_file_url}{knowledge_id}"
                flag = await self.has_files(file_url)
                if flag is True:
                    ids.append(knowledge_id)
        for note in note_list:
            file_url = f"{KleeSettings.temp_file_url}{note.id}"
            flag = await self.has_files(file_url)
            if flag is True:
                documents_url = f"{KleeSettings.temp_file_url}{note.id}"
                documents = self.load_text_document(documents_url)
                # Get vector url
                vector_url = f"{KleeSettings.vector_url}{note.id}"
                index = self.build_auto_merging_index(
                    documents=documents,
                    save_dir=vector_url
                )

                query_engine = self.get_auto_merging_query_engine(index=index)

                # convert query engine to tool
                query_engine_tool = QueryEngineTool.from_defaults(
                    query_engine=query_engine,
                    name=note.title,
                    description=note.content
                )
                agent_tools.append(query_engine_tool)

        chat_engine = AgentRunner.from_llm(
            tools=agent_tools,
            llm=llamaSettings.llm,
            system_prompt="""""", # TODO: add system prompt
            verbose=False,
            chat_history=chat_history
        )

        return chat_engine

    @db_transaction
    async def init_global_model_settings(
            self,
            session=None
    ):
        stmt = select(GlobalSettings)
        result = await session.execute(stmt)
        result = result.scalars().one_or_none()

        if result is None:
            global_settings = GlobalSettings(
                id=str(uuid.uuid4()),
                create_at=datetime.now().timestamp(),
                update_at=datetime.now().timestamp(),
                model_id='',
                model_name='',
                model_path='',
                provider_id='',
                local_mode=True
            )
            session.add(global_settings)
            try:
                session.commit()
            except Exception as e:
                logger.error(f"Init global model settings error: {e}")
                session.rollback()

            KleeSettings.local_mode = True
            KleeSettings.model_id = None
            KleeSettings.model_name = None
            KleeSettings.model_path = None
            KleeSettings.provider_id = None
        else:
            KleeSettings.local_mode = result.local_mode
            KleeSettings.model_id = result.model_id
            KleeSettings.model_name = result.model_name
            KleeSettings.model_path = result.model_path
            KleeSettings.provider_id = result.provider_id
        logger.info(f"Init global model settings: {KleeSettings.local_mode}, {KleeSettings.model_id}, {KleeSettings.model_name}, {KleeSettings.model_path}, {KleeSettings.provider_id}")

    @db_transaction
    async def update_global_model_settings(
            self,
            local_mode: bool,
            model_id: str,
            model_name: str,
            model_path: str,
            provider_id: str,
            session=None
    ):
        stmt = select(GlobalSettings)
        result = await session.execute(stmt)
        result = result.scalars().one_or_none()

        result.local_mode = local_mode
        result.model_id = model_id
        result.model_name = model_name
        result.model_path = model_path
        result.provider_id = provider_id
        result.update_at = datetime.now().timestamp()

        try:
            session.commit()
            session.refresh(result)
        except Exception as e:
            logger.error(f"Init global model settings error: {e}")
            session.rollback()


