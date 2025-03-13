import logging
import os

from dotenv import load_dotenv
from llama_cloud import PipelineCreate, PresetRetrievalParams, RetrievalMode, EvalExecutionParams, SupportedLlmModel, \
    LlamaParseParameters, ParserLanguages, PipelineType
from llama_cloud.client import AsyncLlamaCloud

from app.services.llama_cloud.llama_cloud_embedding_service import LlamaCloudEmbeddingService
from app.model.klee_settings import Settings as KleeSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class LlamaCloudPipelinesService:
    def __init__(self):
        load_dotenv(f".env")
        self.async_client = AsyncLlamaCloud(token=os.getenv("LLAMA_CLOUD_API_KEY"))
        self.llama_cloud_embedding_service = LlamaCloudEmbeddingService()

    async def create_pipeline(
            self,
            embedding_config_id: str,
            name: str,
            project_id: str = None,
            organization_id: str = None
    ):
        """
        Create a pipeline
        Args:
            embedding_config_id: str
            name: str
            project_id: str
            organization_id: str
        Returns: Pipeline
        """
        request_data = {
            "embedding_config_id": embedding_config_id,
            "name": name
        }

        logger.info(f"Creating pipeline with name: {name} and embedding config id: {embedding_config_id} and project id: {project_id} and organization id: {organization_id}")

        pipeline_create_request = PipelineCreate(
            embedding_model_config_id=embedding_config_id,
            name=name,
            transform_config={
                "chunk_size": 512,
                "chunk_overlap": 50,
                "skip_empty_chunks": True,
                "text_splitter": "TOKEN"
            },
            data_sink_id=None,
            preset_retrieval_parameters = PresetRetrievalParams(
                dense_similarity_top_k=12,
                dense_similarity_cutoff=0.5,
                sparse_similarity_top_k=10,
                enable_reranking=True,
                rerank_top_n=5,
                alpha=0.5,
                search_filters=None,
                retrieval_mode=RetrievalMode.CHUNKS,
                retrieve_image_nodes=False,
                files_top_k=2,
                class_name="Test"
            ),
            eval_parameters= {
                "llm_model": "GPT_4O",
                "qa_prompt_tmpl": """
                    Context information is below. 
                    --------------------- {context_str} --------------------- 
                    Given the context information and not prior knowledge, 
                    answer the query. 
                    Query: {query_str} 
                    Answer:
                """
            },
            llama_parse_parameters = LlamaParseParameters(
                languages=[ParserLanguages.EN, ParserLanguages.CH_SIM, ParserLanguages.CH_TRA],
                parsing_instruction="",
                disable_ocr=False,
                annotate_links=False,
                disable_reconstruction=False,
                disable_image_extraction=False,
                invalidate_cache=False,
                output_pdf_of_document=False,
                do_not_cache=False,
                fast_mode=False,
                skip_diagonal_text=False,
                preserve_layout_alignment_across_pages=False,
                gpt4o_mode=False,
                gpt4o_api_key="",
                do_not_unroll_columns=False,
                extract_layout=False,
                html_make_all_elements_visible=False,
                html_remove_navigation_elements=False,
                html_remove_fixed_elements=False,
                guess_xlsx_sheet_name=False,
                page_separator=""
            ),
            embedding_config=None,
            configured_transformations=None,
            data_sink=None,
            pipeline_type=PipelineType.MANAGED,
            managed_pipeline_id=None,
        )
        response = await self.async_client.pipelines.create_pipeline(project_id=project_id,request=pipeline_create_request)
        logger.info(f"Pipeline created with id: {response.id}")
        return response

    async def get_pipeline_by_id(
            self,
            pipeline_id: str
    ):
        """
        Get a pipeline by id
        Args:
            pipeline_id: str
        Returns: Pipeline
        """
        response = await self.async_client.pipelines.get_pipeline(pipeline_id)
        logger.info(f"Pipeline with id: {pipeline_id} retrieved")
        return response

    async def delete_pipeline_by_id(
            self,
            pipeline_id: str
    ):
        """
        Delete a pipeline by id
        Args:
            pipeline_id: str
        Returns: None
        """
        await self.async_client.pipelines.delete_pipeline(pipeline_id)
        logger.info(f"Pipeline with id: {pipeline_id} deleted")

    async def search_pipelines(
            self,
            project_id: str
    ):
        """
        Search for pipelines
        Args:
            project_id: str
        Returns: List[Pipeline]
        """
        response = await self.async_client.pipelines.search_pipelines(project_id=project_id)
        logger.info(f"Pipelines retrieved for project: {project_id}")
        logger.info(f"Search response: {response}")
        return response

    async def add_files_to_pipeline(
            self,
            pipeline_id: str
    ):
        """
        Add files to a pipeline
        Args:
            pipeline_id: str
        Returns: List[PipelineFile]
        """
        files = [
            {"file_id": "53c860da-1712-4288-8cd4-b5660bbd0990"}
        ]
        response = await self.async_client.pipelines.add_files_to_pipeline(pipeline_id=pipeline_id, request=files)
        logger.info(f"Files added to pipeline: {pipeline_id}")
        return response