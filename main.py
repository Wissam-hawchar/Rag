from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from typing import Dict, List, Tuple, Optional
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_together import ChatTogether
from neo4j import GraphDatabase
import trafilatura
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
import logging 
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.messages import HumanMessage, AIMessage  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UrlRequest(BaseModel):
    url: str
# Configuration
class Settings(BaseModel):
    neo4j_uri: str = "neo4j+s://ffff5c48.databases.neo4j.io"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "dLAoJC2M4uo-jhymolkOc4fH6MKzFInaHInujc85W1g"
    together_api_key: str = "1fb14972d9a3b44b6082102760de47ec8888d0266d7729a0cf7264366e172038"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    class Config:
        env_file = ".env"

settings = Settings()

class AppDependencies:
    def __init__(self):
        self.llm = None
        self.graph = None
        self.vector_index = None
        self.entity_chain = None
        self.rewrite_chain = None
        self.qa_chain = None


# FastAPI Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up application resources"""
    deps = AppDependencies()
    
    try:
        # Initialize Neo4j
        deps.graph = Neo4jGraph(
            url=settings.neo4j_uri,
            username=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        # Initialize Embeddings
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        
        # Initialize Vector Index
        deps.vector_index = Neo4jVector.from_existing_graph(
            embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        
        # Initialize LLM
        deps.llm = init_chat_model("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", model_provider="together")
        
        # Initialize Chains
        initialize_chains(deps)
        
        app.state.dependencies = deps
        yield
        
    finally:
        # Cleanup
        if deps.graph:
            deps.graph.close()

app = FastAPI(lifespan=lifespan)

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    chat_history: Optional[List[Tuple[str, str]]] = None

class AnswerResponse(BaseModel):
    question: str
    answer: str
    rewritten_queries: List[str]
    sources: List[str]

class Entities(BaseModel):
    names: List[str] = Field(
        ...,
        description="Extracted entities from the text."
    )

    @field_validator('names', mode='before')
    @classmethod
    def split_string(cls, v):
        if isinstance(v, str):
            return v.strip("[]").replace("'", "").split(", ")
        return v



def _format_chat_history(chat_history: List[Tuple[str, str]]):
    """Format chat history into message list"""
    return [
        HumanMessage(content=human) if i % 2 == 0 else AIMessage(content=ai)
        for i, (human, ai) in enumerate(chat_history)
    ]

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    query = " AND ".join([f"{word}~2" for word in words])
    return query


# Core Processing Functions
def initialize_chains(deps: AppDependencies):
    entity_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract organization and person entities from the text. Return as a JSON list even if empty. Example: ['Entity1', 'Entity2']"),
    ("human", "Text: {question}"),
    ])
    deps.entity_chain = entity_prompt | deps.llm.with_structured_output(Entities)
    
    system_rewrite = """
    You are an assistant that enhances user queries using known relationship types.
    Rephrase the query incorporating relevant relationship terms where applicable.
    Return 3 different versions of the query.
    """

    examples = [
        {"question": "Who are Amelia Earhart’s relatives?", "answer": "Who are Amelia Earhart's FAMILY members?"},
        {"question": "Which countries does the US have agreements with?", "answer": "Which countries have DIPLOMATIC_RELATIONS with the US?"},
        {"question": "Who worked under Steve Jobs?", "answer": "Who was a SECRETARY or TUTOR under Steve Jobs?"}
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}"),
        ("ai", "{answer}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    query_rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", system_rewrite),
        few_shot_prompt,
        ("human", "{question}"),
    ])
    
    deps.rewrite_chain = query_rewrite_prompt | ChatTogether(model='meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo')
    
    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
            RunnablePassthrough.assign(chat_history=lambda x: _format_chat_history(x["chat_history"]))
            | PromptTemplate.from_template("Given the chat history, rewrite the follow-up question: {question}")
            | deps.llm
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x: x["question"]),
    )
    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    # Return the final chain
    deps.qa_chain= (
        RunnableParallel(
            {
                "context": _search_query | RunnableLambda(lambda q: retriever(q, deps)),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | deps.llm
        | StrOutputParser()
    )

def retriever(question: str,deps: AppDependencies)-> str:
    logger.info(f"Search query: {question}")
    structured_data = structured_retriever(question, deps)
    unstructured_data = [el.page_content for el in deps.vector_index.similarity_search(question)]
    return f"""Structured data:\n{structured_data}\nUnstructured data:\n{'#Document '.join(unstructured_data)}"""


# Existing Functions from Notebook (modified for API use)
def structured_retriever(question: str, deps: AppDependencies)-> List[Dict]:
    try:
        rewritten_response = deps.rewrite_chain.invoke({"question": question})
        if hasattr(rewritten_response, 'content'):
            rewritten_text = rewritten_response.content
            rewritten_queries = [line.split(": ")[1] for line in rewritten_text.split("\n") if line.startswith("Query")]
        else:
            rewritten_queries = [question]

        logger.info(f"Original: {question} | Rewritten: {rewritten_queries}")

        results = []
        for query in rewritten_queries:
            
            entities = deps.entity_chain.invoke({"question": query})
            logger.info(f"Processing query: {query} | Entities: {entities.names}")

            for entity in entities.names:

                ft_query = generate_full_text_query(entity)
                logger.debug(f"Entity search: {entity} | Query: {ft_query}")


                response = deps.graph.query(
                    """
                    CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                    YIELD node,score
                    MATCH (node)-[r]->(neighbor)
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    """,
                    {"query": ft_query},
                )
                if response:
                    results.extend(response)
                return results
    except Exception as e:
        logger.error(f"Structured retriever error: {str(e)}")
        return []

def parse_rewritten_queries(response) -> List[str]:
    """Parse rewritten queries from LLM response"""
    if hasattr(response, 'content'):
        return [
            line.split(": ")[1]
            for line in response.content.split("\n")
            if line.startswith("Query")
        ]
    return []

@app.post("/process-url")
async def process_url(
    request: UrlRequest,
    deps: AppDependencies = Depends(lambda: app.state.dependencies)
):
    """Process a URL and add its content to the knowledge graph"""
    try:
        # 1. Load and process URL
        def process_response(response, *args, **kwargs):
            html_content = response.text
            main_text = trafilatura.extract(html_content) or ""
            response._content = main_text.encode("utf-8")
            response.encoding = "utf-8"
            return response

        loader = WebBaseLoader(web_paths=(request.url,))
        loader.requests_kwargs = {"hooks": {"response": process_response}}
        raw_documents = loader.load()

        # 2. Split documents
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        documents = text_splitter.split_documents(raw_documents[:])

        
        llm_transformer = LLMGraphTransformer(llm=deps.llm)
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        
        # 5. Add to Neo4j
        deps.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        deps.vector_index = Neo4jVector.from_existing_graph(
            HuggingFaceEmbeddings(model_name=settings.embedding_model),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        return {"status": "success", "processed_chunks": len(documents)}
    
    except Exception as e:
        logger.error(f"URL processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
