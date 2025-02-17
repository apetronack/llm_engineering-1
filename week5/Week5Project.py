import os
import glob
import json
from typing import List, Dict
import logging
from datetime import datetime
from itertools import islice
from langchain_openai import ChatOpenAI
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from DocProcessor import DocProcessor
from langchain.chains import ConversationalRetrievalChain
import gradio as gr
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainFilter
from typing import List, Dict
import hashlib

class DocumentProcessor:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        db_name: str = "vector_db",
        data_folders: str = "KnowledgeBase/*",
        document_metadata_dump: str = "documents_metadata.json",
        memory_k: int = 5,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_batch_size: int = 40000,
        min_chunk_length: int = 500,
        max_chunk_length: int = 2000
    ):
        self.model = model
        self.db_name = db_name
        self.data_folders = data_folders
        self.document_metadata_dump = document_metadata_dump
        self.memory_k = memory_k
        self.max_batch_size = max_batch_size
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        
        # Set up logging
        self.setup_logging()
        
        # Load environment variables
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        # Initialize processors
        self.doc_processor = DocProcessor(max_table_preview_rows=5)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize vector store
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        # Check if vector store already exists with that db_name. If not, create one.
        self.find_vector_store()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'doc_processing_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def filter_chunks(self, chunks: List[str]) -> List[str]:
        """Filter chunks based on length and content quality"""
        return [
            chunk for chunk in chunks 
            if self.min_chunk_length <= len(chunk.strip()) <= self.max_chunk_length
            and not chunk.isspace()
            and len(chunk.split()) > 5  # Ensure minimum word count
        ]

    def process_folders(self) -> Dict:
        """Process all documents in the specified folders"""
        folders = glob.glob(self.data_folders)
        if not folders:
            raise ValueError(f"No folders found in {self.data_folders}")
            
        # Load previously processed metadata
        try:
            with open(self.document_metadata_dump, 'r') as f:
                previous_metadata = json.load(f)
        except FileNotFoundError:
            previous_metadata = {}
        
        documents_metadata = previous_metadata.copy()
        all_chunks = []
        all_metadata = []
        
        for folder in folders:
            folder_name = os.path.basename(folder)
            self.logger.info(f"Processing folder: {folder_name}")
            
            # Process Word documents
            for doc_path in glob.glob(os.path.join(folder, "*.docx")):
                try:
                    # Check if the document has been processed before
                    current_metadata = self.doc_processor.get_document_metadata(doc_path)
                    if doc_path in previous_metadata and previous_metadata[doc_path] == current_metadata:
                        self.logger.info(f"Skipping already processed document: {doc_path}")
                        continue
                    
                    doc_chunks, doc_metadata = self.process_single_document(doc_path, folder_name)
                    all_chunks.extend(doc_chunks)
                    all_metadata.extend(doc_metadata)
                    documents_metadata[doc_path] = doc_metadata[0] if doc_metadata else {}

                    # Update metadata dump
                    self.update_metadata_dump(documents_metadata)

                except Exception as e:
                    self.logger.error(f"Error processing {doc_path}: {str(e)}")
                    continue
        
        # Update vector store if it exists
        if self.vector_store is not None:
            self.update_vector_store(all_chunks, all_metadata)
        else:
            self.logger.warning("Vector store not initialized. Skipping update.")
        
        return documents_metadata

    def process_single_document(self, doc_path: str, folder_name: str) -> tuple[List[str], List[Dict]]:
        """Process a single document and prepare it for vectorization"""
        self.logger.info(f"Processing document: {doc_path}")
        
        # Extract text and tables
        text_chunks, tables = self.doc_processor.process_document(doc_path)
        metadata = self.doc_processor.get_document_metadata(doc_path)
        
        # Split text into chunks
        processed_chunks = self.text_splitter.split_text('\n\n'.join(text_chunks))
        
        # Handle tables
        for table in tables:
            table_context = {
                'table_summary': f"Table from {folder_name}",
                'table_data': table.to_dict(orient='records')
            }
            table_chunk = json.dumps(table_context, indent=2)
            processed_chunks.append(table_chunk)

        # Filter chunks
        processed_chunks = self.filter_chunks(processed_chunks)
        
        # Create metadata for each chunk
        chunk_metadata = [metadata.copy() for _ in processed_chunks]
        
        return processed_chunks, chunk_metadata
    
    def batch_generator(self, chunks: List[str], metadata: List[Dict], batch_size: int):
        """Generate batches of chunks and metadata"""
        iterator = zip(chunks, metadata)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            batch_chunks, batch_metadata = zip(*batch)
            yield list(batch_chunks), list(batch_metadata)
        
    def update_vector_store(self, chunks: List[str], metadata: List[Dict]):
        """Update the vector store with new documents using batching"""
        if not chunks:
            self.logger.warning("No chunks to add to vector store")
            return
            
        try:
            # Calculate batch size based on average chunk length
            avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            batch_size = min(1000, int(self.max_batch_size / avg_chunk_size))
            self.logger.info(f"Processing in batches of {batch_size} chunks")

            # Process in batches
            total_processed = 0
            for batch_chunks, batch_metadata in self.batch_generator(chunks, metadata, batch_size):
                self.vector_store.add_texts(
                    texts=batch_chunks,
                    metadatas=batch_metadata
                )
                total_processed += len(batch_chunks)
                self.logger.info(f"Processed {total_processed}/{len(chunks)} chunks")
                
            self.vector_store.persist()
            self.logger.info(f"Successfully added all {len(chunks)} chunks to vector store")
            
        except Exception as e:
            self.logger.error(f"Error updating vector store: {str(e)}")
            raise

    def initialize_vector_store(self):
        """Initialize the vector store"""
        if self.vector_store is None:
            self.vector_store = Chroma(
                collection_name=self.db_name,
                embedding_function=self.embeddings,
                persist_directory=f"./{self.db_name}"
            )
            self.logger.info("Vector store initialized")

    def find_vector_store(self):
        """Find the vector store if it exists"""
        if os.path.exists(f"./{self.db_name}"):
            self.vector_store = Chroma(
                collection_name=self.db_name,
                embedding_function=self.embeddings,
                persist_directory=f"./{self.db_name}"
            )
            self.logger.info("Vector store found and loaded")
        else:
            self.logger.warning("Vector store not found. Creating a new one.")
            self.initialize_vector_store()

    def clear_vector_store(self):
        """Clear the vector store"""
        if os.path.exists(f"./{self.db_name}"):
            import shutil
            shutil.rmtree(f"./{self.db_name}")
            self.logger.info("Vector store cleared")
        self.vector_store = None

    def update_metadata_dump(self, new_metadata: Dict):
        """Update the metadata dump file with new or changed metadata"""
        try:
            with open(self.document_metadata_dump, 'r') as f:
                current_metadata = json.load(f)
        except FileNotFoundError:
            current_metadata = {}

        # Update current metadata with new metadata
        current_metadata.update(new_metadata)

        with open(self.document_metadata_dump, 'w') as f:
            json.dump(current_metadata, f, indent=2, default=str)
        self.logger.info(f"Metadata dump updated: {self.document_metadata_dump}")

class ImprovedDocumentRetriever:
    def __init__(
        self,
        vector_store,
        embedding_function,
        llm,
        k: int = 4,
        similarity_threshold: float = 0.7
    ):
        self.base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k * 2}  # Fetch more initially for filtering
        )
        
        # Create compression pipeline
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""])
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding_function)
        relevant_filter = EmbeddingsFilter(embeddings=embedding_function, similarity_threshold=similarity_threshold)
        compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter],
        )
        
        # Create the compression retriever
        self.retriever = ContextualCompressionRetriever(
            base_retriever=self.base_retriever,
            base_compressor=compressor,
        )
        
    def _get_document_hash(self, content: str) -> str:
        """Generate a hash for document content to identify duplicates"""
        return hashlib.md5(content.encode()).hexdigest()
        
    async def get_relevant_documents(self, query: str) -> List[Dict]:
        """Get relevant documents with deduplication"""
        docs = await self.retriever.aget_relevant_documents(query)
        
        # Additional deduplication by content hash
        seen_hashes = set()
        unique_docs = []
        
        for doc in docs:
            doc_hash = self._get_document_hash(doc.page_content)
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_docs.append(doc)
                
        return unique_docs[:self.base_retriever.search_kwargs["k"] // 2]

def create_chatbot(llm_name: str) -> object:
    """Create a chatbot instance"""
    if llm_name == "gpt-4o-mini":
        return ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
    elif llm_name == "llama3.2":
        return ChatOpenAI(temperature=0.7, model_name='llama3.2', base_url='http://localhost:11434/v1', api_key='ollama')
    else:
        raise ValueError(f"Unsupported LLM name: {llm_name}")
    
def establish_memory() -> object:
    """Establish memory for the chatbot"""
    from langchain.memory import ConversationBufferMemory
    return ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key ='answer',
        max_token_limit = 1000 # Adjust as needed
    )

def setup_conversation_chain(processor: DocumentProcessor, k: int, llm_name: str) -> object:
    """Set up the conversation chain with improved retrieval"""
    llm = create_chatbot(llm_name)
    memory = establish_memory()
    
    # Create improved retriever
    retriever = ImprovedDocumentRetriever(
        vector_store=processor.vector_store,
        embedding_function=processor.embeddings,
        llm=llm,
        k=k,
        similarity_threshold=0.7
    )

    # Set up the conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever.retriever,
        memory=memory,
        callbacks=[StdOutCallbackHandler()],
        return_source_documents=True,  # Helpful for debugging
        output_key='answer'  # Explicitly set the output key
    )

    return conversation_chain



if __name__ == "__main__":
    processor = DocumentProcessor(
        model="gpt-4o-mini",
        db_name="vector_db",
        data_folders="KnowledgeBase/*",
        document_metadata_dump="documents_metadata.json",
        memory_k=5,
        chunk_size=1000,
        chunk_overlap=50,
        max_batch_size=40000,
        min_chunk_length=500,
        max_chunk_length=2000
    )
    
    try:
        documents_metadata = processor.process_folders() 
        print("Documents processed successfully.")

         # Set up the conversation chain with more reasonable retrieval count
        NUM_RETRIEVE = 50  # Reduced from 50
        LLM_NAME = "gpt-4o-mini"
        conversation_chain = setup_conversation_chain(processor, NUM_RETRIEVE, LLM_NAME)
        print("Conversation chain set up successfully.")

        # Add debug chat function
        def chat_with_sources(question, history):
            result = conversation_chain.invoke({
                "question": question,
                "chat_history": history[-5:] if len(history) > 5 else history
            })
            return f"{result['answer']}\n\nSources: {[doc.metadata for doc in result['source_documents']]}"

        # Use debug version for development
        view = gr.ChatInterface(chat_with_sources, type="messages").launch(inbrowser=True)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")