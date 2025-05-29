import json 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv 
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from  langchain.schema import Document 
import os 
from tqdm import tqdm
import uuid 


load_dotenv(dotenv_path = ".env")

class VectorStore: 

    def __init__(self, collection_name: str = "multi-hop-rag"):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model = "models/text-embedding-004", 
            google_api_key = os.getenv("GEMINI_API_KEY")
        )

        self.collection_name = collection_name

        self.qdrant_client = QdrantClient(
            url = os.getenv("QDRANT_URL"), 
            api_key = os.getenv("QDRANT_API_KEY")
        )


        self._setup()

        self.vector_store = QdrantVectorStore(
            client = self.qdrant_client, 
            collection_name = self.collection_name, 
            embedding = self.embedding_model
        )

    def _setup(self): 

        try : 
            all_collections = self.qdrant_client.get_collections()
            collection_names = [collection.name for collection in all_collections.collections]

            if "multi-hop-rag" not in collection_names:

                self.qdrant_client.create_collection(
                    collection_name = "multi-hop-rag", 
                    vectors_config = VectorParams(
                        size = 768, 
                        distance = Distance.COSINE
                    )
                )

        except Exception as e:
            raise Exception(f"Error setting up Qdrant collection: {e}")



    def load_documents_from_json(self, json_path: str): 

        try : 
            with open(json_path, 'r') as file:
                data = json.load(file)

            documents = []
            texts = []
            metadata = []
            points = []
            for item in tqdm(data, desc="Loading documents"):

                title = item.get('title', 'No Title')
                passage = item.get('passage', 'No Passage')
                embedding = item.get('embeddings', None)

                doc = Document(
                    page_content = f"{title}\n{passage}",
                    metadata = {
                        'title': title,
                        'passage': passage,
                    }
                )

                documents.append(doc)
                texts.append(doc.page_content)
                metadata.append(doc.metadata)

                point = PointStruct(
                    id = str(uuid.uuid4()), 
                    vector = embedding if embedding is not None and len(embedding) == 768 else self.embedding_model.embed_query(doc.page_content),
                    payload = doc.metadata
                )

                points.append(point)

                if len(points) >= 100: 
                    self.qdrant_client.upsert(
                        collection_name = self.collection_name, 
                        points = points
                    )
                    points = [] 
            
            if points: 
                self.qdrant_client.upsert(
                    collection_name = self.collection_name, 
                    points = points
                )
            
            print(f"Successfully loaded {len(documents)} documents from {json_path} into the vector store.")

        except Exception as e:
            raise Exception(f"An error occurred while loading documents: {e}")


    def similarity_search(self, query: str, k: int = 5, filter_dict: dict = None):
        try:
            
            query_embedding = self.embedding_model.embed_query(query)

            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k
            )
            
            documents = []
            for result in search_results:
    
                doc = Document(
                    page_content=f"{result.payload.get('title', '')}\n{result.payload.get('passage', '')}",
                    metadata={
                        'title': result.payload.get('title', ''),
                        'passage': result.payload.get('passage', ''),
                        'score': result.score  
                    }
                )
                documents.append(doc)
            
            return documents  
        except Exception as e:
            raise Exception(f"Error during similarity search: {e}")















if __name__ == "__main__": 

    vector_store = VectorStore()
    # vector_store.load_documents_from_json("data/embedded_chunks.json")

    results = vector_store.similarity_search(
        query = "What is the battery capacity of the Tesla Model 3?", 
        k = 5
    )
    for doc in results:
        print(f"Title: {doc.metadata.get('title')}")
        print(f"Passage: {doc.page_content}\n")
        print(f"Score: {doc.metadata.get('score')}\n")
        print("-" * 80)
    



