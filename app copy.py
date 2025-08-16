import json
import os
import requests
from typing import List
import threading
import time
import hashlib
from datetime import datetime, timedelta
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from fastapi import FastAPI
from pydantic import BaseModel
from confluent_kafka import Consumer, KafkaError
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CHP Data Q&A Model")

# Make sure the app instance is available at module level
# This ensures ASGI servers can find it
__all__ = ["app"]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:9000", 
        "http://localhost:9001", 
        "http://localhost:9002", 
        "http://localhost:9003",
        "http://127.0.0.1:9000", 
        "http://127.0.0.1:9001", 
        "http://127.0.0.1:9002", 
        "http://127.0.0.1:9003",
        "https://preview--ask-livinggoods-ai.lovable.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

# Global variables
documents = []
qa_chain = None
pending_updates = []  # Buffer for pending updates
last_rag_build_time = None
data_hash = None  # Hash of current data to detect changes
rag_build_lock = threading.Lock()
UPDATE_INTERVAL = 120  # 2 minutes in seconds

def calculate_data_hash(docs: List[Document]) -> str:
    """Calculate hash of document contents to detect changes"""
    content = ''.join([doc.page_content for doc in sorted(docs, key=lambda x: x.page_content)])
    return hashlib.md5(content.encode()).hexdigest()

def build_or_refresh_rag(force_rebuild=False):
    """Build RAG model only if there are changes or forced rebuild"""
    global qa_chain, last_rag_build_time, data_hash
    
    with rag_build_lock:
        if not documents:
            print("No documents to build RAG.")
            return False
            
        # Calculate current data hash
        current_hash = calculate_data_hash(documents)
        
        # Check if rebuild is needed
        if not force_rebuild and current_hash == data_hash and qa_chain is not None:
            print("No changes detected, skipping RAG rebuild.")
            return False
        
        print(f"Building RAG model with {len(documents)} documents...")
        start_time = time.time()
        
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(documents, embeddings)
            model_name = os.getenv('LLM_MODEL', 'phi3')
            llm = Ollama(model=model_name, base_url=os.getenv('OLLAMA_URL', 'http://localhost:11434'))
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            # Update tracking variables
            data_hash = current_hash
            last_rag_build_time = datetime.now()
            
            build_time = time.time() - start_time
            print(f"RAG model built successfully in {build_time:.2f} seconds!")
            return True
            
        except Exception as e:
            print(f"Error building RAG model: {e}")
            return False

def add_pending_update(new_documents: List[Document], source: str):
    """Add documents to pending updates buffer"""
    global pending_updates
    pending_updates.append({
        'documents': new_documents,
        'source': source,
        'timestamp': datetime.now()
    })
    print(f"Added {len(new_documents)} documents to pending updates from {source}")

def process_pending_updates():
    """Process all pending updates and merge with existing documents"""
    global documents, pending_updates
    
    if not pending_updates:
        return False
    
    print(f"Processing {len(pending_updates)} pending updates...")
    
    # Create a map of existing documents by their unique identifier
    existing_docs_map = {}
    for doc in documents:
        record_id = doc.metadata.get('record_id')
        chp_id = doc.metadata.get('chp_id')
        key = f"{record_id}_{chp_id}" if chp_id else str(record_id)
        existing_docs_map[key] = doc
    
    # Process pending updates
    new_docs_added = 0
    updated_docs = 0
    
    for update in pending_updates:
        for new_doc in update['documents']:
            record_id = new_doc.metadata.get('record_id')
            chp_id = new_doc.metadata.get('chp_id')
            key = f"{record_id}_{chp_id}" if chp_id else str(record_id)
            
            if key in existing_docs_map:
                # Update existing document if content changed
                if existing_docs_map[key].page_content != new_doc.page_content:
                    existing_docs_map[key] = new_doc
                    updated_docs += 1
            else:
                # Add new document
                existing_docs_map[key] = new_doc
                new_docs_added += 1
    
    # Rebuild documents list
    documents = list(existing_docs_map.values())
    
    # Clear pending updates
    pending_updates = []
    
    print(f"Processed updates: {new_docs_added} new docs, {updated_docs} updated docs")
    return new_docs_added > 0 or updated_docs > 0

def periodic_rag_updater():
    """Background thread that processes updates every 2 minutes"""
    while True:
        try:
            time.sleep(UPDATE_INTERVAL)
            
            # Process pending updates
            if process_pending_updates():
                # Rebuild RAG only if there were actual changes
                build_or_refresh_rag()
            else:
                print("No pending updates to process")
                
        except Exception as e:
            print(f"Error in periodic RAG updater: {e}")

def process_cha_dashboard_message(message):
    """Process CHA dashboard message and add to pending updates"""
    try:
        data = json.loads(message.value.decode('utf-8'))
        chps = data.get('chps', [])
        new_documents = []
        
        for chp in chps:
            chp_id = chp.get('chpId')
            chp_username = chp.get('chpUsername', 'unknown')
            chp_email = chp.get('chpEmail', 'N/A')
            stats = chp.get('stats', {})
            
            for record in chp.get('commodityRecords', []):
                doc_text = (
                    f"CHP ID: {chp_id}, Username: {chp_username}, Email: {chp_email}. "
                    f"Record ID: {record['id']}. "
                    f"Community Unit: {record['communityUnitName']}, ID: {record['communityUnitId']}. "
                    f"Commodity: {record['commodityName']}, ID: {record['commodityId']}. "
                    f"Quantity Expired: {record['quantityExpired']}. "
                    f"Quantity Damaged: {record['quantityDamaged']}. "
                    f"Stock On Hand: {record['stockOnHand']}. "
                    f"Quantity Issued: {record['quantityIssued']}. "
                    f"Excess Quantity Returned: {record['excessQuantityReturned']}. "
                    f"Quantity Consumed: {record['quantityConsumed']}. "
                    f"Closing Balance: {record['closingBalance']}. "
                    f"Last Restock Date: {record['lastRestockDate']}. "
                    f"Stock Out Date: {record['stockOutDate']}. "
                    f"Consumption Period: {record['consumptionPeriod']}. "
                    f"Record Date: {record['recordDate']}. "
                    f"Created By Username: {record['createdByUsername'] or 'N/A'}. "
                    f"County: {record['countyName']}, ID: {record['countyId']}. "
                    f"Sub County: {record['subCountyName']}. "
                    f"Ward: {record['wardName']}. "
                    f"Facility: {record['facilityName']}, ID: {record['facilityId']}. "
                    f"Earliest Expiry Date: {record['earliestExpiryDate']}. "
                    f"Quantity To Order: {record['quantityToOrder']}. "
                    f"CHA Name: {record['chaName'] or 'N/A'}. "
                    f"CHP Username: {record['chpUsername'] or 'N/A'}. "
                    f"Stats: Total Records: {stats.get('totalRecords', 0)}, "
                    f"Total Issued: {stats.get('totalIssued', 0)}, "
                    f"Total Consumed: {stats.get('totalConsumed', 0)}, "
                    f"Total Expired: {stats.get('totalExpired', 0)}, "
                    f"Total Damaged: {stats.get('totalDamaged', 0)}, "
                    f"Commodities to Reorder: {', '.join(stats.get('commoditiesToReorder', []))}. "
                    f"Advice: {stats.get('advice', 'N/A')}."
                )
                new_documents.append(Document(
                    page_content=doc_text,
                    metadata={"chp_id": chp_id, "record_id": record['id'], "topic": "cha-dashboard"}
                ))
        
        add_pending_update(new_documents, "cha-dashboard")
        
    except Exception as e:
        print(f"Error processing cha-dashboard message: {e}")

def process_commodity_records_message(message):
    """Process commodity records message and add to pending updates"""
    try:
        raw_data = json.loads(message.value.decode('utf-8'))['data']
        new_documents = []
        
        for item in raw_data:
            doc_text = (
                f"Record ID: {item['id']}. "
                f"Community Unit: {item['communityUnitName']}, ID: {item['communityUnitId']}. "
                f"Commodity: {item['commodityName']}, ID: {item['commodityId']}. "
                f"Quantity Expired: {item['quantityExpired']}. "
                f"Quantity Damaged: {item['quantityDamaged']}. "
                f"Stock On Hand: {item['stockOnHand']}. "
                f"Quantity Issued: {item['quantityIssued']}. "
                f"Excess Quantity Returned: {item['excessQuantityReturned']}. "
                f"Quantity Consumed: {item['quantityConsumed']}. "
                f"Closing Balance: {item['closingBalance']}. "
                f"Last Restock Date: {item['lastRestockDate']}. "
                f"Stock Out Date: {item['stockOutDate']}. "
                f"Consumption Period: {item['consumptionPeriod']}. "
                f"Record Date: {item['recordDate']}. "
                f"Created By Username: {item['createdByUsername'] or 'N/A'}. "
                f"County: {item['countyName']}, ID: {item['countyId']}. "
                f"Sub County: {item['subCountyName']}. "
                f"Ward: {item['wardName']}. "
                f"Facility: {item['facilityName']}, ID: {item['facilityId']}. "
                f"Earliest Expiry Date: {item['earliestExpiryDate']}. "
                f"Quantity To Order: {item['quantityToOrder']}. "
                f"CHA Name: {item['chaName'] or 'N/A'}. "
                f"CHP Username: {item['chpUsername'] or 'N/A'}."
            )
            new_documents.append(Document(
                page_content=doc_text,
                metadata={"record_id": item['id'], "topic": "commodity-records"}
            ))
        
        add_pending_update(new_documents, "commodity-records")
        
    except Exception as e:
        print(f"Error processing commodity-records message: {e}")

def start_kafka_consumer():
    """Start Kafka consumer (fixed import issue)"""
    kafka_broker = os.getenv('KAFKA_BROKER', 'kafka:9092')
    try:
        # Use confluent_kafka.Consumer instead of KafkaConsumer
        consumer = Consumer({
            'bootstrap.servers': kafka_broker,
            'group.id': 'chp-model-group',
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
        })
        
        consumer.subscribe(['cha-dashboard', 'commodity-records'])
        print(f"Connected to Kafka at {kafka_broker}")
        
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Kafka error: {msg.error()}")
                    break
            
            # Process message
            if msg.topic() == 'cha-dashboard':
                process_cha_dashboard_message(msg)
            elif msg.topic() == 'commodity-records':
                process_commodity_records_message(msg)
                
    except Exception as e:
        print(f"Kafka consumer error: {e}")

def load_initial_data():
    """Load initial data from API or sample data"""
    global documents
    try:
        response = requests.get("http://backend:9000/api/records", timeout=5)
        if response.status_code == 200:
            raw_data = response.json()['data']
            for item in raw_data:
                doc_text = (
                    f"Record ID: {item['id']}. "
                    f"Community Unit: {item['communityUnitName']}, ID: {item['communityUnitId']}. "
                    f"Commodity: {item['commodityName']}, ID: {item['commodityId']}. "
                    f"Quantity Expired: {item['quantityExpired']}. "
                    f"Quantity Damaged: {item['quantityDamaged']}. "
                    f"Stock On Hand: {item['stockOnHand']}. "
                    f"Quantity Issued: {item['quantityIssued']}. "
                    f"Excess Quantity Returned: {item['excessQuantityReturned']}. "
                    f"Quantity Consumed: {item['quantityConsumed']}. "
                    f"Closing Balance: {item['closingBalance']}. "
                    f"Last Restock Date: {item['lastRestockDate']}. "
                    f"Stock Out Date: {item['stockOutDate']}. "
                    f"Consumption Period: {item['consumptionPeriod']}. "
                    f"Record Date: {item['recordDate']}. "
                    f"Created By Username: {item['createdByUsername'] or 'N/A'}. "
                    f"County: {item['countyName']}, ID: {item['countyId']}. "
                    f"Sub County: {item['subCountyName']}. "
                    f"Ward: {item['wardName']}. "
                    f"Facility: {item['facilityName']}, ID: {item['facilityId']}. "
                    f"Earliest Expiry Date: {item['earliestExpiryDate']}. "
                    f"Quantity To Order: {item['quantityToOrder']}. "
                    f"CHA Name: {item['chaName'] or 'N/A'}. "
                    f"CHP Username: {item['chpUsername'] or 'N/A'}."
                )
                documents.append(Document(
                    page_content=doc_text,
                    metadata={"record_id": item['id'], "topic": "initial"}
                ))
        else:
            raise ValueError("API fetch failed")
    except Exception as e:
        print(f"Initial data load failed: {e}. Using sample cha-dashboard data.")
        # Sample data loading code here (same as your original)
        sample_json = '''{
            "chps": [
                {
                    "chpId": 4,
                    "chpUsername": "chp1",
                    "chpEmail": "chp@gmail.com",
                    "phoneNumber": null,
                    "commodityRecords": [
                        {
                            "id": 1,
                            "communityUnitId": 1,
                            "communityUnitName": "Kobura",
                            "commodityId": 1,
                            "commodityName": "Panadol",
                            "quantityExpired": 1,
                            "quantityDamaged": 2,
                            "stockOnHand": 1,
                            "quantityIssued": 4,
                            "excessQuantityReturned": 55,
                            "quantityConsumed": 69,
                            "closingBalance": 0,
                            "lastRestockDate": [2025, 8, 1, 0, 0],
                            "stockOutDate": [2025, 7, 31, 0, 0],
                            "consumptionPeriod": 1,
                            "recordDate": [2025, 8, 2, 21, 5, 13, 422170000],
                            "createdByUsername": null,
                            "countyName": "kISUMU",
                            "subCountyName": "Nyando",
                            "wardName": "Kobura",
                            "countyId": 1,
                            "subCountyId": 1,
                            "wardId": 1,
                            "facilityId": 1,
                            "facilityName": "Okana Linked facility",
                            "earliestExpiryDate": [2025, 8, 4, 0, 0],
                            "quantityToOrder": 104,
                            "chaName": null,
                            "chpUsername": null,
                            "createdBy": null
                        }
                    ],
                    "stats": {
                        "totalRecords": 2,
                        "totalIssued": 8,
                        "totalConsumed": 73,
                        "totalExpired": 4,
                        "totalDamaged": 6,
                        "totalOutOfStock": 0,
                        "commoditiesToReorder": ["Panadol", "Panadol Extra"],
                        "commoditiesInExcess": ["Panadol", "Panadol Extra"],
                        "slowMovingCommodities": ["Panadol Extra"],
                        "outOfStockCommodities": [],
                        "advice": "Reorder: Panadol, Panadol Extra. Excess: Panadol, Panadol Extra. Slow moving: Panadol Extra.",
                        "forecast": {"Panadol": 2070.0}
                    }
                }
            ]
        }'''
        
        raw_data = json.loads(sample_json)['chps']
        for chp in raw_data:
            chp_id = chp.get('chpId')
            chp_username = chp.get('chpUsername', 'unknown')
            chp_email = chp.get('chpEmail', 'N/A')
            stats = chp.get('stats', {})
            for record in chp.get('commodityRecords', []):
                doc_text = (
                    f"CHP ID: {chp_id}, Username: {chp_username}, Email: {chp_email}. "
                    f"Record ID: {record['id']}. "
                    f"Community Unit: {record['communityUnitName']}, ID: {record['communityUnitId']}. "
                    f"Commodity: {record['commodityName']}, ID: {record['commodityId']}. "
                    f"Quantity Expired: {record['quantityExpired']}. "
                    f"Quantity Damaged: {record['quantityDamaged']}. "
                    f"Stock On Hand: {record['stockOnHand']}. "
                    f"Quantity Issued: {record['quantityIssued']}. "
                    f"Excess Quantity Returned: {record['excessQuantityReturned']}. "
                    f"Quantity Consumed: {record['quantityConsumed']}. "
                    f"Closing Balance: {record['closingBalance']}. "
                    f"Last Restock Date: {record['lastRestockDate']}. "
                    f"Stock Out Date: {record['stockOutDate']}. "
                    f"Consumption Period: {record['consumptionPeriod']}. "
                    f"Record Date: {record['recordDate']}. "
                    f"Created By Username: {record['createdByUsername'] or 'N/A'}. "
                    f"County: {record['countyName']}, ID: {record['countyId']}. "
                    f"Sub County: {record['subCountyName']}. "
                    f"Ward: {record['wardName']}. "
                    f"Facility: {record['facilityName']}, ID: {record['facilityId']}. "
                    f"Earliest Expiry Date: {record['earliestExpiryDate']}. "
                    f"Quantity To Order: {record['quantityToOrder']}. "
                    f"CHA Name: {record['chaName'] or 'N/A'}. "
                    f"CHP Username: {record['chpUsername'] or 'N/A'}. "
                    f"Stats: Total Records: {stats.get('totalRecords', 0)}, "
                    f"Total Issued: {stats.get('totalIssued', 0)}, "
                    f"Total Consumed: {stats.get('totalConsumed', 0)}, "
                    f"Total Expired: {stats.get('totalExpired', 0)}, "
                    f"Total Damaged: {stats.get('totalDamaged', 0)}, "
                    f"Commodities to Reorder: {', '.join(stats.get('commoditiesToReorder', []))}. "
                    f"Advice: {stats.get('advice', 'N/A')}."
                )
                documents.append(Document(
                    page_content=doc_text,
                    metadata={"chp_id": chp_id, "record_id": record['id'], "topic": "initial"}
                ))

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("Starting CHP Data Q&A Model...")
    
    # Load initial data
    load_initial_data()
    
    # Build initial RAG model
    build_or_refresh_rag(force_rebuild=True)
    
    # Start background threads
    kafka_thread = threading.Thread(target=start_kafka_consumer, daemon=True)
    kafka_thread.start()
    
    updater_thread = threading.Thread(target=periodic_rag_updater, daemon=True)
    updater_thread.start()
    
    print("Application started successfully!")

@app.post("/ask")
async def ask_question(question: Question):
    """Answer question using the RAG model"""
    if qa_chain is None:
        return {"error": "Model not initialized"}
    
    try:
        result = qa_chain({"query": question.query})
        return {
            "answer": result["result"],
            "sources": [doc.page_content[:200] + "..." for doc in result["source_documents"]],
            "last_updated": last_rag_build_time.isoformat() if last_rag_build_time else None
        }
    except Exception as e:
        return {"error": f"Error processing question: {str(e)}"}

@app.post("/refresh")
async def refresh_data():
    """Force refresh of RAG model"""
    success = build_or_refresh_rag(force_rebuild=True)
    return {
        "status": "Data refreshed successfully" if success else "Failed to refresh data",
        "last_updated": last_rag_build_time.isoformat() if last_rag_build_time else None
    }

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "documents_count": len(documents),
        "pending_updates": len(pending_updates),
        "last_rag_build": last_rag_build_time.isoformat() if last_rag_build_time else None,
        "model_initialized": qa_chain is not None,
        "data_hash": data_hash
    }

@app.get("/analyze/chp-stats")
async def analyze_chp_stats():
    """Analyze CHP statistics"""
    chp_stats = {}
    for doc in documents:
        chp_id = doc.metadata.get('chp_id')
        if chp_id:
            if chp_id not in chp_stats:
                chp_stats[chp_id] = {'records': 0, 'totalConsumed': 0, 'commodities': set()}
            chp_stats[chp_id]['records'] += 1
            for line in doc.page_content.split('. '):
                if 'Total Consumed' in line:
                    try:
                        chp_stats[chp_id]['totalConsumed'] += int(line.split(': ')[1])
                    except (IndexError, ValueError):
                        pass
                if 'Commodity: ' in line:
                    try:
                        chp_stats[chp_id]['commodities'].add(line.split(': ')[1].split(',')[0])
                    except IndexError:
                        pass
    
    return {
        "stats": {
            chp_id: {
                "records": stats['records'],
                "totalConsumed": stats['totalConsumed'],
                "commodities": list(stats['commodities'])
            } for chp_id, stats in chp_stats.items()
        },
        "last_updated": last_rag_build_time.isoformat() if last_rag_build_time else None
    }