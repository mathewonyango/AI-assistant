import json
import os
import requests
from typing import List
import threading
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from fastapi import FastAPI
from pydantic import BaseModel
from confluent_kafka import Consumer, KafkaError
from fastapi.middleware.cors import CORSMiddleware  # Add this line

app = FastAPI(title="CHP Data Q&A Model")  # Ensure FastAPI instance is named 'app'


# Add CORS middleware - ADD THIS SECTION RIGHT HERE
# Add CORS middleware - ADD THIS SECTION RIGHT HERE
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
        "https://preview--ask-livinggoods-ai.lovable.app"    ],  # Your React app ports
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

documents = []
qa_chain = None

def build_or_refresh_rag():
    global qa_chain
    if documents:
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
        print("RAG model refreshed!")
    else:
        print("No documents to build RAG.")

def process_cha_dashboard_message(message):
    global documents
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
        documents.clear()  # Replace with new data
        documents.extend(new_documents)
        build_or_refresh_rag()
        print(f"Processed {len(new_documents)} records from Kafka topic 'cha-dashboard'.")
    except Exception as e:
        print(f"Error processing cha-dashboard message: {e}")

def process_commodity_records_message(message):
    global documents
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
        documents.extend(new_documents)  # Append to existing documents
        build_or_refresh_rag()
        print(f"Processed {len(new_documents)} records from Kafka topic 'commodity-records'.")
    except Exception as e:
        print(f"Error processing commodity-records message: {e}")

def start_kafka_consumer():
    kafka_broker = os.getenv('KAFKA_BROKER', 'kafka:9092')
    try:
        consumer = KafkaConsumer(
            ['cha-dashboard', 'commodity-records'],
            bootstrap_servers=[kafka_broker],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='chp-model-group',
            value_deserializer=lambda x: x
        )
        print(f"Connected to Kafka at {kafka_broker}")
        for message in consumer:
            if message.topic == 'cha-dashboard':
                process_cha_dashboard_message(message)
            elif message.topic == 'commodity-records':
                process_commodity_records_message(message)
    except KafkaError as e:
        print(f"Kafka consumer error: {e}")

def load_initial_data():
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
                        },
                        {
                            "id": 2,
                            "communityUnitId": 1,
                            "communityUnitName": "Kobura",
                            "commodityId": 2,
                            "commodityName": "Panadol Extra",
                            "quantityExpired": 3,
                            "quantityDamaged": 4,
                            "stockOnHand": 1,
                            "quantityIssued": 4,
                            "excessQuantityReturned": 4,
                            "quantityConsumed": 4,
                            "closingBalance": 0,
                            "lastRestockDate": [2025, 7, 30, 0, 0],
                            "stockOutDate": [2025, 7, 30, 0, 0],
                            "consumptionPeriod": 0,
                            "recordDate": [2025, 8, 2, 21, 7, 10, 467248000],
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
                            "quantityToOrder": 6,
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
            ],
            "stats": {
                "totalRecords": 2,
                "totalIssued": 8,
                "totalConsumed": 73,
                "totalExpired": 4,
                "totalDamaged": 6,
                "totalClosingBalance": 0,
                "stats": null
            },
            "advice": ""
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
    load_initial_data()
    build_or_refresh_rag()
    consumer_thread = threading.Thread(target=start_kafka_consumer, daemon=True)
    consumer_thread.start()

@app.post("/ask")
async def ask_question(question: Question):
    if qa_chain is None:
        return {"error": "Model not initialized"}
    result = qa_chain({"query": question.query})
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] + "..." for doc in result["source_documents"]]
    }

@app.post("/refresh")
async def refresh_data():
    build_or_refresh_rag()
    return {"status": "Data refreshed"}

@app.get("/analyze/chp-stats")
async def analyze_chp_stats():
    chp_stats = {}
    for doc in documents:
        chp_id = doc.metadata.get('chp_id')
        if chp_id:
            if chp_id not in chp_stats:
                chp_stats[chp_id] = {'records': 0, 'totalConsumed': 0, 'commodities': set()}
            chp_stats[chp_id]['records'] += 1
            for line in doc.page_content.split('. '):
                if 'Total Consumed' in line:
                    chp_stats[chp_id]['totalConsumed'] += int(line.split(': ')[1])
                if 'Commodity: ' in line:
                    chp_stats[chp_id]['commodities'].add(line.split(': ')[1])
    return {
        "stats": {
            chp_id: {
                "records": stats['records'],
                "totalConsumed": stats['totalConsumed'],
                "commodities": list(stats['commodities'])
            } for chp_id, stats in chp_stats.items()
        }
    }