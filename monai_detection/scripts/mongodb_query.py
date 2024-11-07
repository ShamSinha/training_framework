import pymongo
from tqdm.auto import tqdm
import pandas as pd
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Return memory usage in MB



def get_mongo_db_pacs(db_name="PACS", user_name="pacs_query", password="shinypoint41"):
    client = pymongo.MongoClient("mongodb://192.168.1.58:27017")
    db = client[db_name]
    db.authenticate(user_name, password)
    return db 

# Define your specific query
# query = {
#     "Report": {"$exists": True}
# }

query = {"StudyInstanceUID": {"$exists": True},
         "SeriesInstanceUID": {"$exists": True},
         "Modality": "CT"}
         
pacs_db = get_mongo_db_pacs()

chunk_size = 100000  # Adjust the chunk size as needed
c = 0

# total_documents = pacs_db.dicoms.count()

out = []

# cursor = pacs_db.dicoms.find(query, {"StudyInstanceUID":1 ,  "_id":False,  "Report":1}).batch_size(1000)


cursor = pacs_db.dicoms.find(query, {"SeriesInstanceUID":1 ,"StudyInstanceUID":1,  "_id":False,  "Modality":1, "BodyPartExamined":1 , "SeriesDescription":1}).batch_size(1000)

 
pbar = tqdm()

print("Starting query")

try:
    for document in cursor:
        if document is not None:
            if document == {}:
                continue
            out.append(document)
            pbar.update(1)

            if len(out) % (chunk_size) == 0 :
                c +=1
                df = pd.DataFrame(out)
                df.to_csv(f"/cache/fast_data_nas8/qct/shubham/mongodb_query_datav3/pacs_db_info_{c}.csv", index=False)
                out = []
                gc.collect()  # Force garbage collection
                print(f"Batch {c} saved. Memory usage: {monitor_memory()} MB")

except Exception as e:
    print(f"An error occurred: {e}")
    
finally:
    # Save any remaining documents
    if out:
        c += 1
        df = pd.DataFrame(out)
        df.to_csv(f"/cache/fast_data_nas8/qct/shubham/mongodb_query_datav3/pacs_db_info_{c}.csv", index=False)
        print(f"Saving final batch {c}")

    cursor.close()  # Close the cursor
    pbar.close()
    gc.collect()  # Force garbage collection one last time
    print("Data extraction complete")
    print(f"Total memory usage: {monitor_memory()} MB")
