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


CT_study_uids = []

for i in range(1,47) : 
    df = pd.read_csv(f"/cache/fast_data_nas8/qct/shubham/mongodb_query_datav3/pacs_db_info_{i}.csv")
    study_uids = df.StudyInstanceUID.unique().tolist()
    CT_study_uids.extend(study_uids)
         
pacs_db = get_mongo_db_pacs()

c = 0
chunk_size = 10000  # Adjust the chunk size as needed

out = []

fields = {
    "StudyInstanceUID": 1,
    "Report": 1,
    "_id": 0  # Exclude the MongoDB internal ID
}

pbar = tqdm()
print("Starting query")

try:
    for i in range(0, len(CT_study_uids) , chunk_size) :
        query = {
        "_id": {"$in": CT_study_uids[i:i+chunk_size]}
    }
        cursor = pacs_db.dicoms.find(query,fields).batch_size(1000)
        for document in cursor:
            if document is not None:
                if document == {}:
                    continue

                out.append(document)
                pbar.update(1)

                if len(out) % (chunk_size) == 0 :
                    c +=1
                    df = pd.DataFrame(out)
                    df.to_csv(f"/cache/fast_data_nas8/qct/shubham/mongodb_query_datav2/pacs_db_reports_{c}.csv", index=False)
                    out = []
                    print(f"Batch {c} saved. Memory usage: {monitor_memory()} MB")

        cursor.close()

except Exception as e:
    print(f"An error occurred: {e}")
    
finally:
    # Save any remaining documents
    if out:
        c += 1
        df = pd.DataFrame(out)
        df.to_csv(f"/cache/fast_data_nas8/qct/shubham/mongodb_query_datav2/pacs_db_reports_{c}.csv", index=False)
        print(f"Saving final batch {c}")

    cursor.close()  # Close the cursor
    pbar.close()
    print("Data extraction complete")
    print(f"Total memory usage: {monitor_memory()} MB")
