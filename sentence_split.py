import time
import ray
import kss
import psutil
# from itertools import chain

from tqdm import tqdm
from dbConn.mongo_conn import config

num_cpus = psutil.cpu_count(psutil.cpu_count(logical=False)) # 10

def get_seq(last_id=None):
    conn = config()
    conn_db = conn["travel_ai"]
    if not last_id:
        last_id = conn_db.blog_contents.find_one(sort=[("doc_id", -1)])['doc_id']
    conn.close()
    return [i for i in range(last_id+1)]


@ray.remote # 2.5G (memory=2500*1024*1024)
def sent_split(seq):
    conn = config()
    conn_db = conn["travel_ai"]
    
    for idx in tqdm(seq):
        content = conn_db.blog_contents.find_one({"doc_id": idx})['doc']
        splited = kss.split_sentences(content)
        conn_db.blog_contents.update_one({"doc_id": idx}, {"$set":{"num_docs": len(splited), "docs": splited}})
    
    conn.close()


def chunker_list(seq, size):
    return list(seq[i::size] for i in range(size))

term = 100000
for i in range(30768, 258928418, term): # 1 loop 211.8580768108368 
    try:
        ray.init(num_cpus=num_cpus)
        start = time.time()
        sentences_chunk = list(chunker_list([i for i in range(i,i+term)], num_cpus))
        futures = [sent_split.remote(sentences_chunk[j]) for j in range(num_cpus)]
        ray.get(futures)
        print(time.time()-start)
    except:
        pass
    finally:
        ray.shutdown()