import re
import unicodedata

def unicode_norm(sent):
    normalized = unicodedata.normalize("NFC", sent)
    return normalized


def join_sent(sent_list):
    joined_cleaned = []
    for i in range(0, len(sent_list)-1, 2):
        n_token_l, n_token_r = len(sent_list[i].split()), len(sent_list[i+1].split())
        if n_token_l <= 5 or n_token_r <= 5:
            joined_cleaned.append(sent_list[i] + ' ' + sent_list[i+1])
        else:
            joined_cleaned.append(sent_list[i])
            joined_cleaned.append(sent_list[i+1])
    flag = True
    if len(joined_cleaned) == 1 or min([len(nc.split()) for nc in joined_cleaned]) > 5:
        flag = False
    return joined_cleaned, flag


filter_list = ["관심 장소를 플레이스에 저장할 수 있어요.", "팝업 닫기", "내 장소 폴더에 저장했습니다.", "상세보기", "지도보기",
                "플레이스 가기", "관심 장소를 플레이스에 저장할 수 있어요."]

def filter_sent(sent_list):
    filter_sents = []
    for sent in sent_list:
        for filt_sent in filter_list:
            if filt_sent in sent:
                filter_sents.append(sent)
                break
    for filt_sent in filter_sents:
        sent_list.remove(filt_sent)
    
    return sent_list


def cleaning(sent):
    if sent and type(sent) == str and len(sent.split()) > 10:
        sent = unicode_norm(sent)

        splited = sent.split('\n')
        new_splited = []
        for sp in splited:
            new_splited.extend(sp.split('\r'))
        del splited
        
        cleaned = []
        for sp in new_splited:
            # sp = re.sub(r"[{]{2,}", " ", sp)
            # sp = re.sub(r"[}]{2,}", " ", sp)
            # sp = re.sub(r"[<]{2,}", " ", sp)
            # sp = re.sub(r"[>]2,}", " ", sp)
            # sp = re.sub(r"\<[^)]*\>", "", sp) # <> tag
            # sp = re.sub(r"\{[^)]*\}", "", sp) # {} tag
            sp = re.sub(r"[^0-9가-힣,.?:]+", " ", sp) # 주소의 \-
            sp = re.sub(r"[\s]{2,}", " ", sp)  # 공백 한개로
            sp = re.sub(r"[, ]{2,}|[.]{2,}", ", ", sp)
            sp = re.sub(r"[. ]{2,}|[.]{2,}", ". ", sp)
            sp = sp.strip()
            if sp:
                cleaned.append(sp)

        cleaned = filter_sent(cleaned)
        if len(cleaned) < 2:
            return None

        min_token_n = True
        while min_token_n:
            cleaned, min_token_n = join_sent(cleaned)
                
        return cleaned
    else:
        return None
    
    
import gc
import time
import psutil
from tqdm import tqdm

import pymongo
from dbConn.mongo_conn import config

import ray
num_cpus = 10
ray.init(num_cpus=num_cpus, ignore_reinit_error=True, dashboard_host="0.0.0.0", dashboard_port=8265, include_dashboard=True)


@ray.remote
def sent_split(index_chunk):
    conn = config()
    col = conn["travel_ai"].blog_contents
    col_ner = conn["travel_ai"].blog_contents_ner
        
    s, e = index_chunk
    # blog_contents = list(col.find({'$and': [{"num_docs": {"$exists": False}}, {"post_idx": {"$gte": s, "$lte": e}}]}))
    blog_contents = list(col.find({"post_idx": {"$gte": s, "$lte": e}}))
                         
    result_data, i, total_l = [], 0, len(blog_contents)
    for bc in blog_contents:
        cleaned = cleaning(bc['raw_content'])
        
        if cleaned:
            bc['cleaned_content'] = cleaned
            bc['num_docs'] = len(cleaned)
        else:
            col.delete_one({'post_idx': bc['post_idx']})
        i += 1
        result_data.append(bc)
        
        if i % 1000 == 0:
            col_ner.insert_many(result_data)
            result_l = len(result_data)
            print(f'i => {i}, pushed data : {result_l}')
            result_data.clear()
        gc.collect()

    conn.close()


cursor = config()
col = cursor["travel_ai"].blog_contents
# post_idxs = col.find({"num_docs": {"$exists": False}},{'post_idx': 1})
post_idxs = col.find({},{'post_idx': 1})
all_post_idx = [pi['post_idx'] for pi in post_idxs]
all_post_idx.sort()
cursor.close()


def idx_chunker(post_idxs, chunk_num):
    num_idxs = len(post_idxs)
    result = []
    chunk_size = num_idxs // chunk_num
    for i in range(0, num_idxs, chunk_size):
        if len(result) == chunk_num - 1:
            result.append((i, num_idxs-1))
            break
        result.append((i, i+chunk_size-1))
    print(result)
    return [(post_idxs[r[0]], post_idxs[r[-1]]) for r in result]

idx_chunk = idx_chunker(all_post_idx, num_cpus)
futures = [sent_split.remote(idx_chunk[x]) for x in range(num_cpus)]

try:
    ray.get(futures)
finally:
    ray.shutdown()
    
    
def delete_origin():
    conn = config()
    col_ner = conn["travel_ai"].blog_contents_ner
    all_post_idxs = []
    for data in col_ner.find({},{'post_idx': 1}):
        all_post_idxs.append(data['post_idx'])

    col = conn["travel_ai"].blog_contents
    col_f = conn["travel_ai"].blog_contents_filtered
    for post_idx in tqdm(all_post_idxs[1730550+785494:]):
        col.delete_one({'post_idx': post_idx})
        col_f.delete_one({'post_idx': post_idx})
    conn.close()
delete_origin()