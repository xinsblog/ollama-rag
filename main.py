import faiss
import ollama
from tqdm import tqdm
import numpy as np


def encode(text):
    return ollama.embeddings(model='nomic-embed-text', prompt=text)['embedding']


# 读取文档并分段
chunks = []
file = open("洗衣机常见错误编码及解决办法.txt")
for line in file:
    line = line.strip()
    if line:
        chunks.append(line.strip())
file.close()

# 计算每个分段的embedding
chunk_embeddings = []
for i in tqdm(range(len(chunks)), desc='计算chunks的embedding'):
    chunk_embeddings.append(encode(chunks[i]))
chunk_embeddings = np.array(chunk_embeddings)
chunk_embeddings = chunk_embeddings.astype('float32')

# 建立faiss索引
faiss.normalize_L2(chunk_embeddings)
faiss_index = faiss.index_factory(chunk_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
faiss_index.add(chunk_embeddings)

while True:
    # 提示用户输入
    question = input("请输入一个问题: ")
    print(question)

    # 将问题编码
    question_embedding = encode(question)

    # 检索到最相关的top1分段
    question_embedding = np.array([question_embedding])
    question_embedding = question_embedding.astype('float32')
    faiss.normalize_L2(question_embedding)
    _, index_matrix = faiss_index.search(question_embedding, k=1)

    # 构造prompt
    prompt = f'根据参考文档回答问题，回答尽量简洁，不超过20个字\n' \
             f'问题是："{question}"\n' \
             f'参考文档是："{chunks[index_matrix[0][0]]}"'
    print(f'prompt:\n{prompt}')

    # 获取答案
    stream = ollama.chat(model='qwen:4b', messages=[{'role': 'user', 'content': prompt}], stream=True)
    print('answer:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print()
