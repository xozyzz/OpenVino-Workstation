import os
import re
import openvino as ov
import openvino_genai as ov_genai
import numpy as np
from numpy.linalg import norm


#Path to the Qwen3 embedding model we downloaded.
model_path = r"models\Qwen3-Embedding-0.6B-int8-ov"
#LAST_TOKEN uses the last token's embedding as the sentence embedding, that is how Qwen3 works. normalize=True makes every vector length 1 so the dot product gives the similarity directly.
pipeline = ov_genai.EmbeddingPipeline(model_path, "GPU", pooling_type=ov_genai.TextEmbeddingPipeline.PoolingType.LAST_TOKEN,normalize=True)


def main():
    print(len(join_string()))
    

def read_folder():
    texts = []                                                                          #reading folders to give as output, then would be used for chunking.
    for folder , sub_folder , files in os.walk(r"D:\Workstation_rag_files"):            #walking into the folder and telling us which files are there.      
        if ".git" in sub_folder:
            sub_folder.remove(".git")

        for name in files:
            if name.endswith(".md"):
                full_path = os.path.join(folder,name)
                with open(full_path,encoding="utf-8") as file:                          #opens the .md files.
                    text = file.read()
                    texts.append(text)
    return texts
        

def split_sentences():                                                                  #'re' module to split sentences, not perfect but better than fixed chunking. 
    lines = []
    for read in read_folder():
        line = re.split(r"(?<=[.!?])\s+|\n+", read)                                     # splits sentences and keeps the endings symbols
        lines.append(line)
    return lines


def embed_sentences():                                                                # Openvino's own embedding feature
    embeds = []
    for read in split_sentences():
        embedding = pipeline.embed(read).embeddings                                     #embeding the sentences.
        embeds.append(embedding.data)
    return embeds


def comparison_operator():                                                              #Comparing two sentence and giving them a score.
    multi_scores = []
    for embeds in embed_sentences():
        file_scores = []
        for i in range(len(embeds)-1):
            similarity = np.dot(embeds[i] , embeds[i+1]) / (norm(embeds[i]) * norm(embeds[i+1]))    #We used cosine similarity to compare the direction of two sentence vectors (somehow). Since they are normalized it is basically just a dot product.
            file_scores.append(similarity)
        multi_scores.append(file_scores)
    return multi_scores


def comparison_cut():                                                                   #Using the score as with percentile and storing where to cut the sections.
    file_cuts = []
    scores = comparison_operator()
    for score in scores:
        cuts = []
        if len(score) == 0:
            file_cuts.append([])
            continue
        cutoff = np.percentile(score,10)
        for i in range(len(score)):
            if score[i] < cutoff:
                cuts.append(i)
        file_cuts.append(cuts)
    return file_cuts


def join_chunks():                                                                      #Joining the sentences into chunks using cuts.
    file_chunks = []
    cuts = comparison_cut()
    sentences = split_sentences()
    for cut,sentence in zip(cuts,sentences):                                            #Takes both list as input.
        a = 0
        for c in cut:
            chunks = sentence[a:c+1]                                                    #Makes a chunk till the cut. and loops for next.
            file_chunks.append(chunks)            
            a = c+1
        chunks = sentence[a:]                                                           #Uses the leftover strings after all cuts are used up.
        file_chunks.append(chunks)
    return file_chunks


def join_string():                                                                      #Joining the individual strings inside the chunks into one string.
    string_chunks = []
    chunks = join_chunks()
    for chunk in chunks:
        strings = " ".join(chunk)
        string_chunks.append(strings)
    return string_chunks


'''def fixed_chunking(text,size):                                                          #fixed chunking for learning purposes.
    results = []
    chunks = text.split()
    chunking_range = len(chunks)//size
    remaining = len(chunks)% size
    if remaining != 0:
        chunking_range += 1   
    a , b = 0 , size
    for _ in range(chunking_range):
        results.append(chunks[a:b])
        a += size
        b += size
    return results'''


if __name__ == "__main__":
    main()
