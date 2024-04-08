import editdistance
from heapq import heappop, heappush, heapify
import time, sys, json
import random as rnd
import fasttext.util
import pyspark as pk



VOCABULARY_PATH = "data/SemEval2018-Task9/1B.italian.vocabulary.txt"
K = 5
NEIGHBORS_POOL = 15


def get_vocabulary():
    return set([line.strip().lower() for line in open(VOCABULARY_PATH, 'r')])

def generate_similar_words_vocab_map(word, vocabulary,embeddings=None, use_fasttext=False, use_embeddings=False, progress_bar=False):
    similar_words = {}
    similar_words[word] = get_vocabulary_neighbors(word, vocabulary, embeddings) if use_fasttext else get_possible_distractors_by_edit_dist({word}, vocabulary)
    return similar_words


def get_possible_distractors(targets: set[str],similar_words):
    distractors = set()
    for target in targets:
        distractors = distractors.union(similar_words[target])
    return distractors


def get_neighbors(targets, embeddings, vocabulary):
    distractors = set()
    for target in targets:
        distractors.union(get_vocabulary_neighbors(target, vocabulary, embeddings))
    return distractors

def get_vocabulary_neighbors(target, vocabulary, embeddings):
    neighbors = embeddings.get_nearest_neighbors(target, k=NEIGHBORS_POOL)
    similar_words = set()
    for neighbor in neighbors:
        if neighbor in vocabulary:
            similar_words.add(neighbor)
        if len(similar_words) == K:
            break
    return similar_words



def get_possible_distractors_by_edit_dist(targets: set[str], vocabulary: set[str]):
    distractors = set()
    for target in targets:
        heap = []
        heapify(heap)
        for word in vocabulary:
            if word not in distractors and word not in targets:
                if len(heap) < K:
                    heappush(heap, (-1 * editdistance.eval(target, word), word))
                elif editdistance.eval(target, word) < -1 * heap[0][0]:
                    heappop(heap)
                    heappush(heap, (-1 * editdistance.eval(target, word), word))
        for word in heap:
            distractors.add(word[1])
    return distractors


def get_distractors(target,target_pos, possible_distractors):
    distractors = rnd.sample(possible_distractors, 3)
    for distractor in distractors:
        possible_distractors.remove(distractor)
    distractors.insert(target_pos, target)
    return distractors

def get_data(hyponym_file, hypernym_file):
    vocabulary = set()
    hyponyms = []
    hypernyms = []
    with open(hyponym_file, 'r') as file:
        for line in file:
            hyponym = line.strip().split("\t")[0]
            hyponyms.append(hyponym)
            vocabulary.add(hyponym)
    with open(hypernym_file, 'r') as file:
        for line in file:
            hypernyms_tmp = line.strip().split("\t")
            hypernyms.append(hypernyms_tmp)
            vocabulary.union(set(hypernyms_tmp))
    return {hyponym: hypernyms for hyponym, hypernyms in zip(hyponyms, hypernyms)}, vocabulary

def create_json(data, output_file, similar_words):
    json_entries = []
    id = 0
    for hyponym, hypernyms in data.items():
        targets = set([hyponym] + hypernyms)
        possible_distractors = get_possible_distractors(targets, similar_words)
        for hypernym in hypernyms:
            target_pos = rnd.randint(0, 3)
            json_entry = {
                "id": id,
                "text": hyponym,
                "choices": get_distractors(hypernym, target_pos,possible_distractors),
                "label": target_pos
            }
            json_entries.append(json_entry)
            id += 1
    with open(output_file, 'w', encoding='utf8') as jsonl_file:
        for item in json_entries:
            json.dump(item, jsonl_file)  # Scrivi l'oggetto JSON
            jsonl_file.write('\n')

def create_json_entries(hyponym, hypernyms, similar_words):
    targets = set([hyponym] + hypernyms)
    json_entries = []
    possible_distractors = get_possible_distractors(targets, similar_words)
    for hypernym in hypernyms:
        target_pos = rnd.randint(0, 3)
        json_entry = {
            "id": id,
            "text": hyponym,
            "choices": get_distractors(hypernym, target_pos, possible_distractors),
            "label": target_pos
        }
        json_entries.append(json_entry)
    return json_entries

def write_json_entries(json_entries, output_file, set_id=False):
    with open(output_file, 'w', encoding='utf8') as jsonl_file:
        for i,item in enumerate(json_entries):
            if set_id:
                item["id"] = i
            json.dump(item, jsonl_file)  # Scrivi l'oggetto JSON
            jsonl_file.write('\n')



def test():
    fasttext.util.download_model('it', if_exists='ignore')  # Italian
    embeddings = fasttext.load_model('cc.it.300.bin')
    print(embeddings.get_nearest_neighbors("cane_gatto", k=1))

def main():
    if len(sys.argv) != 5:
        print("Usage: python managing_data_task_26.py <input_file1> <input_file2> <output_file>")
        sys.exit(1)
    hyponym_file = sys.argv[1]
    hypernym_file = sys.argv[2]
    output_file = sys.argv[3]
    partitions = int(sys.argv[4])
    fasttext.util.download_model('it', if_exists='ignore')  # Italian
    embeddings = fasttext.load_model('cc.it.300.bin')
    sc = pk.SparkContext("local[*]")
    data, vocab = get_data(hyponym_file, hypernym_file)
    vocabulary = get_vocabulary()
    vocabulary = vocabulary.union(vocab)
    vocab = sc.parallelize(vocabulary, partitions)
    mapping = vocab.map(
        lambda x: generate_similar_words_vocab_map(x, vocabulary, embeddings, use_fasttext=True, use_embeddings=False))
    print("generating similar words")
    similar_words = mapping.reduce(
        lambda x, y: {k: x.get(k, []) + y.get(k, []) for k in set(x.keys()).union(set(y.keys()))})
    print("creating json")
    data = sc.parallelize(data.items(), partitions)
    json_entries = data.flatMap(lambda x: create_json_entries(x[0], x[1], similar_words)).collect()
    write_json_entries(json_entries, output_file, set_id=True)
    #create_json(data, output_file, similar_words)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python managing_data_task_26.py <input_file1> <input_file2> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    test()
