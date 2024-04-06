import editdistance
from heapq import heappop, heappush, heapify
import time, sys, json
import random as rnd
import fasttext.util



VOCABULARY_PATH = "data/SemEval2018-Task9/1B.italian.vocabulary.txt"
K = 5
NEIGHBORS_POOL = 15


def get_vocabulary():
    return set([line.strip().lower() for line in open(VOCABULARY_PATH, 'r')])


def get_possible_distractors(targets: set[str], vocabulary: set[str], embeddings = None):
    return get_possible_distractors_by_edit_dist(targets, vocabulary) if embeddings is None else get_neighbors(targets, embeddings, vocabulary)


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
    with open(hyponym_file, 'r') as file:
        hyponyms = [line.strip().split("\t")[0] for line in file]
    with open(hypernym_file, 'r') as file:
        hypernyms = [line.strip().split("\t") for line in file]
    return {hyponym: hypernyms for hyponym, hypernyms in zip(hyponyms, hypernyms)}

def create_json(data, output_file, vocabulary):
    json_entries = []
    id = 0
    for hyponym, hypernyms in data.items():
        targets = set([hyponym] + hypernyms)
        possible_distractors = get_possible_distractors(targets, vocabulary)
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


def test():
    fasttext.util.download_model('it', if_exists='ignore')  # Italian
    embeddings = fasttext.load_model('cc.it.300.bin')
    print(embeddings.get_nearest_neighbors("cane_gatto", k=1))

def main():
    if len(sys.argv) < 4:
        print("Usage: python managing_data_task_26.py <input_file1> <input_file2> <output_file>")
        sys.exit(1)
    hyponym_file = sys.argv[1]
    hypernym_file = sys.argv[2]
    output_file = sys.argv[3]
    data = get_data(hyponym_file, hypernym_file)
    vocabulary = get_vocabulary()
    create_json(data, output_file, vocabulary)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python managing_data_task_26.py <input_file1> <input_file2> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    test()
