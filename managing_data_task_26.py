import editdistance
from heapq import heappop, heappush, heapify
import time, sys, json
import random as rnd
import fasttext.util
import pyspark as pk
import numpy as np

# Constants
K = 5
NEIGHBORS_POOL = 15
FAVOURITE_SIMIL_TARGET = 2
THRESHOLD = 7


# Function to get the vocabulary from a file
def get_vocabulary(vocabulary_path: str) -> set[str]:
    return set([line.strip().lower() for line in open(vocabulary_path, 'r')])


# Function to generate a vocabulary of similar words made for running on multiple threads
def generate_similar_words_vocab_map(word: str, vocabulary: set[str], embeddings=None,
                                     use_fasttext: bool = False, load_embeddings: bool = True) -> dict[str, set[str]]:
    if use_fasttext and load_embeddings:
        embeddings = fasttext.load_model('cc.it.300.bin')
    similar_words = {}
    similar_words[word] = get_vocabulary_neighbors(word, vocabulary, embeddings) if use_fasttext else \
            get_possible_distractors_by_edit_dist({word}, vocabulary)
    return similar_words


# Function to generate a vocabulary of similar words running on a single thread
def generate_similar_words_vocab(vocabulary: set[str], embeddings=None, use_fasttext: bool = False, progress_bar=False) -> dict[str, list[str]]:
    similar_words = {}
    for i, word in enumerate(vocabulary):
        if progress_bar:
            print_progress_bar(i / len(vocabulary))
        update = generate_similar_words_vocab_map(word, vocabulary, embeddings, use_fasttext, False)
        similar_words.update(update)
    return similar_words


# Function to print a progress bar
def print_progress_bar(percentuale: float, lunghezza_barra: int = 30) -> None:
    blocchi_compilati = int(lunghezza_barra * percentuale)
    barra = "[" + "=" * (blocchi_compilati - 1) + ">" + " " * (lunghezza_barra - blocchi_compilati) + "]"
    sys.stdout.write(f"\r{barra} {percentuale * 100:.2f}% complete")
    sys.stdout.flush()


# Function to get possible distractors based on similar words
def get_possible_distractors(targets: set[str], similar_words: dict[str, list[str]]) -> set[str]:
    distractors = set()
    for target in targets:
        distractors = distractors.union(similar_words[target])
    return distractors


# Function to get neighbors in the vocabulary based on embeddings
def get_vocabulary_neighbors(target: str, vocabulary: set[str], embeddings) -> set[str]:
    neighbors = embeddings.get_nearest_neighbors(target, k=NEIGHBORS_POOL)
    similar_words = set()
    for neighbor in neighbors:
        if neighbor in vocabulary:
            similar_words.add(neighbor)
        if len(similar_words) == K:
            break
    return similar_words


# Function to get possible distractors based on edit distance
def get_possible_distractors_by_edit_dist(targets: set[str], vocabulary: set[str]) -> set[str]:
    distractors = set()
    for target in targets:
        heap = []
        heapify(heap)
        for word in vocabulary:
            if word not in distractors and word not in targets:
                distance = editdistance.eval(target, word)
                if len(heap) < K and distance < THRESHOLD:
                    heappush(heap, (-1 * distance, word))
                elif len(heap) >= K and distance < -1 * heap[0][0] and distance < THRESHOLD:
                    heappop(heap)
                    heappush(heap, (-1 * editdistance.eval(target, word), word))
        for word in heap:
            distractors.add(word[1])
    return distractors


# Function to get distractors
def get_distractors(target: str, similar_target: list[str], target_pos: int, possible_distractors: set[str]) -> list[
    str]:
    elements = similar_target
    for distractor in possible_distractors:
        if distractor not in similar_target:
            elements.append(distractor)
    weights = np.array(
        [FAVOURITE_SIMIL_TARGET] * len(similar_target) + [1] * (len(possible_distractors) - len(similar_target)),
        dtype=float)
    weights /= weights.sum()
    distractors = np.random.choice(elements, p=weights, size=3, replace=False).tolist()
    for distractor in distractors:
        possible_distractors.remove(distractor)
    distractors.insert(target_pos, target)
    return distractors


# Function read data from the files
def get_data(hyponym_file: str, hypernym_file: str) -> tuple[dict[str, list[str]], set[str]]:
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


# Function to create JSON entries from data
def create_json_entries_from_data(data: dict[str, list[str]], similar_words: dict[str, list[str]]) -> list[dict]:
    json_sentences = []
    for id, (hyponym, hypernyms) in enumerate(data.items()):
        print_progress_bar(id / len(data))
        json_sentences += create_json_entries(hyponym, hypernyms, similar_words)
    return json_sentences


# Function to create JSON entries from a single hyponym and its hypernyms
def create_json_entries(hyponym: str, hypernyms: list[str], similar_words: dict[str, list[str]]) -> list[dict]:
    targets = set([hyponym] + hypernyms)
    json_entries = []
    possible_distractors = get_possible_distractors(targets, similar_words)
    for hypernym in hypernyms:
        target_pos = rnd.randint(0, 3)
        json_entry = {
            "text": hyponym,
            "choices": get_distractors(hypernym, similar_words[hypernym], target_pos, possible_distractors),
            "label": target_pos
        }
        json_entries.append(json_entry)
    return json_entries


# Function to write JSON entries
def write_json_entries(json_entries: list[dict], output_file: str, set_id=True) -> None:
    with open(output_file, 'w', encoding='utf8') as jsonl_file:
        for i, item in enumerate(json_entries):
            if set_id:
                item["id"] = i
            json.dump(item, jsonl_file)  # Write the JSON object
            jsonl_file.write('\n')


# Function to execute the code for creating the reformed dataset using multiple threads
def spark_execution(data: dict[str, list[str]], vocabulary: set[str], partitions: int,
                    use_fasttext: bool = False) -> list[dict]:

    sc = pk.SparkContext("local[*]")
    vocab = sc.parallelize(vocabulary, partitions)
    mapping = vocab.map(
        lambda x: generate_similar_words_vocab_map(x, vocabulary, use_fasttext=use_fasttext))
    print("generating similar words")
    similar_words = mapping.reduce(
        lambda x, y: {k: x.get(k, []) + y.get(k, []) for k in set(x.keys()).union(set(y.keys()))})
    print("creating json")
    data = sc.parallelize(data.items(), partitions)
    json_entries = data.flatMap(lambda x: create_json_entries(x[0], x[1], similar_words)).collect()
    return json_entries


# Function to execute the code for creating the reformed dataset using a single thread
def single_thread_execution(data: dict[str, list[str]], vocabulary: set[str],
                            use_fasttext: bool = False) -> list[dict]:
    embeddings = None
    if use_fasttext:
        fasttext.util.download_model('it', if_exists='ignore')  # Italian
        embeddings = fasttext.load_model('cc.it.300.bin')
    similar_words = generate_similar_words_vocab(vocabulary, embeddings, use_fasttext=use_fasttext, progress_bar=True)
    json_entries = create_json_entries_from_data(data, similar_words)
    return json_entries


# Main function
def main() -> None:
    if len(sys.argv) != 6:
        print("Usage: python managing_data_task_26.py <input_file1> <input_file2> <output_file> <vocabulary_file> <partitions> [<use_fasttext>]")
        sys.exit(1)
    hyponym_file = sys.argv[1]
    hypernym_file = sys.argv[2]
    output_file = sys.argv[3]
    vocabulary_file = sys.argv[4]
    partitions = int(sys.argv[5])
    use_fasttext = False if len(sys.argv) == 6 else True
    if partitions < 1:
        print("Invalid number of partitions, it must be greater than 0")
        sys.exit(1)
    data, vocab = get_data(hyponym_file, hypernym_file)
    vocabulary = get_vocabulary(vocabulary_file)
    vocabulary = vocabulary.union(vocab)
    json_entries = spark_execution(data, vocabulary, partitions, use_fasttext) if partitions > 1 else \
        single_thread_execution(data, vocabulary, use_fasttext)

    write_json_entries(json_entries, output_file, set_id=True)


if __name__ == '__main__':
    main()
