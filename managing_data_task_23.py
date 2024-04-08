import json
import random as rnd
import sys
import time
import editdistance
import numpy as np
import fasttext.util
from heapq import heappop, heappush, heapify
from sklearn.metrics.pairwise import cosine_similarity
import pyspark as pk

# Definition of constants
SENTENCE_START = "_____"
ENG_VOCABULARY_PATH = "data/eng_vocabulary.txt"
# Mapping of from categorical to natural language labels
DECODE_TAGS = {
    "ADJ": "aggettivo",
    "AD": "aggettivo",
    "ADP": "preposizione semplice",
    "ADP_A": "preposizione articolata",
    "ADV": "avverbio",
    "ADV2": "avverbio",
    "AUX": "verbo ausiliare",
    "CONJ": "congiunzione coordinativa",
    "DET": "determinante",
    "INTJ": "interiezione",
    "NOUN": "sostantivo",
    "NUM": "numero",
    "PART": "particella",
    "PRON": "pronome",
    "PROPN": "nome proprio",
    "PUNCT": "punteggiatura",
    "SCONJ": "congiunzione subordinativa",
    "SYM": "simbolo",
    "EMO": "emoticon",
    "URL": "indirizzo web",
    "EMAIL": "indirizzo email",
    "HASHTAG": "hashtag",
    "MENTION": "menzione",
    "VERB": "verbo",
    "VERB_CLIT": "verbo + pronome clitico",
    "X": "altro"
}
MAX_COMMON_TAGS_TO_CONSIDER = 7
NEAR_WORDS_TO_CONSIDER = 3
NEIGHBORS_POOL = 10
SWAP_PROB = 0.4
THRESHOLD = 7


# This function reads the input file and returns a list of sentences.
# Each sentence is a list of tuples, where each tuple contains a word and its tag.
def get_list_sentences(file_path: str) -> list[list[tuple[str, str]]]:
    with open(file_path, 'r', encoding='utf8') as file:
        sentences = []
        first_line = True
        for line in file:
            if line.startswith(SENTENCE_START):
                if not first_line:
                    sentences.append(sentence)
                first_line = False
                sentence = [(line.strip().strip(SENTENCE_START), "")]  # sentence id
            elif line.strip() != "":
                tmp = line.split()
                pair = (tmp[0], tmp[1])
                sentence.append(pair)
        sentences.append(sentence)
    return sentences


# Function to generate similar words vocabulary in single thread execution
def generate_similar_words_vocab(vocabulary: set[str], tags_for_word: dict[str, dict[str, int]], embeddings=None,
                                 use_fasttext: bool = False, use_embeddings: bool = False,
                                 progress_bar=False) -> dict[str, list[str]]:
    similar_words = {}
    for i, word in enumerate(vocabulary):
        if progress_bar:
            print_progress_bar(i / len(vocabulary))
        update = generate_similar_words_vocab_map(word, vocabulary, tags_for_word, embeddings, use_fasttext, False,
                                                  use_embeddings)
        similar_words.update(update)
    return similar_words


# Function to generate similar words vocabulary, used in spark execution and as block for the single thread function
def generate_similar_words_vocab_map(word: str, vocabulary: set[str], tags_for_word: dict[str, dict[str, int]],
                                     embeddings=None, use_fasttext: bool = False, load_embeddings: bool = True,
                                     use_embeddings: bool = False) -> dict[str, list[str]]:
    if use_fasttext and load_embeddings:
        embeddings = fasttext.load_model('cc.it.300.bin')
    similar_words = {}
    if len(tags_for_word[word.lower()]) < 4:
        similar_words[word] = get_similar_words(word, embeddings,
                                                vocabulary) if use_fasttext else find_nearest_neighbors(word,
                                                                                                        embeddings,
                                                                                                        use_embeddings)
    return similar_words


# Function to get information about sentences, tags and vocabulary of words in sentences
def get_info_sentences(sentences: list[list[tuple[str, str]]]) -> dict:
    sentences_info = {
        "num_sentences": len(sentences)
    }
    tags = {}
    vocab = set()
    for sentence in sentences:
        for word in sentence[1:]:
            vocab.add(word[0])
            tags[word[1]] = tags.get(word[1], 0) + 1
    sentences_info["tags"] = tags
    sentences_info["vocab"] = vocab
    return sentences_info


# Function to extract most common tags
def extract_most_common_tags(sentences_info: dict) -> list[str]:
    list_tags = list(sentences_info["tags"].items())
    list_tags.sort(key=lambda x: x[1], reverse=True)
    return [DECODE_TAGS[tag[0]] for tag in list_tags]


# Function to find nearest neighbors of a target word using cosine similarity or edit distance
def find_nearest_neighbors(target: str, embeddings: dict = None, use_embeddings: bool = False) -> list[str]:
    neighbors = []
    heap = []
    heapify(heap)
    for word in embeddings.keys():
        similarity = cosine_similarity(embeddings[target],
                                       embeddings[word]) if use_embeddings else -1 * editdistance.eval(target, word)
        if len(heap) < NEAR_WORDS_TO_CONSIDER and (use_embeddings or similarity < THRESHOLD):
            heappush(heap, (similarity, word))
        elif similarity < heap[0][0] and similarity < THRESHOLD or (use_embeddings and similarity > heap[0][0]):
            heappop(heap)
            heappush(heap, (similarity, word))
    for word in heap:
        neighbors.append(word[1])
    return neighbors


# Function to get similar words to a target word using fasttext embeddings
def get_similar_words(word: str, embeddings, vocabulary: set[str]) -> list[str]:
    neighbors = embeddings.get_nearest_neighbors(word, k=NEIGHBORS_POOL)
    similar_words = []
    for neighbor in neighbors:
        if neighbor in vocabulary:
            similar_words.append(neighbor)
        if len(similar_words) == NEAR_WORDS_TO_CONSIDER:
            break
    return similar_words


# Function to sample tags depending on their frequency
def sample_tags(tags_for_word: dict[str, int], tag: str, choices: list[str] = []) -> list[str]:
    tags_for_word.pop(tag, None)
    for distractor in choices:
        tags_for_word.pop(distractor, None)
    if len(tags_for_word) != 0:
        tags_for_word = list(tags_for_word.items())
        elements, weights = zip(*tags_for_word)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        return np.random.choice(elements, p=weights, size=min(len(elements), 3 - len(choices)), replace=False).tolist()
    else:
        return []


# Function to generate distractors for a target word
def generate_distractors(word: str, tag: str, tag_position: int, most_common_tags_for_words: dict[str, dict[str, int]],
                         most_common_tags: list[str], similar_words: dict[str, list[str]]) -> list[str]:
    tag = DECODE_TAGS[tag]
    tags_for_word = most_common_tags_for_words[word.lower()].copy()
    choices = sample_tags(tags_for_word, tag)
    if len(tags_for_word) > 3 and rnd.random() < SWAP_PROB / (len(tags_for_word) - 3):
        most_common_tags.remove(tag)
        for choice in choices:
            most_common_tags.remove(choice)
        swap = rnd.sample(most_common_tags[:MAX_COMMON_TAGS_TO_CONSIDER], 1)[0]
        choices[rnd.randint(0, 2)] = swap
    if len(choices) < 3:
        # find similar words
        similar_words = similar_words[
            word]  # find_nearest_neighbors(word, embeddings)  # get_similar_words(word, embeddings, vocabulary)
        i = 0
        len_choices = len(choices)
        while len_choices < 3 and i < len(similar_words):
            word = similar_words[i]
            tags_for_word = most_common_tags_for_words[word.lower()].copy()
            distractors = sample_tags(tags_for_word, tag, choices)
            choices += distractors
            len_choices = len(choices)
            i += 1

        most_common_tags.remove(tag)
        for choice in choices:
            most_common_tags.remove(choice)
        distractors = rnd.sample(most_common_tags[:MAX_COMMON_TAGS_TO_CONSIDER], 3 - len(choices))
        choices += distractors
    choices.insert(tag_position, tag)
    return choices


# Function to find repeated words in sentences and write them out if needed
def find_repeated_words(sentences: list[list[tuple[str, str]]], write_output: bool = False) -> tuple[
    list[str], set[str], dict[str, int]]:
    dangerous_sentences = []
    dangerous_sentences_idx = []
    dangerous_words = {}
    for sentence in sentences:
        sentence_id = sentence[0][0]
        words = set()
        words_and_tags = set()
        for word in sentence[1:]:
            if word[0] in words and word not in words_and_tags:
                dangerous_sentences.append(
                    sentence_id + " | " + " ".join(
                        [word_[0] if word_[0] != word[0] else "##" + word_[0] for word_ in sentence[1:]]))
                dangerous_sentences_idx.append(sentence_id)
                dangerous_words[word[0]] = dangerous_words.get(word[0], 0) + 1
                break
            else:
                words.add(word[0])
                words_and_tags.add(word)
    if write_output:
        with open("data/postwita/dangerous_sentences.txt", 'w', encoding='utf8') as file:
            for sentence in dangerous_sentences:
                file.write(sentence + "\n")
    return dangerous_sentences, set(dangerous_sentences_idx), dangerous_words


# Function to find repeated words in a spark friendly version
def find_repeated_words_map(sentence: list[list[tuple[str, str]]]) -> set[tuple[str, str]]:
    dangerous_sentences_idx = []
    sentence_id = sentence[0][0]
    words = set()
    words_and_tags = set()
    for word in sentence[1:]:
        if word[0] in words and word not in words_and_tags:
            dangerous_sentences_idx.append(sentence_id)
            break
        else:
            words.add(word[0])
            words_and_tags.add(word)
    return set(dangerous_sentences_idx)


# Function to find English words in sentences and write them out if needed
def find_english_words(sentences: list[list[tuple[str, str]]], write_output: bool = False) -> tuple[
    list[str], set[str], set[str]]:
    suspected_sentences = []
    suspected_sentences_idx = []
    words_found = set()
    with open(ENG_VOCABULARY_PATH, 'r') as file:
        eng_words = set([line.strip().lower() for line in file])
    for sentence in sentences:
        for word in sentence[1:]:
            if word[0].lower() in eng_words and word[1] != "PROPN":
                words_found.add(word[0].lower())
                suspected_sentences.append(
                    sentence[0][0] + " | " + " ".join(
                        [word_[0] if word_[0] != word[0] else "##" + word_[0] for word_ in sentence[1:]]))
                suspected_sentences_idx.append(sentence[0][0])
                break
    if write_output:
        with open("data/postwita/suspected_eng_sentences.txt", 'w', encoding='utf8') as file:
            for sentence in suspected_sentences:
                file.write(sentence + "\n")
    return suspected_sentences, set(suspected_sentences_idx), words_found


# Function to builda dictionary containing all tags used for each word
def build_common_tags_for_word(sentences: list[list[tuple[str, str]]]) -> dict[str, dict[str, int]]:
    common_tags = {}
    for sentence in sentences:
        for word in sentence[1:]:
            tags_word = common_tags.get(word[0].lower(), {})
            tags_word[DECODE_TAGS[word[1]]] = tags_word.get(DECODE_TAGS[word[1]], 0) + 1
            common_tags[word[0].lower()] = tags_word
    return common_tags


# Function to print a progress bar
def print_progress_bar(percentuale: float, lunghezza_barra: int = 30) -> None:
    blocchi_compilati = int(lunghezza_barra * percentuale)
    barra = "[" + "=" * (blocchi_compilati - 1) + ">" + " " * (lunghezza_barra - blocchi_compilati) + "]"
    sys.stdout.write(f"\r{barra} {percentuale * 100:.2f}% completo")
    sys.stdout.flush()


# Function to create JSON entries from sentences
def create_json_entries_from_sentences(sentences: list[list[tuple[str, str]]], similar_words: dict[str, list[str]],
                                       sentences_to_remove=set(),
                                       most_common_tags_for_words: dict[str, dict[str, int]] = None,
                                       most_common_tags: list[str] = None, sentences_info: dict = None) -> list[dict]:
    json_sentences = []
    if most_common_tags_for_words is None:
        most_common_tags_for_words = build_common_tags_for_word(sentences)
    if most_common_tags is None:
        most_common_tags = extract_most_common_tags(sentences_info)
    for j, sentence in enumerate(sentences):
        print_progress_bar(j / len(sentences))
        json_sentences += create_json_entries(sentence, similar_words, most_common_tags_for_words, most_common_tags,
                                              sentences_to_remove)
    return json_sentences


# Function to create JSON entries from a single sentence
def create_json_entries(sentence: list[tuple[str, str]], similar_words: dict[str, list[str]],
                        most_common_tags_for_words: dict[str, dict[str, int]], most_common_tags: list[str],
                        sentences_to_remove: set[str] = set()) -> list[dict]:
    json_sentences = []
    sentence_id = sentence[0][0]
    if sentence_id not in sentences_to_remove:
        sentence_text = " ".join([word[0] for word in sentence[1:]])
        for i, word in enumerate(sentence[1:]):
            label = rnd.randint(0, 3)
            sentence_dict = {
                "sentence_id": sentence_id,
                "sentence": sentence_text,
                "target_word": word[0],
                "word_idx": i,
                "choices": generate_distractors(word[0], word[1], label, most_common_tags_for_words,
                                                most_common_tags.copy(), similar_words),
                "label": label
            }
            json_sentences.append(sentence_dict)
    return json_sentences


# Function to write JSON entries to a file
def write_json(json_sentences: list[dict], file_name: str) -> None:
    with open(file_name, 'w', encoding='utf8') as jsonl_file:
        for item in json_sentences:
            json.dump(item, jsonl_file)  # Scrivi l'oggetto JSON
            jsonl_file.write('\n')


# Function to remove sentences from a list
def remove_sentences(sentences: list[list[tuple[str, str]]], sentences_to_remove: set[str]) -> list[
    list[tuple[str, str]]]:
    new_sentences = []
    for sentence in sentences:
        sentence_id = sentence[0][0]
        if sentence_id not in sentences_to_remove:
            new_sentences.append(sentence)
    return new_sentences


# Function to build embedding vocabulary
def build_embedding_vocab(embeddings, vocabulary: set[str]) -> dict[str, np.ndarray]:
    embedding_vocab = {}
    for word in vocabulary:
        embedding_vocab[word] = embeddings.get_word_vector(word).reshape(1, -1)
    return embedding_vocab


# Function to get common tags for a single word
def common_tags_for_word(sentence: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    common_tags = {}
    for word in sentence:
        tags_word = common_tags.get(word[0].lower(), {})
        tags_word[DECODE_TAGS[word[1]]] = tags_word.get(DECODE_TAGS[word[1]], 0) + 1
        common_tags[word[0].lower()] = tags_word
    return common_tags


# Function to merge dictionaries used in spark reduce function
def merge_dicts(x: dict, y: dict) -> dict:
    z = {}
    for key in set(x.keys()).union(set(y.keys())):
        tags_x = x.get(key, {})
        tags_y = y.get(key, {})
        tags = {k: tags_x.get(k, 0) + tags_y.get(k, 0) for k in set(tags_x.keys()).union(set(tags_y.keys()))}
        z[key] = tags
    return z


# Function to execute the code for creating the reformed dataset using multiple threads
def spark_execution(sentences_list: list[list[tuple[str, str]]], partitions: int, use_embeddings: bool,
                    use_fasttext: bool = False) -> list[dict]:
    sc = pk.SparkContext("local[*]")
    sentences = sc.parallelize(sentences_list, partitions)
    mapping = sentences.map(lambda x: find_repeated_words_map(x))
    dangerous_sentences_idx = mapping.reduce(lambda x, y: x.union(y))
    sentences = sentences.filter(lambda x: x[0][0] not in dangerous_sentences_idx)
    sentences_list = sentences.collect()
    sentences_info = get_info_sentences(sentences_list)
    embeddings = None
    if use_embeddings or use_fasttext:
        fasttext.util.download_model('it', if_exists='ignore')
        embeddings = fasttext.load_model('cc.it.300.bin')
        if use_embeddings:
            embeddings = build_embedding_vocab(embeddings, sentences_info["vocab"])
    mapping = sentences.map(lambda x: common_tags_for_word(x[1:]))
    most_common_tags_for_word = mapping.reduce(lambda x, y: merge_dicts(x, y))
    most_common_tags = extract_most_common_tags(sentences_info)
    vocab = sc.parallelize(sentences_info["vocab"], partitions)
    mapping = vocab.map(
        lambda x: generate_similar_words_vocab_map(x, sentences_info["vocab"], most_common_tags_for_word, embeddings,
                                                   use_embeddings=use_embeddings, use_fasttext=use_fasttext))
    print("generating similar words")
    similar_words = mapping.reduce(
        lambda x, y: {k: x.get(k, []) + y.get(k, []) for k in set(x.keys()).union(set(y.keys()))})
    mapping = sentences.map(
        lambda x: create_json_entries(x, similar_words, most_common_tags_for_word, most_common_tags))
    print("generating json entries")
    json_sentences = mapping.reduce(lambda x, y: x + y)
    return json_sentences


# Function to execute the code for creating the reformed dataset using a single thread
def single_thread_execution(sentences: list[list[tuple[str, str]]], use_embeddings: bool = False,
                            use_fasttext: bool = False) -> list[dict]:
    dangerous_sentences, dangerous_sentences_idx, dangerous_words = find_repeated_words(sentences)
    sentences = remove_sentences(sentences, dangerous_sentences_idx)
    sentences_info = get_info_sentences(sentences)
    embeddings = None
    if use_embeddings or use_fasttext:
        fasttext.util.download_model('it', if_exists='ignore')
        embeddings = fasttext.load_model('cc.it.300.bin')
        if use_embeddings:
            embeddings = build_embedding_vocab(embeddings, sentences_info["vocab"])
    most_common_tags_for_word = build_common_tags_for_word(sentences)
    most_common_tags = extract_most_common_tags(sentences_info)
    similar_words = generate_similar_words_vocab(sentences_info["vocab"], most_common_tags_for_word, embeddings,
                                                 use_fasttext=use_fasttext, use_embeddings=use_embeddings,
                                                 progress_bar=True)
    json_entries = create_json_entries_from_sentences(sentences, similar_words,
                                                      most_common_tags_for_words=most_common_tags_for_word,
                                                      most_common_tags=most_common_tags)
    return json_entries


# Main function
def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python managing_data_task_23.py <input_file> <output_file> <partitions> [<similarity_method>]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    partitions = int(sys.argv[3])
    if partitions < 1:
        print("Invalid number of partitions, it must be greater than 0")
        sys.exit(1)
    similarity_method = "editdistance" if len(sys.argv) == 4 else sys.argv[4]
    sentences_list = get_list_sentences(input_file)
    use_embeddings = similarity_method == "cosine_similarity"
    use_fasttext = similarity_method == "fasttext"
    if not use_embeddings and not use_fasttext and similarity_method != "editdistance":
        print("Invalid similarity method, valid values are: 'editdistance', 'cosine_similarity' and 'fasttext'")
        sys.exit(1)
    if partitions == 1:
        json_sentences = single_thread_execution(sentences_list, use_embeddings=use_embeddings,
                                                 use_fasttext=use_fasttext)
    else:
        json_sentences = spark_execution(sentences_list, partitions, use_embeddings=use_embeddings,
                                         use_fasttext=use_fasttext)

    write_json(json_sentences, output_file)


if __name__ == '__main__':
    main()
