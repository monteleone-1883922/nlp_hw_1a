"""
- struttura json finale:  { "sentence_id": ..., "sentence": ..., "target_word":
    ..., "word_idx": ..., "choices": [...], "label": ... }
- eliminare parole duplicate in stessa frase
- dividere parole della frase in diverse entries del json
- eliminare parole straniere
"""
import json
import pprint as pp
import random as rnd
import sys
import numpy as np
import fasttext.util

SENTENCE_START = "_____"
ENG_VOCABULARY_PATH = "data/eng_vocabulary.txt"

DECODE_TAGS = {
    "ADJ": "aggettivo",
    "ADP": "preposizione semplice",
    "ADP_A": "preposizione articolata",
    "ADV": "avverbio",
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


def get_list_sentences(file_path):
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


def get_info_sentences(sentences):
    sentences_info = {
        "num_sentences": len(sentences)
    }
    tags = {}
    vocab = set()
    for sentence in sentences:
        for word in sentence[1:]:
            vocab.add(word)
            tags[word[1]] = tags.get(word[1], 0) + 1
    sentences_info["tags"] = tags
    sentences_info["vocab"] = vocab
    return sentences_info


def extract_most_common_tags(sentences_info):
    list_tags = list(sentences_info["tags"].items())
    list_tags.sort(key=lambda x: x[1], reverse=True)
    return [DECODE_TAGS[tag[0]] for tag in list_tags]


def get_similar_words(word, embeddings):
    return [word[1] for word in embeddings.get_nearest_neighbors(word, k=NEAR_WORDS_TO_CONSIDER)]


def sample_tags(tags_for_word, tag, choices=[]):
    tags_for_word.pop(tag)
    for distractor in choices:
        tags_for_word.pop(distractor)
    if len(tags_for_word) != 0:
        tags_for_word = list(tags_for_word.items())
        elements, weights = zip(*tags_for_word)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        return np.random.choice(elements, p=weights, size=min(len(elements), 3 - len(choices)), replace=False).tolist()
    else:
        return []


def generate_distractors(word, tag, tag_position, most_common_tags_for_words, most_common_tags, embeddings):
    tag = DECODE_TAGS[tag]
    tags_for_word = most_common_tags_for_words[word].copy()
    choices = sample_tags(tags_for_word, tag)
    if len(choices) < 3:
        # find similar words
        similar_words = get_similar_words(word, embeddings)
        i = 0
        len_choices = len(choices)
        while len_choices < 3 and i < len(similar_words):
            word = similar_words[i]
            tags_for_word = most_common_tags_for_words[word].copy()
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


def find_repeated_words(sentences, write_output=False):
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


def find_english_words(sentences, write_output=False):
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


def build_common_tags_for_word(sentences):
    common_tags = {}
    for sentence in sentences:
        for word in sentence[1:]:
            tags_word = common_tags.get(word[0].lower(), {})
            tags_word[DECODE_TAGS[word[1]]] = tags_word.get(DECODE_TAGS[word[1]], 0) + 1
            common_tags[word[0].lower()] = tags_word
    return common_tags


def create_json(sentences, file_name, sentences_info, embeddings, sentences_to_remove=set()):
    json_sentences = []
    most_common_tags_for_words = build_common_tags_for_word(sentences)
    most_common_tags = extract_most_common_tags(sentences_info)
    for sentence in sentences:
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
                                                    most_common_tags.copy(), embeddings),
                    "label": label
                }
                json_sentences.append(sentence_dict)
    with open(file_name, 'w', encoding='utf8') as jsonl_file:
        for item in json_sentences:
            json.dump(item, jsonl_file)  # Scrivi l'oggetto JSON
            jsonl_file.write('\n')


def remove_unnedded_embeddings(embeddings, words_to_remove):
    new_embeddings = {}
    for word in embeddings.words:
        if word not in words_to_remove:
            new_embeddings[word] = embeddings[word]
    return new_embeddings


def main():
    if len(sys.argv) != 3:
        print("Usage: python managing_data_task_23.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    fasttext.util.download_model('it', if_exists='ignore')  # Italian
    embeddings = fasttext.load_model('cc.it.300.bin')
    sentences = get_list_sentences(input_file)
    sentences_info = get_info_sentences(sentences)
    embeddings = remove_unnedded_embeddings(embeddings, sentences_info["vocab"])
    dangerous_sentences, dangerous_sentences_idx, dangerous_words = find_repeated_words(sentences)
    create_json(sentences, output_file, sentences_info, embeddings, sentences_to_remove=dangerous_sentences_idx)


if __name__ == '__main__':
    main()
