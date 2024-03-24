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
                sentence.append(tuple(line.split()))
        sentences.append(sentence)
    return sentences


def get_info_sentences(sentences):
    sentences_info = {
        "num_sentences": len(sentences)
    }
    tags = {}
    for sentence in sentences:
        for word in sentence[1:]:
            tags[word[1]] = tags.get(word[1], 0) + 1
    sentences_info["tags"] = tags
    return sentences_info


def extract_most_common_tags(sentences_info):
    list_tags = list(sentences_info["tags"].items())
    list_tags.sort(key=lambda x: x[1], reverse=True)
    return [DECODE_TAGS[tag[0]] for tag in list_tags]


def generate_distractors(word, tag, tag_position, most_common_tags_for_words, most_common_tags):
    choiches = []
    tag = DECODE_TAGS[tag]
    tags_for_word = most_common_tags_for_words[word].copy()
    tags_for_word.remove(tag)
    tags_for_word = list(tags_for_word)
    tmp = [i for i in range(len(tags_for_word))]
    rnd.shuffle(tmp)
    for i in range(min(len(tmp), 3)):
        choice = tags_for_word[tmp[i]]
        choiches.append(choice)
    if len(choiches) < 3:
        most_common_tags.remove(tag)
        for choice in choiches:
            most_common_tags.remove(choice)
        tmp = [i for i in range(MAX_COMMON_TAGS_TO_CONSIDER)]
        rnd.shuffle(tmp)
        l = len(choiches)
        for i in range(3 - l):
            choice = most_common_tags[tmp[i]]
            choiches.append(choice)
    choiches.insert(tag_position, tag)
    return choiches


def find_repeated_words(sentences):
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
                    " ".join([word_[0] if word_[0] != word[0] else "##" + word_[0] for word_ in sentence[1:]]))
                dangerous_sentences_idx.append(sentence_id)
                dangerous_words[word[0]] = dangerous_words.get(word[0], 0) + 1
                break
            else:
                words.add(word[0])
                words_and_tags.add(word)
    return dangerous_sentences, dangerous_sentences_idx, dangerous_words


def find_english_words(sentences):
    suspected_sentences = []
    with open(ENG_VOCABULARY_PATH, 'r') as file:
        eng_words = set([line.strip() for line in file])
    for sentence in sentences:
        for word in sentence[1:]:
            if word[0] in eng_words and word[1] != "PROPN":
                suspected_sentences.append(
                    " ".join([word_[0] if word_[0] != word[0] else "##" + word_[0] for word_ in sentence[1:]]))
                break
    return suspected_sentences



def build_common_tags_for_word(sentences):
    common_tags = {}
    for sentence in sentences:
        for word in sentence[1:]:
            tags_word = common_tags.get(word[0], set())
            tags_word.add(DECODE_TAGS[word[1]])
            common_tags[word[0]] = tags_word
    return common_tags


def create_json(sentences, file_name, sentences_to_remove=set()):
    json_sentences = []
    most_common_tags_for_words = build_common_tags_for_word(sentences)
    most_common_tags = extract_most_common_tags(get_info_sentences(sentences))
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
                                                    most_common_tags.copy()),
                    "label": label
                }
                json_sentences.append(sentence_dict)
    with open(file_name, 'w', encoding='utf8') as jsonl_file:
        for item in json_sentences:
            json.dump(item, jsonl_file)  # Scrivi l'oggetto JSON
            jsonl_file.write('\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python managing_data_task_23.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    sentences = get_list_sentences(input_file)
    suspected_sentences = find_english_words(sentences)
    with open("suspected_sentences.txt", 'w') as file:
        for sentence in suspected_sentences:
            file.write(sentence + "\n")
    # dangerous_sentences, dangerous_sentences_idx, dangerous_words = find_repeated_words(sentences)
    # create_json(sentences, output_file)
    # info = get_info_sentences(sentences)
    # print(extract_most_common_tags(info))
    # pp.pprint(info)

