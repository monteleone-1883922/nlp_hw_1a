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

SENTENCE_START = "_____"

DECODE_TAGS = {
    "ADJ": "Adjective",  # aggettivo
    "ADP": "Adposition",  # preposizione semplice
    "ADP A": "Adposition+Article",  # preposizione articolata
    "ADV": "Adverb",  # avverbio
    "AUX": "Auxiliary Verb",  # verbo ausiliare
    "CONJ": "Coordinating Conjunction",  # congiunzione coordinativa
    "DET": "Determiner",  # determinante
    "INTJ": "Interjection",  # interiezione
    "NOUN": "Noun",  # nome
    "NUM": "Numeral",  # numero
    "PART": "Particle",  # particella
    "PRON": "Pronoun",  # pronome
    "PROPN": "Proper Noun",  # nome proprio
    "PUNCT": "punctuation",  # punteggiatura
    "SCONJ": "Subordinating Conjunction",  # congiunzione subordinativa
    "SYM": "Symbol",  # simbolo
    "EMO": "Emoticon",  # emoticon
    "URL": "Web Address",  # indirizzo web
    "EMAIL": "Email Address",  # indirizzo email
    "HASHTAG": "Hashtag",  # hashtag
    "MENTION": "Mention",  # menzione
    "VERB": "Verb",  # verbo
    "VERB CLIT": "Verb + Clitic pronoun cluster",  # verbo + pronome clitico
    "X": "Other"  # altro
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
                sentence = [(line.strip(SENTENCE_START), "")]  # sentence id
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
    return [tag[0] for tag in list_tags]


def generate_distractors(word, tag, tag_position, most_common_tags_for_words, most_common_tags):
    choiches = []
    tags_for_word = most_common_tags_for_words[word]
    tags_for_word.remove(tag)
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


def build_common_tags_for_word(sentences):
    common_tags = {}
    for sentence in sentences:
        for word in sentence[1:]:
            common_tags[word[0]] = common_tags.get(word[0], []) + [word[1]]
    return common_tags


def create_json(sentences):
    json_sentences = []
    most_common_tags_for_words = build_common_tags_for_word(sentences)
    most_common_tags = extract_most_common_tags(get_info_sentences(sentences))
    for sentence in sentences:
        sentence_id = sentence[0][0]
        sentence_text = " ".join([word[0] for word in sentence[1:]])
        for i, word in enumerate(sentence[1:]):
            label = rnd.randint(0, 3)
            sentence_dict = {
                "sentence_id": sentence_id,
                "sentence": sentence_text,
                "target_word": word[0],
                "word_idx": i,
                "choices": generate_distractors(word[0], word[1], label, most_common_tags_for_words, most_common_tags),
                "label": label
            }
            json_sentences.append(sentence_dict)



if __name__ == '__main__':
    sentences = get_list_sentences("data/postwita/goldDEVset-2016_09_05_anon_rev.txt")
    info = get_info_sentences(sentences)
    print(extract_most_common_tags(info))
    pp.pprint(info)
    # dangerous_sentences, dangerous_sentences_idx, dangerous_words = find_repeated_words(sentences)
    # with open("dangerous_sentences.txt", 'w', encoding='utf8') as file:
    #     for sentence in dangerous_sentences:
    #         file.write(sentence + "\n")
