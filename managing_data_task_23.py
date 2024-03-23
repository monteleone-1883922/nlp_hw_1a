
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
    "ADJ": "Adjective", # aggettivo
    "ADP": "Adposition", # preposizione semplice
    "ADP A": "Adposition+Article", # preposizione articolata
    "ADV": "Adverb", # avverbio
    "AUX": "Auxiliary Verb", # verbo ausiliare
    "CONJ": "Coordinating Conjunction", # congiunzione coordinativa
    "DET": "Determiner", # determinante
    "INTJ": "Interjection", # interiezione
    "NOUN": "Noun", # nome
    "NUM": "Numeral", # numero
    "PART": "Particle", # particella
    "PRON": "Pronoun", # pronome
    "PROPN": "Proper Noun", # nome proprio
    "PUNCT": "punctuation", # punteggiatura
    "SCONJ": "Subordinating Conjunction", # congiunzione subordinativa
    "SYM": "Symbol", # simbolo
    "EMO": "Emoticon", # emoticon
    "URL": "Web Address", # indirizzo web
    "EMAIL": "Email Address", # indirizzo email
    "HASHTAG": "Hashtag", # hashtag
    "MENTION": "Mention", # menzione
    "VERB": "Verb", # verbo
    "VERB CLIT": "Verb + Clitic pronoun cluster", # verbo + pronome clitico
    "X": "Other" # altro
}

def get_list_sentences(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        sentences = []
        first_line = True
        for line in file:
            if line.startswith(SENTENCE_START):
                if not first_line:
                    sentences.append(sentence)
                first_line = False
                sentence = [(line.strip(SENTENCE_START),"")] # sentence id
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

def generate_distractors(tag,tag_position):
    distractors = []
    for i in range(4):
        if i == tag_position:
            distractors.append(DECODE_TAGS[tag])
        else:
            pass

def find_repeated_words(sentences):
    dangerous_sentences = []
    dangerous_sentences_idx = []
    for sentence in sentences:
        sentence_id = sentence[0][0]
        words = set()
        words_and_tags = set()
        for word in sentence[1:]:
            if word[0] in words and word not in words_and_tags:
                dangerous_sentences.append(" ".join([word_[0] if word_[0] != word[0] else "##" + word_[0] for word_ in sentence[1:]]))
                dangerous_sentences_idx.append(sentence_id)
                break
            else:
                words.add(word[0])
                words_and_tags.add(word)
    return dangerous_sentences, dangerous_sentences_idx



def create_json(sentences):
    json_sentences = []
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
                "choices": generate_distractors(word[1], label),
                "label": label
            }
            json_sentences.append(sentence_dict)



if __name__ == '__main__':
    sentences = get_list_sentences("data/postwita/goldDEVset-2016_09_05_anon_rev.txt")
    # pp.pprint(get_info_sentences(sentences))
    dangerous_sentences, dangerous_sentences_idx = find_repeated_words(sentences)
    with open("dangerous_sentences.txt", 'w', encoding='utf8') as file:
        for sentence in dangerous_sentences:
            file.write(sentence + "\n")







