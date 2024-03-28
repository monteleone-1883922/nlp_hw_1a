import editdistance
from heapq import heappop, heappush, heapify
import time

VOCABULARY_PATH = "data/SemEval2018-Task9/1B.italian.vocabulary.txt"
K = 5


def get_vocabulary():
    return set([line.strip().lower() for line in open(VOCABULARY_PATH, 'r')])

def get_distance(word1, word2):
    return editdistance.eval(word1, word2)


def get_possible_distractors(targets: set[str], vocabulary: set[str]):
    distractors = set()
    for target in targets:
        heap = []
        heapify(heap)
        for word in vocabulary:
            if word not in distractors and word not in targets:
                if len(heap) < K:
                    heappush(heap,  (-1 * get_distance(target, word), word))
                elif editdistance.eval(target, word) < -1 * heap[0][0]:
                    heappop(heap)
                    heappush(heap,  (-1 * get_distance(target, word), word))
        for word in heap:
            distractors.add(word[1])
    return distractors

def test():
    vocabulary = get_vocabulary()
    targets = ["sesto", "grado", "numero ordinale",	"frazione",	"carica"]
    tmp = time.time()
    distractors = get_possible_distractors(set(targets), vocabulary)
    print("the function has been executed in ",time.time() - tmp, " seconds")
    print(distractors)




if __name__ == "__main__":
    test()



