#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

const std::string VOCABULARY_PATH = "data/SemEval2018-Task9/1B.italian.vocabulary.txt";
const int K = 5;

std::set<std::string> loadVocabulary(const std::string &path) {
    std::set <std::string> vocabulary;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line[line.size() - 1] == '\n') {
            line.erase(line.size() - 1);
        }
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
        vocabulary.insert(line);
    }
    file.close();
    return vocabulary;
}


int editDistance(std::string s, std::string t) {
    int m = s.size();
    int n = t.size();

    std::vector<int> prev(n + 1, 0), curr(n + 1, 0);

    for (int j = 0; j <= n; j++) {
        prev[j] = j;
    }

    for (int i = 1; i <= m; i++) {
        curr[0] = i;
        for (int j = 1; j <= n; j++) {
            if (s[i - 1] == t[j - 1]) {
                curr[j] = prev[j - 1];
            }
            else {
                int mn
                        = std::min(1 + prev[j], 1 + curr[j - 1]);
                curr[j] = std::min(mn, 1 + prev[j - 1]);
            }
        }
        prev = curr;
    }

    return prev[n];
}

int getDistance(const std::string &word1, const std::string &word2) {
    return editDistance(word1, word2);
}


std::set<std::string> getPossibleDistractors(const std::set<std::string> &targets, const std::set<std::string> &vocabulary) {
    std::set<std::string> distractors;
    for (const std::string &target : targets) {
        std::vector<std::pair<int, std::string>> heap;
        std::make_heap(heap.begin(), heap.end(), [](const std::pair<int, std::string>& a, const std::pair<int, std::string>& b) {
            return a.first > b.first;
        });
        for (const std::string &word : vocabulary) {
            if (distractors.find(word) == distractors.end()  // not found in distractor set
                && targets.find(word) == targets.end()) { // not found in target set
                if (heap.size() < K) {
                    heap.push_back(std::make_pair(-getDistance(target, word), word));
                } else if (getDistance(target, word) < -heap[0].first) {
                    heap.pop_back();
                    heap.push_back(std::make_pair(-getDistance(target, word), word));
                }
            }
        }
        for (const std::pair<int, std::string> &word : heap) {
            distractors.insert(word.second);
        }
    }
    return distractors;
}


int main(int argc, char *argv[]) {
    return 0;
}
