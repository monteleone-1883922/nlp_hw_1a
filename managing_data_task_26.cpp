#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <string>
#include "nlohmann/json.hpp"
#include <random>

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


std::vector<std::pair<std::string, std::vector<std::string>>> getData(const std::string &hyponymFile, const std::string &hypernymFile) {
    std::vector<std::pair<std::string, std::vector<std::string>>> data;
    std::ifstream hyponymStream(hyponymFile);
    std::ifstream hypernymStream(hypernymFile);
    std::string hyponymLine;
    std::string hypernymLine;
    while (std::getline(hyponymStream, hyponymLine) && std::getline(hypernymStream, hypernymLine)) {
        std::string hyponym = hyponymLine.substr(0, hyponymLine.find('\t'));
        std::vector<std::string> hypernyms;
        std::string hypernym;
        std::istringstream hypernymStream(hypernymLine);
        while (std::getline(hypernymStream, hypernym, '\t')) {
            hypernyms.emplace_back(hypernym);
        }
        data.emplace_back(hyponym, hypernyms);
    }
    return data;
}



std::vector<std::string> getDistractors(const std::string &target, int targetPos, std::set<std::string> &possibleDistractors, std::mt19937 &rnd) {
    std::vector<std::string> distractors;
    std::vector<std::string> possibleDistractorsVector(possibleDistractors.begin(), possibleDistractors.end());
    std::shuffle(possibleDistractorsVector.begin(), possibleDistractorsVector.end(), rnd);
    for (int i = 0; i < 3; i++) {
        distractors.push_back(possibleDistractorsVector[i]);
        possibleDistractors.erase(possibleDistractorsVector[i]);
    }
    distractors.insert(distractors.begin() + targetPos, target);
    return distractors;

}



void createJson(const std::vector<std::pair<std::string, std::vector<std::string>>> &data, const std::string &outputFile, const std::set<std::string> &vocabulary) {
    std::vector<nlohmann::json> jsonEntries;
    int id = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 3);
    for (const std::pair<std::string, std::vector<std::string>> &entry : data) {
        std::set<std::string> targets;
        targets.insert(entry.first);
        for (const std::string &hypernym : entry.second) {
            targets.insert(hypernym);
        }
        std::set<std::string> possibleDistractors = getPossibleDistractors(targets, vocabulary);
        for (const std::string &hypernym : entry.second) {
            int targetPos = distrib(gen);
            nlohmann::json jsonEntry;
            jsonEntry["id"] = id;
            jsonEntry["text"] = entry.first;
            jsonEntry["label"] = targetPos;
            jsonEntry["choices"] = getDistractors(hypernym, targetPos, possibleDistractors, gen);
            jsonEntries.push_back(jsonEntry);
            id++;
        }
    }
    std::ofstream jsonlFile(outputFile);
    for (const nlohmann::json &item : jsonEntries) {
        jsonlFile << item << std::endl;
    }
    jsonlFile.close();
}

int main(int argc, char *argv[]) {

    if (argc < 4) {
        std::cerr << "Usage: ./managing_data_task_26 <input_file1> <input_file2> <output_file>" << std::endl;
        return 1;
    }
    std::string hyponymFile = argv[1];
    std::string hypernymFile = argv[2];
    std::string outputFile = argv[3];
    return 0;
}
