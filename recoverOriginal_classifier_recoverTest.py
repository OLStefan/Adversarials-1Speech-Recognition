import numpy as np
import sklearn.base as base
import sklearn.model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

targets = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
attacks = ["alzantot", "carlini"]
classifiers = [MLPClassifier(hidden_layer_sizes=(4, 8, 16), max_iter=250, learning_rate_init=0.1),
               KNeighborsClassifier(10),
               DecisionTreeClassifier(max_depth=5)]
clfNames = ["Neural Network", "kNN", "Decision Tree"]
reps = 50
f = open("data.txt", "r")
f1 = f.readlines()

samples = [[] for i in range(len(targets))]
labels = [[] for i in range(len(targets))]
samples_test = [[] for i in range(len(targets))]
labels_test = [[] for i in range(len(targets))]
result = [[[] for j in range(len(targets))] for i in range(len(classifiers))]
count = [[[] for j in range(len(targets))] for i in range(len(classifiers))]

for line in f1:
    file, orig, attack_1, target_1, attack_2, target_2, norms = line.split(",", 6)
    if attack_1 != "carlini" or attack_2 != "alzantot":
        continue
    a = []
    for item in norms.split(","):
        a.append(float(item))
    if attack_1 == "-" or target_1 == target_2:
        continue

    samples[targets.index(target_2)].append(a)
    labels[targets.index(target_2)].append(targets.index(orig))

f.close()

f = open("data_recoverTest.txt", "r")
f1 = f.readlines()
for line in f1:
    file, orig, attack_1, target_1, attack_2, target_2, norms = line.split(",", 6)
    a = []
    for item in norms.split(","):
        a.append(float(item))
    if attack_1 == "-" or target_1 == target_2:
        continue

    samples_test[targets.index(target_2)].append(a)
    labels_test[targets.index(target_2)].append(targets.index(orig))

f.close()

# Count Samples
cnt_samples = 0
for i in range(len(samples)):
    for j in range(len(samples[i])):
        cnt_samples += 1
print(str(cnt_samples))

def classify(classifier, samples_train, samples_test, labels_train, labels_test):
    a = [[0 for i in range(len(targets))] for j in range(len(targets))]
    b = [0 for i in range(len(targets))]
    clf = base.clone(classifier)
    clf.fit(samples_train, labels_train)
    pred = clf.predict(samples_test)
    for i in range(len(labels_test)):
        a[labels_test[i]][pred[i]] += 1
        b[labels_test[i]] += 1
    return a, b

for rep in range(reps):
    print("Rep: " + str(rep))
    for target in targets:
        test = samples_test[targets.index(target)]
        if len(test) == 0:
            continue
        for classifier in classifiers:
            a, b = classify(classifier, samples[targets.index(target)], samples_test[targets.index(target)], labels[targets.index(target)], labels_test[targets.index(target)])
            result[classifiers.index(classifier)][targets.index(target)].append(a)
            count[classifiers.index(classifier)][targets.index(target)].append(b)

f = open("results_recoverOriginal_classifier_recoverTest.txt", "w+")
sum_pred = [[[0 for i in range(len(targets))] for j in range(len(targets))] for k in range(len(classifiers))]
for target in targets:
    print("\tTarget: " + target)
    f.write("\tTarget: " + target + "\n")
    for classifier in range(len(classifiers)):
        sum_res = [[0 for i in range(len(targets))] for j in range(len(targets))]
        sum_cnt = [0 for i in range(len(targets))]
        for i in range(len(count[classifier][targets.index(target)])):
            for j in range(len(targets)):
                for k in range(len(targets)):
                    sum_res[j][k] += result[classifier][targets.index(target)][i][j][k]
                sum_cnt[j] += count[classifier][targets.index(target)][i][j]
        print("\tClassifier: " + clfNames[classifier])
        f.write("\tClassifier: " + clfNames[classifier] + "\n")
        for orig in range(len(targets)):
            print("\t\tOriginal: " + targets[orig])
            f.write("\t\tOriginal: " + targets[orig] + "\n")
            for pred in range(len(targets)):
                sum_pred[classifier][orig][pred] += sum_res[orig][pred]
                print("\t\t\t" + targets[pred] + ":" + str(round(sum_res[orig][pred] / max(sum_cnt[orig], 1), 3) * 100) + " (" +str(sum_res[orig][pred]) + "/" + str(sum_cnt[orig]) + ")")
                f.write("\t\t\t" + targets[pred] + ": " + str(round(sum_res[orig][pred] / max(sum_cnt[orig], 1), 3) * 100) + " (" + str(sum_res[orig][pred]) + "/" + str(sum_cnt[orig]) + ")\n")
print()
print()
print()
f.write("\n")
f.write("\n")
f.write("\n")
print("Maximum Prediction:")
for orig in range(len(targets)):
    print("Original: " + targets[orig])
    f.write("Original: " + targets[orig]+"\n")
    for classifier in range(len(classifiers)):
        max = 0
        maxIndex = -1
        for pred in range(len(targets)):
            if sum_pred[classifier][orig][pred] > max:
                max = sum_pred[classifier][orig][pred]
                maxIndex = pred
        print("\tClassifier " + clfNames[classifier] + ": " + targets[maxIndex] + "(" + str(max) + ")")
        f.write("\tClassifier " + clfNames[classifier] + ": " + targets[maxIndex] + "(" + str(max) + ")"+"\n")

f.close()
