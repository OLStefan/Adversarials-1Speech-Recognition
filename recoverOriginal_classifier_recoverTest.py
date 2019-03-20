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
files = []
samples_test = []
labels_test = []

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

    if not any(file in s for s in files):
        files.append(file)
        c = [[] for i in range(len(targets))]
        d = [[] for i in range(len(targets))]
        samples_test.append(c)
        labels_test.append(d)

    samples_test[files.index(file)][targets.index(target_2)].append(a)
    labels_test[files.index(file)][targets.index(target_2)].append(targets.index(orig))

f.close()

result = [[[[] for k in range(len(targets))] for j in range(len(files))] for i in range(len(classifiers))]
count = [[[[] for k in range(len(targets))] for j in range(len(files))] for i in range(len(classifiers))]

# Count Samples
cnt_samples = 0
for i in range(len(samples)):
    for j in range(len(samples[i])):
        cnt_samples += 1
print(str(cnt_samples))
print(len(files))

def classify(classifier, samples_train, samples_test, labels_train, labels_test):
    a = [0 for i in range(len(targets))]
    b = 0
    clf = base.clone(classifier)
    clf.fit(samples_train, labels_train)
    pred = clf.predict(samples_test)
    for i in range(len(labels_test)):
        a[pred[i]] += 1
        b += 1
    return a, b


for rep in range(reps):
    print("Rep: " + str(rep))
    for file in files:
        print("\tFile: " + str(files.index(file)))
        for target in targets:
            if len(samples_test[files.index(file)][targets.index(target)]) == 0:
                continue
            for classifier in classifiers:
                a, b = classify(classifier, samples[targets.index(target)], samples_test[files.index(file)][targets.index(target)], labels[targets.index(target)], labels_test[files.index(file)][targets.index(target)])
                result[classifiers.index(classifier)][files.index(file)][targets.index(target)].append(a)
                count[classifiers.index(classifier)][files.index(file)][targets.index(target)].append(b)

f = open("results_recoverOriginal_classifier_recoverTest.txt", "w+")
sum_pred = [[[0 for i in range(len(targets))] for j in range(len(files))] for k in range(len(classifiers))]
for file in files:
    print("File: " + file + ", Original: " + targets[labels_test[files.index(file)][0][0]])
    f.write("File: " + file + ", Original: " + targets[labels_test[files.index(file)][0][0]] + "\n")
    for classifier in range(len(classifiers)):
        print("\tClassifier: " + clfNames[classifier])
        f.write("\tClassifier: " + clfNames[classifier] + "\n")
        for target in range(len(targets)):
            print("\t\tTarget: " + targets[target])
            f.write("\t\tTarget: " + targets[target] + "\n")
            sum_res = [0 for i in range(len(targets))]
            sum_cnt = 0
            for i in range(len(count[classifier][files.index(file)][target])):
                for pred in range(len(targets)):
                    sum_res[pred] += result[classifier][files.index(file)][target][i][pred]
                sum_cnt += count[classifier][files.index(file)][target][i]
            for pred in range(len(targets)):
                sum_pred[classifier][files.index(file)][pred] += sum_res[pred]
                print("\t\t\t" + targets[pred] + ": " + str(round(sum_res[pred] / max(sum_cnt, 1), 4) * 100) + "% (" + str(sum_res[pred]) + "/" + str(sum_cnt) + ")")
                f.write("\t\t\t" + targets[pred] + ": " + str(round(sum_res[pred] / max(sum_cnt, 1), 4) * 100) + "% (" + str(sum_res[pred]) + "/" + str(sum_cnt) + ")\n")
print()
print()
print()
f.write("\n")
f.write("\n")
f.write("\n")
print("Most Frequent Predictions")
f.write("Most Frequent Predictions\n")
correct = [0 for i in range(len(classifiers))]
for file in files:
    orig = targets[labels_test[files.index(file)][0][0]]
    print("File: " + file + ", Original: " + orig)
    f.write("File: " + file + ", Original: " + orig + "\n")
    for classifier in range(len(classifiers)):
        max = 0
        maxIndex = -1
        for pred in range(len(targets)):
            if sum_pred[classifier][files.index(file)][pred] > max:
                max = sum_pred[classifier][files.index(file)][pred]
                maxIndex = pred
        if maxIndex == targets.index(orig):
            correct[classifier] += 1
        print("\tClassifier " + clfNames[classifier] + ": " + targets[maxIndex] + "(" + str(max) + "")
        f.write("\tClassifier " + clfNames[classifier] + ": " + targets[maxIndex] + "(" + str(max) + ")"+"\n")
print()
print()
print()
f.write("\n")
f.write("\n")
f.write("\n")
print("Correct Predictions")
f.write("Correct Predictions\n")
for classifier in range(len(classifiers)):
    print("\tClassifier " + clfNames[classifier] + ": " + str(correct[classifier]) + "/" + str(len(files)))
    f.write("\tClassifier " + clfNames[classifier] + ": " + str(correct[classifier]) + "/" + str(len(files)) + "\n")

f.close()
