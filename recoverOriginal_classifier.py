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

samples = [[[[] for i in range(len(targets))] for j in range(len(attacks))] for k in range(len(attacks))]
labels = [[[[] for i in range(len(targets))] for j in range(len(attacks))] for k in range(len(attacks))]

cnt_difAtk_difTgt = [[[[[] for i in range(len(targets))] for j in range(len(attacks))] for k in range(len(attacks))] for
                     l in range(len(classifiers))]
cnt_difAtk = [[[[] for j in range(len(attacks))] for k in range(len(attacks))] for l in range(len(classifiers))]
cnt_difTgt = [[[] for i in range(len(targets))] for l in range(len(classifiers))]
cnt_all = [[] for l in range(len(classifiers))]

res_difAtk_difTgt = [[[[[] for i in range(len(targets))] for j in range(len(attacks))] for k in range(len(attacks))] for
                     l in range(len(classifiers))]
res_difAtk = [[[[] for j in range(len(attacks))] for k in range(len(attacks))] for l in range(len(classifiers))]
res_difTgt = [[[] for i in range(len(targets))] for l in range(len(classifiers))]
res_all = [[] for l in range(len(classifiers))]

for line in f1:
    file, orig, attack_1, target_1, attack_2, target_2, norms = line.split(",", 6)
    a = []
    for item in norms.split(","):
        a.append(float(item))
    if attack_1 == "-" or target_1 == target_2:
        continue

    samples[attacks.index(attack_1)][attacks.index(attack_2)][targets.index(target_2)].append(a)
    labels[attacks.index(attack_1)][attacks.index(attack_2)][targets.index(target_2)].append(targets.index(orig))

f.close()

# Count Samples
cnt_samples = 0
for i in range(len(samples)):
    for j in range(len(samples[i])):
        for k in range(len(samples[i][j])):
            for l in range(len(samples[i][j][k])):
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
    for attack_1 in attacks:
        for attack_2 in attacks:
            for target in targets:
                print(
                    "Rep: " + str(rep + 1) + ", Attack 1: " + attack_1 + ", Attack 2: " + attack_2 + ", Target: " + str(
                        target))
                samples_part = samples[attacks.index(attack_1)][attacks.index(attack_2)][targets.index(target)]
                labels_part = labels[attacks.index(attack_1)][attacks.index(attack_2)][targets.index(target)]
                samples_train, samples_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
                    samples_part, labels_part, test_size=0.25)
                for classifier in classifiers:
                    a, b = classify(classifier, samples_train, samples_test, labels_train, labels_test)
                    res_difAtk_difTgt[classifiers.index(classifier)][attacks.index(attack_1)][attacks.index(attack_2)][
                        targets.index(target)].append(a)
                    cnt_difAtk_difTgt[classifiers.index(classifier)][attacks.index(attack_1)][attacks.index(attack_2)][
                        targets.index(target)].append(b)
            print("Rep: " + str(rep + 1) + ", Attack 1: " + attack_1 + ", Attack 2: " + attack_2 + ", all Targets")
            samples_part = []
            labels_part = []
            for i in range(len(samples[attacks.index(attack_1)][attacks.index(attack_2)])):
                for j in range(len(samples[attacks.index(attack_1)][attacks.index(attack_2)][i])):
                    samples_part.append(samples[attacks.index(attack_1)][attacks.index(attack_2)][i][j])
                    labels_part.append(labels[attacks.index(attack_1)][attacks.index(attack_2)][i][j])
            samples_train, samples_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
                samples_part, labels_part, test_size=0.25)
            for classifier in classifiers:
                a, b = classify(classifier, samples_train, samples_test, labels_train, labels_test)
                res_difAtk[classifiers.index(classifier)][attacks.index(attack_1)][attacks.index(attack_2)].append(a)
                cnt_difAtk[classifiers.index(classifier)][attacks.index(attack_1)][attacks.index(attack_2)].append(b)
    for target in targets:
        print("Rep: " + str(rep + 1) + ", all Attacks, Target: " + target)
        samples_part = []
        labels_part = []
        for i in range(len(samples)):
            for j in range(len(samples[i])):
                for k in range(len(samples[i][j][targets.index(target)])):
                    samples_part.append(samples[i][j][targets.index(target)][k])
                    labels_part.append(labels[i][j][targets.index(target)][k])
        samples_train, samples_test, labels_train, labels_test = sklearn.model_selection.train_test_split(samples_part,
                                                                                                          labels_part,
                                                                                                          test_size=0.25)
        for classifier in classifiers:
            a, b = classify(classifier, samples_train, samples_test, labels_train, labels_test)
            res_difTgt[classifiers.index(classifier)][targets.index(target)].append(a)
            cnt_difTgt[classifiers.index(classifier)][targets.index(target)].append(b)
    print("Rep: " + str(rep + 1) + ", all Attacks, all Targets")
    samples_part = []
    labels_part = []
    for i in range(len(samples)):
        for j in range(len(samples[i])):
            for k in range(len(samples[i][j])):
                for l in range(len(samples[i][j][k])):
                    samples_part.append(samples[i][j][k][l])
                    labels_part.append(labels[i][j][k][l])
    samples_train, samples_test, labels_train, labels_test = sklearn.model_selection.train_test_split(samples_part,
                                                                                                      labels_part,
                                                                                                      test_size=0.25)
    for classifier in classifiers:
        a, b = classify(classifier, samples_train, samples_test, labels_train, labels_test)
        res_all[classifiers.index(classifier)].append(a)
        cnt_all[classifiers.index(classifier)].append(b)

f = open("results_recoverOriginal_classifier.txt", "w+")

# Ausgabe
for attack_1 in attacks:
    for attack_2 in attacks:
        print("Attack 1: " + attack_1 + ", Attack 2: " + attack_2)
        f.write("Attack 1: " + attack_1 + ", Attack 2: " + attack_2 + "\n")
        for target in targets:
            print("\tTarget: " + target)
            f.write("\tTarget: " + target + "\n")
            for classifier in range(len(classifiers)):
                sum_res = [[0 for i in range(len(targets))] for j in range(len(targets))]
                sum_cnt = [0 for i in range(len(targets))]
                for i in range(len(cnt_difAtk_difTgt[classifier][attacks.index(attack_1)][attacks.index(attack_2)][
                                       targets.index(target)])):
                    for j in range(len(targets)):
                        for k in range(len(targets)):
                            sum_res[j][k] += res_difAtk_difTgt[classifier][attacks.index(attack_1)][attacks.index(attack_2)][
                                targets.index(target)][i][j][k]
                        sum_cnt[j] += cnt_difAtk_difTgt[classifier][attacks.index(attack_1)][attacks.index(attack_2)][
                            targets.index(target)][i][j]
                print("\t\tClassifier: " + clfNames[classifier])
                f.write("\t\tClassifier: " + clfNames[classifier] + "\n")
                for orig in range(len(targets)):
                    print("\t\t\tOriginal: " + targets[orig])
                    f.write("\t\t\tOriginal: " + targets[orig] + "\n")
                    for pred in range(len(targets)):
                         print("\t\t\t\t" + targets[pred] + ":" + str(round(sum_res[orig][pred] / sum_cnt[orig], 3) * 100) + " (" +str(sum_res[orig][pred]) + "/" + str(sum_cnt[orig]) + ")")
                         f.write("\t\t\t\t" + targets[pred] + ": " + str(round(sum_res[orig][pred] / sum_cnt[orig], 3) * 100) + " (" + str(sum_res[orig][pred]) + "/" + str(sum_cnt[orig]) + ")\n")

for attack_1 in attacks:
    for attack_2 in attacks:
        print("Attack 1: " + attack_1 + ", Attack 2: " + attack_2)
        print("\tAll Targets")
        f.write("Attack 1: " + attack_1 + ", Attack 2: " + attack_2 + "\n")
        f.write("\tAll Targets\n")
        for classifier in range(len(classifiers)):
            sum_res = [[0 for i in range(len(targets))] for j in range(len(targets))]
            sum_cnt = [0 for i in range(len(targets))]
            for i in range(len(cnt_difAtk[classifier][attacks.index(attack_1)][attacks.index(attack_2)])):
                for j in range(len(targets)):
                    for k in range(len(targets)):
                        sum_res[j][k] += res_difAtk[classifier][attacks.index(attack_1)][attacks.index(attack_2)][i][j][k]
                    sum_cnt[j] += cnt_difAtk[classifier][attacks.index(attack_1)][attacks.index(attack_2)][i][j]
            print("\t\tClassifier: " + clfNames[classifier])
            f.write("\t\tClassifier: " + clfNames[classifier] + "\n")
            for orig in range(len(targets)):
                print("\t\t\tOriginal: " + targets[orig])
                f.write("\t\t\tOriginal: " + targets[orig] + "\n")
                for pred in range(len(targets)):
                    print("\t\t\t\t" + targets[pred] + ":" + str(round(sum_res[orig][pred] / sum_cnt[orig], 3) * 100) + " (" +str(sum_res[orig][pred]) + "/" + str(sum_cnt[orig]) + ")")
                    f.write("\t\t\t\t" + targets[pred] + ": " + str(round(sum_res[orig][pred] / sum_cnt[orig], 3) * 100) + " (" + str(sum_res[orig][pred]) + "/" + str(sum_cnt[orig]) + ")\n")

print("All Attacks")
f.write("All Attacks\n")
for target in targets:
    print("\tTarget: " + target)
    f.write("\tTarget: " + target + "\n")
    for classifier in range(len(classifiers)):
        sum_res = [[0 for i in range(len(targets))] for j in range(len(targets))]
        sum_cnt = [0 for i in range(len(targets))]
        for i in range(len(cnt_difTgt[classifier][targets.index(target)])):
            for j in range(len(targets)):
                for k in range(len(targets)):
                    sum_res[j][k] += res_difTgt[classifier][targets.index(target)][i][j][k]
                sum_cnt[j] += cnt_difTgt[classifier][targets.index(target)][i][j]
        print("\t\tClassifier: " + clfNames[classifier])
        f.write("\t\tClassifier: " + clfNames[classifier] + "\n")
        for orig in range(len(targets)):
            print("\t\t\tOriginal: " + targets[orig])
            f.write("\t\t\tOriginal: " + targets[orig] + "\n")
            for pred in range(len(targets)):
                 print("\t\t\t\t" + targets[pred] + ":" + str(round(sum_res[orig][pred] / sum_cnt[orig], 3) * 100) + " (" +str(sum_res[orig][pred]) + "/" + str(sum_cnt[orig]) + ")")
                 f.write("\t\t\t\t" + targets[pred] + ": " + str(round(sum_res[orig][pred] / sum_cnt[orig], 3) * 100) + " (" + str(sum_res[orig][pred]) + "/" + str(sum_cnt[orig]) + ")\n")

print("All Attacks, All Targets")
f.write("All Attacks, All Targets\n")
for classifier in range(len(classifiers)):
    sum_res = [[0 for i in range(len(targets))] for j in range(len(targets))]
    sum_cnt = [0 for i in range(len(targets))]
    for i in range(len(cnt_all[classifier])):
        for j in range(len(targets)):
            for k in range(len(targets)):
                sum_res[j][k] += res_all[classifier][i][j][k]
            sum_cnt[j] += cnt_all[classifier][i][j]
    print("\t\tClassifier: " + clfNames[classifier])
    f.write("\t\tClassifier: " + clfNames[classifier] + "\n")
    for orig in range(len(targets)):
        print("\t\t\tOriginal: " + targets[orig])
        f.write("\t\t\tOriginal: " + targets[orig] + "\n")
        for pred in range(len(targets)):
            print("\t\t\t\t" + targets[pred] + ":" + str(round(sum_res[orig][pred] / sum_cnt[orig], 3) * 100) + " (" +str(sum_res[orig][pred]) + "/" + str(sum_cnt[orig]) + ")")
            f.write("\t\t\t\t" + targets[pred] + ": " + str(round(sum_res[orig][pred] / sum_cnt[orig], 3) * 100) + " (" + str(sum_res[orig][pred]) + "/" + str(sum_cnt[orig]) + ")\n")

f.close()
