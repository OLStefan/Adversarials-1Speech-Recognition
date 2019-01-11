import numpy as np

import sklearn.base as base
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN

targets = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
attacks = ["alzantot", "carlini"]
clusterMethods = [KMeans(10), DBSCAN(metric='cityblock'), DBSCAN(metric='euclidean'), DBSCAN(metric='l1'),
                  DBSCAN(metric='l2'), DBSCAN(metric='manhattan')]
clusterNames = ["KMeans", "DBSCAN (Cityblock)", "DBSCAN (Euclidean)", "DBSCAN (L1)", "DBSCAN (L2)",
                "DBSCAN (Manhattan)"]
reps = 1
f = open("data.txt", "r")
f1 = f.readlines()

samples = [[[[] for i in range(len(targets))] for j in range(len(attacks))] for k in range(len(attacks))]
labels = [[[[] for i in range(len(targets))] for j in range(len(attacks))] for k in range(len(attacks))]

res_difAtk_difTgt = [[[[0 for i in range(len(targets))] for j in range(len(attacks))] for k in range(len(attacks))]
                     for l in range(len(clusterMethods))]
res_difAtk = [[[0 for j in range(len(attacks))] for k in range(len(attacks))] for l in range(len(clusterMethods))]
res_difTgt = [[0 for i in range(len(targets))] for l in range(len(clusterMethods))]
res_all = [0 for l in range(len(clusterMethods))]

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


def cluster(cluster, samples, labels_true):
    clt = base.clone(cluster)

    labels_pred = clt.fit_predict(samples)
    print("\t\tScore: " + str(metrics.adjusted_rand_score(labels_true, labels_pred)))
    print("\t\tEst Clusters: " + str(len(set(clt.labels_))))
    return metrics.adjusted_rand_score(labels_true, labels_pred)


for rep in range(reps):
    for attack_1 in attacks:
        for attack_2 in attacks:
            for target in targets:
                print(
                    "Rep: " + str(rep + 1) + ", Attack 1: " + attack_1 + ", Attack 2: " + attack_2 + ", Target: " + str(
                        target))
                samples_part = samples[attacks.index(attack_1)][attacks.index(attack_2)][targets.index(target)]
                labels_part = labels[attacks.index(attack_1)][attacks.index(attack_2)][targets.index(target)]
                for classifier in clusterMethods:
                    print("\tClassifier: " + str(clusterMethods.index(classifier)))
                    res_difAtk_difTgt[clusterMethods.index(classifier)][attacks.index(attack_1)][
                        attacks.index(attack_2)][targets.index(target)] += (
                        cluster(classifier, samples_part, labels_part))
            print("Rep: " + str(rep + 1) + ", Attack 1: " +
                  attack_1 + ", Attack 2: " + attack_2 + ", all Targets")
            samples_part = []
            labels_part = []
            for i in range(len(samples[attacks.index(attack_1)][attacks.index(attack_2)])):
                for j in range(len(samples[attacks.index(attack_1)][attacks.index(attack_2)][i])):
                    samples_part.append(samples[attacks.index(attack_1)][attacks.index(attack_2)][i][j])
                    labels_part.append(labels[attacks.index(attack_1)][attacks.index(attack_2)][i][j])
            for classifier in clusterMethods:
                print("\tClassifier: " + str(clusterMethods.index(classifier)))
                res_difAtk[clusterMethods.index(classifier)][attacks.index(attack_1)][attacks.index(attack_2)] += (
                    cluster(classifier, samples_part, labels_part))
    for target in targets:
        print("Rep: " + str(rep + 1) + ", all Attacks, Target: " + target)
        samples_part = []
        labels_part = []
        for i in range(len(samples)):
            for j in range(len(samples[i])):
                for k in range(len(samples[i][j][targets.index(target)])):
                    samples_part.append(samples[i][j][targets.index(target)][k])
                    labels_part.append(labels[i][j][targets.index(target)][k])
        for classifier in clusterMethods:
            print("\tClassifier: " + str(clusterMethods.index(classifier)))
            res_difTgt[clusterMethods.index(classifier)][targets.index(target)] += (
                cluster(classifier, samples_part, labels_part))
    print("Rep: " + str(rep + 1) + ", all Attacks, all Targets")
    samples_part = []
    labels_part = []
    for i in range(len(samples)):
        for j in range(len(samples[i])):
            for k in range(len(samples[i][j])):
                for l in range(len(samples[i][j][k])):
                    samples_part.append(samples[i][j][k][l])
                    labels_part.append(labels[i][j][k][l])
    for classifier in clusterMethods:
        print("\tClassifier: " + str(clusterMethods.index(classifier)))
        res_all[clusterMethods.index(classifier)] += (cluster(classifier, samples_part, labels_part))

f = open("results_recoverOriginal_clustering.txt", "w+")

# Ausgabe
for attack_1 in attacks:
    for attack_2 in attacks:
        print("Attack 1: " + attack_1 + ", Attack 2: " + attack_2)
        f.write("Attack 1: " + attack_1 + ", Attack 2: " + attack_2 + "\n")
        for target in targets:
            print("\tTarget: " + target)
            f.write("\tTarget: " + target + "\n")
            for classifier in range(len(clusterMethods)):
                print("\t\tClassifier: " + clusterNames[classifier])
                f.write("\t\tClassifier: " + clusterNames[classifier] + "\n")
                print("\t\t\tScore: " + str(
                    res_difAtk_difTgt[classifier][attacks.index(attack_1)][attacks.index(attack_2)][
                        targets.index(target)] / reps))
                f.write("\t\t\tScore: " + str(
                    res_difAtk_difTgt[classifier][attacks.index(attack_1)][attacks.index(attack_2)][
                        targets.index(target)] / reps) + "\n")

for attack_1 in attacks:
    for attack_2 in attacks:
        print("Attack 1: " + attack_1 + ", Attack 2: " + attack_2)
        print("\tAll Targets")
        f.write("Attack 1: " + attack_1 + ", Attack 2: " + attack_2 + "\n")
        f.write("\tAll Targets\n")
        for classifier in range(len(clusterMethods)):
            print("\t\tClassifier: " + clusterNames[classifier])
            f.write("\t\tClassifier: " + clusterNames[classifier] + "\n")
            print(
                "\t\t\tScore: " + str(res_difAtk[classifier][attacks.index(attack_1)][attacks.index(attack_2)] / reps))
            f.write(
                "\t\t\tScore: " + str(
                    res_difAtk[classifier][attacks.index(attack_1)][attacks.index(attack_2)] / reps) + "\n")

print("All Attacks")
f.write("All Attacks\n")
for target in targets:
    print("\tTarget: " + target)
    f.write("\tTarget: " + target + "\n")
    for classifier in range(len(clusterMethods)):
        print("\t\tClassifier: " + clusterNames[classifier])
        f.write("\t\tClassifier: " + clusterNames[classifier] + "\n")
        print("\t\t\tScore: " + str(res_difTgt[classifier][targets.index(target)] / reps))
        f.write("\t\t\tScore: " + str(res_difTgt[classifier][targets.index(target)] / reps) + "\n")

print("All Attacks, All Targets")
f.write("All Attacks, All Targets\n")
for classifier in range(len(clusterMethods)):
    print("\t\tClassifier: " + clusterNames[classifier])
    f.write("\t\tClassifier: " + clusterNames[classifier] + "\n")
    print("\t\t\tScore: " + str(res_all[classifier] / reps))
    f.write("\t\t\tScore: " + str(res_all[classifier] / reps) + "\n")

f.close()
