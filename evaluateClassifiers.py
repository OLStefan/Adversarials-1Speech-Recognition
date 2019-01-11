import sklearn.base as base
import sklearn.model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

targets = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
attacks = ["alzanot", "carlini"]
classifiers = [MLPClassifier(hidden_layer_sizes=(4, 8, 16), max_iter=100, learning_rate_init=0.1),
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
    if attack_1 == "-":
        samples[0][attacks.index(attack_2)][targets.index(target_2)].append(a)
        samples[1][attacks.index(attack_2)][targets.index(target_2)].append(a)
        labels[0][attacks.index(attack_2)][targets.index(target_2)].append(int(target_1 == "-"))
        labels[1][attacks.index(attack_2)][targets.index(target_2)].append(int(target_1 == "-"))
    else:
        samples[attacks.index(attack_1)][attacks.index(attack_2)][targets.index(target_2)].append(a)
        labels[attacks.index(attack_1)][attacks.index(attack_2)][targets.index(target_2)].append(int(target_1 == "-"))

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
    tp, tn, fp, fn = 0, 0, 0, 0
    clf = base.clone(classifier)
    clf.fit(samples_train, labels_train)
    pred = clf.predict(samples_test)
    for i in range(len(pred)):
        if labels_test[i] == 0:
            if pred[i] == labels_test[i]:
                tn += 1
            else:
                fp += 1
        else:
            if pred[i] == labels_test[i]:
                tp += 1
            else:
                fn += 1
    return tp, tn, fp, fn


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
                    a = classify(classifier, samples_train, samples_test, labels_train, labels_test)
                    cnt_difAtk_difTgt[classifiers.index(classifier)][attacks.index(attack_1)][attacks.index(attack_2)][
                        targets.index(target)].append(a)
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
                a = classify(classifier, samples_train, samples_test, labels_train, labels_test)
                cnt_difAtk[classifiers.index(classifier)][attacks.index(attack_1)][attacks.index(attack_2)].append(a)
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
            a = classify(classifier, samples_train, samples_test, labels_train, labels_test)
            cnt_difTgt[classifiers.index(classifier)][targets.index(target)].append(a)
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
        a = classify(classifier, samples_train, samples_test, labels_train, labels_test)
        cnt_all[classifiers.index(classifier)].append(a)

f = open("results_evaluateClassifiers.txt", "w+")

# Ausgabe
for attack_1 in attacks:
    for attack_2 in attacks:
        print("Attack 1: " + attack_1 + ", Attack 2: " + attack_2)
        f.write("Attack 1: " + attack_1 + ", Attack 2: " + attack_2 + "\n")
        for target in targets:
            print("\tTarget: " + target)
            f.write("\tTarget: " + target + "\n")
            for classifier in range(len(classifiers)):
                tp, tn, fp, fn = 0, 0, 0, 0
                for i in range(len(cnt_difAtk_difTgt[classifier][attacks.index(attack_1)][attacks.index(attack_2)][
                                       targets.index(target)])):
                    tp += cnt_difAtk_difTgt[classifier][attacks.index(attack_1)][attacks.index(attack_2)][
                        targets.index(target)][i][0]
                    tn += cnt_difAtk_difTgt[classifier][attacks.index(attack_1)][attacks.index(attack_2)][
                        targets.index(target)][i][1]
                    fp += cnt_difAtk_difTgt[classifier][attacks.index(attack_1)][attacks.index(attack_2)][
                        targets.index(target)][i][2]
                    fn += cnt_difAtk_difTgt[classifier][attacks.index(attack_1)][attacks.index(attack_2)][
                        targets.index(target)][i][3]
                print("\t\tClassifier: " + clfNames[classifier])
                print("\t\t\tSensitivity/True Positive Rate (%): " + str(round(((tp / (tp + fn)) * 100), 2)) + " (" + str(
                    tp) + "/" + str(tp + fn) + ")")
                print("\t\t\tSpecificity/True Negative Rate (%): " + str(round(((tn / (tn + fp)) * 100), 2)) + " (" + str(
                    tn) + "/" + str(tn + fp) + ")")
                print(
                    "\t\t\tOverall Correct Predictions (%): " + str(round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)))
                f.write("\t\tClassifier: " + clfNames[classifier] + "\n")
                f.write(
                    "\t\t\tSensitivity/True Positive Rate (%): " + str(round(((tp / (tp + fn)) * 100), 2)) + " (" + str(
                        tp) + "/" + str(tp + fn) + ")" + "\n")
                f.write(
                    "\t\t\tSpecificity/True Negative Rate (%): " + str(round(((tn / (tn + fp)) * 100), 2)) + " (" + str(
                        tn) + "/" + str(tn + fp) + ")" + "\n")
                f.write("\t\t\tOverall Correct Predictions (%): " + str(
                    round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)) + "\n")

for attack_1 in attacks:
    for attack_2 in attacks:
        print("Attack 1: " + attack_1 + ", Attack 2: " + attack_2)
        print("\tAll Targets")
        f.write("Attack 1: " + attack_1 + ", Attack 2: " + attack_2 + "\n")
        f.write("\tAll Targets\n")
        for classifier in range(len(classifiers)):
            tp, tn, fp, fn = 0, 0, 0, 0
            for i in range(len(cnt_difAtk[classifier][attacks.index(attack_1)][attacks.index(attack_2)])):
                tp += cnt_difAtk[classifier][attacks.index(attack_1)][attacks.index(attack_2)][i][0]
                tn += cnt_difAtk[classifier][attacks.index(attack_1)][attacks.index(attack_2)][i][1]
                fp += cnt_difAtk[classifier][attacks.index(attack_1)][attacks.index(attack_2)][i][2]
                fn += cnt_difAtk[classifier][attacks.index(attack_1)][attacks.index(attack_2)][i][3]
            print("\t\tClassifier: " + clfNames[classifier])
            print("\t\t\tSensitivity/True Positive Rate (%): " + str(round(((tp / (tp + fn)) * 100), 2)) + " (" + str(
                tp) + "/" + str(tp + fn) + ")")
            print("\t\t\tSpecificity/True Negative Rate (%): " + str(round(((tn / (tn + fp)) * 100), 2)) + " (" + str(
                tn) + "/" + str(tn + fp) + ")")
            print("\t\t\tOverall Correct Predictions (%): " + str(round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)))
            f.write("\t\tClassifier: " + clfNames[classifier] + "\n")
            f.write("\t\t\tSensitivity/True Positive Rate (%): " + str(round(((tp / (tp + fn)) * 100), 2)) + " (" + str(
                tp) + "/" + str(tp + fn) + ")" + "\n")
            f.write("\t\t\tSpecificity/True Negative Rate (%): " + str(round(((tn / (tn + fp)) * 100), 2)) + " (" + str(
                tn) + "/" + str(tn + fp) + ")" + "\n")
            f.write("\t\t\tOverall Correct Predictions (%): " + str(
                round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)) + "\n")

print("All Attacks")
f.write("All Attacks\n")
for target in targets:
    print("\tTarget: " + target)
    f.write("\tTarget: " + target + "\n")
    for classifier in range(len(classifiers)):
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(cnt_difTgt[classifier][targets.index(target)])):
            tp += cnt_difTgt[classifier][targets.index(target)][i][0]
            tn += cnt_difTgt[classifier][targets.index(target)][i][1]
            fp += cnt_difTgt[classifier][targets.index(target)][i][2]
            fn += cnt_difTgt[classifier][targets.index(target)][i][3]
        print("\t\tClassifier: " + clfNames[classifier])
        print("\t\t\tSensitivity/True Positive Rate (%): " + str(round(((tp / (tp + fn)) * 100), 2)) + " (" + str(
            tp) + "/" + str(tp + fn) + ")")
        print("\t\t\tSpecificity/True Negative Rate (%): " + str(round(((tn / (tn + fp)) * 100), 2)) + " (" + str(
            tn) + "/" + str(tn + fp) + ")")
        print("\t\t\tOverall Correct Predictions (%): " + str(round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)))
        f.write("\t\tClassifier: " + clfNames[classifier] + "\n")
        f.write("\t\t\tSensitivity/True Positive Rate (%): " + str(round(((tp / (tp + fn)) * 100), 2)) + " (" + str(
            tp) + "/" + str(tp + fn) + ")" + "\n")
        f.write("\t\t\tSpecificity/True Negative Rate (%): " + str(round(((tn / (tn + fp)) * 100), 2)) + " (" + str(
            tn) + "/" + str(tn + fp) + ")" + "\n")
        f.write(
            "\t\t\tOverall Correct Predictions (%): " + str(round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)) + "\n")

print("All Attacks, All Targets")
f.write("All Attacks, All Targets\n")
for classifier in range(len(classifiers)):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(cnt_all[classifier])):
        tp += cnt_all[classifier][i][0]
        tn += cnt_all[classifier][i][1]
        fp += cnt_all[classifier][i][2]
        fn += cnt_all[classifier][i][3]
    print("\tClassifier: " + clfNames[classifier])
    print("\t\t\tSensitivity/True Positive Rate (%): " + str(round(((tp / (tp + fn)) * 100), 2)) + " (" + str(
        tp) + "/" + str(tp + fn) + ")")
    print("\t\t\tSpecificity/True Negative Rate (%): " + str(round(((tn / (tn + fp)) * 100), 2)) + " (" + str(
        tn) + "/" + str(tn + fp) + ")")
    print("\t\t\tOverall Correct Predictions (%): " + str(round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)))
    f.write("\tClassifier: " + clfNames[classifier] + "\n")
    f.write("\t\t\tSensitivity/True Positive Rate (%): " + str(round(((tp / (tp + fn)) * 100), 2)) + " (" + str(
        tp) + "/" + str(tp + fn) + ")" + "\n")
    f.write("\t\t\tSpecificity/True Negative Rate (%): " + str(round(((tn / (tn + fp)) * 100), 2)) + " (" + str(
        tn) + "/" + str(tn + fp) + ")" + "\n")
    f.write("\t\t\tOverall Correct Predictions (%): " + str(round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)) + "\n")

f.close()
