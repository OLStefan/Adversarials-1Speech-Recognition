import os
import numpy as np
import scipy.io.wavfile as wav
import python_speech_features

clean_dir = "originals"
adv_dirs = ["adversarials-alzantot", "adversarials-carlini"]
adv1_dirs = ["adversarials-alzantot-alzantot", "adversarials-alzantot-carlini", "adversarials-carlini-alzantot",
             "adversarials-carlini-carlini"]
targets = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
attacks = ["alzantot", "carlini"]
samples = [[[[] for i in range(len(targets))] for j in range(len(attacks))] for k in range(len(attacks))]
correct = [[[0 for i in range(len(targets))] for j in range(len(attacks))] for k in range(len(attacks))]


class Sample:
    distance = [-1 for i in range(len(targets))]

    def __init__(self, o):
        self.orig = o


for attack_1 in attacks:
    print(attack_1)
    for target_1 in range(len(targets)):
        print("\t" + targets[target_1])
        for file in os.listdir(os.path.join("adversarials-" + attack_1, targets[target_1])):
            for attack_2 in attacks:
                if not os.path.isfile(
                        os.path.join("adversarials-" + attack_1 + "-" + attack_2,
                                     targets[(target_1 + 1) % len(targets)], targets[target_1] + "_" + file)):
                    continue
                orig_rate, orig_wave = wav.read(os.path.join("adversarials-" + attack_1, targets[target_1], file))
                orig, filename = file.split('_', 1)
                s = Sample(orig)
                for target in targets:
                    if target == targets[target_1]:
                        continue
                    if not os.path.isfile(os.path.join("adversarials-" + attack_1 + "-" + attack_2, target,
                                                       targets[target_1] + "_" + file)):
                        print("Missing, " + os.path.join("adversarials-" + attack_1 + "-" + attack_2, target,
                                                         targets[target_1] + "_" + file))
                        break
                    attacked_rate, attacked_wave = wav.read(
                        os.path.join("adversarials-" + attack_1 + "-" + attack_2, target,
                                     targets[target_1] + "_" + file))
                    s.distance[targets.index(target)] = np.linalg.norm(np.subtract(orig_wave, attacked_wave))
                samples[attacks.index(attack_1)][attacks.index(attack_2)][target_1].append(s)
        for attack_2 in attacks:
            print("\t\t" + str(len(samples[attacks.index(attack_1)][attacks.index(attack_2)][target_1])))

cnt_samples = 0
for i in range(len(samples)):
    for j in range(len(samples[i])):
        for k in range(len(samples[i][j])):
            for l in range(len(samples[i][j][k])):
                cnt_samples += 1
print(str(cnt_samples))

for i in range(len(samples)):
    for j in range(len(samples[i])):
        for k in range(len(samples[i][j])):
            for l in range(len(samples[i][j][k])):
                min = -1
                for m in range(len(samples[i][j][k][l].distance)):
                    if m == -1:
                        continue
                    if min == -1 or samples[i][j][k][l].distance[m] < samples[i][j][k][l].distance[min]:
                        min = m
                if min == targets.index(samples[i][j][k][l].orig):
                    correct[i][j][k] += 1

for i in range(len(correct)):
    for j in range(len(correct[i])):
        print("Attack 1: " + attacks[i] + ", Attack 2: " + attacks[j])
        for k in range(len(correct[i][j])):
            print("\tTarget: " + targets[k])
            percent = correct[i][j][k] / len(samples[i][j][k])
            print("\t\tCorrect: " + str(percent))
