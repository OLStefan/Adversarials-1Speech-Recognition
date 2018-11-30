# Repository for my Master's Thesis 'Adversarials<sup>-1</sup> in der Spracherkennung: Erkennung und Abwehr'

## Used Attacks

The Attacks I used to create the samples are:

* https://github.com/carlini/audio_adversarial_examples (named carlini)
* https://github.com/nesl/adversarial_audio (named alzanot)

## Used Datasets

The dataset used to create the samples was an excerpt from the Speech Commands Dataset (https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data).
Only the words used were: down, go, left, no, off, on, right, stop, up, and yes. 

## Structure of Adversarial Samples

The folders adversarials-alazanot and adversarials-carlini contain samples which were manipulated with the attack in the folder name once.
The name of the subfolder was the target word for the attack. The file names are structured as follows: [original_word]_[original_filename].
The orignal files used for the manipulated samples is found in the originals folder.

The other folders contain samples which were manipulated twice. The attacks in the folder name indicate which attacks were used.
The name of the subfolder was the target word for the attack. The file names are structured as follows: [first_attack_target]\_[original_word]\_[original_filename].
The orignal files used for the twice manipulated samples is found in either the adversarials-alazanot or the adversarials-carlini folder depending on which attack was used in the first attack (indicated by the first attack in the folder name).

## Calculated Results

The script recognizeAttack.py uses the samples to train different calssifiers to identify files which were manipulated.

The script uses different partitions of the samples to train the classifiers. Used Partitions are:

* Partitioned by 1st attack, 2nd attack, and target word used in the 2nd attack
* Partitioned by 1st attack, and 2nd attack
* Partitioned by target word used in the 2nd attack
* Not partitioned

For each of these partitioned the script calculates:

* Sensitivity/True Positive Rate (TP/(TP+FN))
* Specificity/True Negative Rate (TN/(TN+FP))
* Overall Correct Predictions ((TP+TN)/(TP+TN+FP+FN))
