# Repository for my Master's Thesis 'Adversarials<sup>-1</sup> in der Spracherkennung: Erkennung und Abwehr'

## Used Attacks

The Attacks I used to create the samples are:

* https://github.com/carlini/audio_adversarial_examples
* https://github.com/nesl/adversarial_audio

## Used Datasets

The dataset used to create the samples was an excerpt from the Speech Commands Dataset (https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data).
Only the following words were used: `down`, `go`, `left`, `no`, `off`, `on`, `right`, `stop`, `up`, and `yes`. 

## Structure of Adversarial Samples
Under releases there are three archives of audio files, `originals` which contains the not attacked-samples from the original dataset, `adversarials` which contains one time attacked samples. The two folders `adversarials-alazanot` and `adversarials-carlini` contain samples which were manipulated with the attack in the folder name once. The name of the subfolder was the target word for the attack. The file names are structured as follows: [original_word]_[original_filename].
`Adversarials-1` contains samples which were manipulated twice. The attacks in the folder name indicate which attacks were used. The name of the subfolder was the target word for the attack. The file names are structured as follows: [first_attack_target]\_[original_word]\_[original_filename]. The orignal files used for the twice manipulated samples is found in either the `adversarials-alazanot` or the `adversarials-carlini` folder depending on which attack was used in the first attack (indicated by the first attack in the folder name).

## ExtractData.py
The script creates the dataset used to train the classifiers in the thesis. It extracts the original word, the first and second target (as applicable) as well as the differences between the L0, L2, and LInfinity norms from the read wav data and the L1, L2, and LInfinity norm of the MFCC matrices.

## EvaluateClassifiers.py

The script `evaluateClassifiers` uses the samples to train different calssifiers to identify files which weren't manipulated.

The script uses different partitions of the samples to train the classifiers. Used Partitions are:

* Partitioned by 1st attack, 2nd attack, and target word used in the 2nd attack
* Partitioned by 1st attack, and 2nd attack
* Partitioned by target word used in the 2nd attack
* Not partitioned

For each of these partitioned the script calculates:

* Sensitivity/True Positive Rate (TP/(TP+FN))
* Specificity/True Negative Rate (TN/(TN+FP))
* Overall Correct Predictions ((TP+TN)/(TP+TN+FP+FN))

For this case, True Positives refers to samples which were correctly predicted as non-manipulated, likewise True Negative refers to files which were correctly predicted as manipulated.

## recoverOriginal_*.py

The scripts are used to recover the original word from twice attacked audio files. The scripts use the same partitions that evaluateClassifiers uses.

* recoverOriginal_classifier uses classfiers and the possible words as classes.
* recoverOriginal_distance calculates the distance between a once attacked sample and all the twice attacked samples based on that and the target resulting in the smallest distance is predicted as original word.
* recoverOriginal_clustering uses KMeans and DBSCAN clustering to predict the original word.
