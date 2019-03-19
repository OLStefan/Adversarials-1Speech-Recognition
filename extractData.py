import os
import numpy as np
import scipy.io.wavfile as wav
import python_speech_features


def main():
    clean_dir = "originals"
    adv_dirs = ["adversarials-alzantot", "adversarials-carlini"]
    adv1_dirs = ["adversarials-alzantot-alzantot", "adversarials-alzantot-carlini", "adversarials-carlini-alzantot",
                 "adversarials-carlini-carlini"]
    attacks = ["alzantot", "carlini"]

    f = open("data.txt", "w+")
    for folder in adv1_dirs:
        print(folder)
        folder_index = adv1_dirs.index(folder)
        for subfolder in os.listdir(folder):
            print("\t" + subfolder)
            for input in os.listdir(os.path.join(folder, subfolder)):
                src_rate, src_wave = wav.read(os.path.join(folder, subfolder, input))
                src_mfcc_feat = python_speech_features.mfcc(src_wave, src_rate).flatten()

                src_wave_L0 = np.linalg.norm(src_wave, ord=0)
                src_wave_L2 = np.linalg.norm(src_wave, ord=2)
                src_wave_Linf = np.linalg.norm(src_wave, ord=np.inf)

                src_mfcc_L0 = np.linalg.norm(src_mfcc_feat, ord=1)
                src_mfcc_L2 = np.linalg.norm(src_mfcc_feat, ord=2)
                src_mfcc_Linf = np.linalg.norm(src_mfcc_feat, ord=np.inf)

                manip_1, orig, file = input.split('_', 2)
                orig_rate, orig_wave = wav.read(
                    os.path.join(adv_dirs[int(folder_index / 2)], manip_1, orig + '_' + file))
                orig_mfcc_feat = python_speech_features.mfcc(orig_wave, orig_rate).flatten()

                orig_wave_L0 = np.linalg.norm(orig_wave, ord=0)
                orig_wave_L2 = np.linalg.norm(orig_wave, ord=2)
                orig_wave_Linf = np.linalg.norm(orig_wave, ord=np.inf)

                orig_mfcc_L0 = np.linalg.norm(orig_mfcc_feat, ord=1)
                orig_mfcc_L2 = np.linalg.norm(orig_mfcc_feat, ord=2)
                orig_mfcc_Linf = np.linalg.norm(orig_mfcc_feat, ord=np.inf)

                diff_wave_L0 = orig_wave_L0 - src_wave_L0
                diff_wave_L2 = orig_wave_L2 - src_wave_L2
                diff_wave_Linf = orig_wave_Linf - src_wave_Linf
                diff_mfcc_L0 = orig_mfcc_L0 - src_mfcc_L0
                diff_mfcc_L2 = orig_mfcc_L2 - src_mfcc_L2
                diff_mfcc_Linf = orig_mfcc_Linf - src_mfcc_Linf

                # filename Orig 1stAttack 1stManip 2ndAttck 2ndManip diff_wave_L0 diff_wave_L2 diff_wave_Linf diff_mfcc_L1 diff_mfcc_L2 diff_mfcc_Linf
                f.write(file + "," + orig + "," + attacks[int(folder_index / 2)] + "," + manip_1 + "," + attacks[
                    folder_index % 2] + "," + subfolder +
                        "," + str(diff_wave_L0) + "," + str(diff_wave_L2) + "," + str(diff_wave_Linf) + "," + str(
                    diff_mfcc_L0) + "," + str(diff_mfcc_L2) + "," + str(diff_mfcc_Linf) + "\n")

    for folder in adv_dirs:
        print(folder)
        folder_index = adv_dirs.index(folder)
        for subfolder in os.listdir(folder):
            print("\t" + subfolder)
            for input in os.listdir(os.path.join(folder, subfolder)):
                src_rate, src_wave = wav.read(os.path.join(folder, subfolder, input))
                src_mfcc_feat = python_speech_features.mfcc(src_wave, src_rate).flatten()

                src_wave_L0 = np.linalg.norm(src_wave, ord=0)
                src_wave_L2 = np.linalg.norm(src_wave, ord=2)
                src_wave_Linf = np.linalg.norm(src_wave, ord=np.inf)

                src_mfcc_L1 = np.linalg.norm(src_mfcc_feat, ord=1)
                src_mfcc_L2 = np.linalg.norm(src_mfcc_feat, ord=2)
                src_mfcc_Linf = np.linalg.norm(src_mfcc_feat, ord=np.inf)

                orig, file = input.split('_', 1)
                orig_rate, orig_wave = wav.read(os.path.join(clean_dir, orig, file))
                orig_mfcc_feat = python_speech_features.mfcc(orig_wave, src_rate).flatten()

                orig_wave_L0 = np.linalg.norm(orig_wave, ord=0)
                orig_wave_L2 = np.linalg.norm(orig_wave, ord=2)
                orig_wave_Linf = np.linalg.norm(orig_wave, ord=np.inf)

                orig_mfcc_L1 = np.linalg.norm(orig_mfcc_feat, ord=1)
                orig_mfcc_L2 = np.linalg.norm(orig_mfcc_feat, ord=2)
                orig_mfcc_Linf = np.linalg.norm(orig_mfcc_feat, ord=np.inf)

                diff_wave_L0 = orig_wave_L0 - src_wave_L0
                diff_wave_L2 = orig_wave_L2 - src_wave_L2
                diff_wave_Linf = orig_wave_Linf - src_wave_Linf
                diff_mfcc_L1 = orig_mfcc_L1 - src_mfcc_L1
                diff_mfcc_L2 = orig_mfcc_L2 - src_mfcc_L2
                diff_mfcc_Linf = orig_mfcc_Linf - src_mfcc_Linf

                # filename Orig - - 2ndAttck 2ndManip diff_wave_L0 diff_wave_L2 diff_wave_Linf diff_mfcc_L1 diff_mfcc_L2 diff_mfcc_Linf
                f.write(file + "," + orig + ",-,-," + attacks[folder_index % 2] + "," + subfolder +
                        "," + str(diff_wave_L0) + "," + str(diff_wave_L2) + "," + str(diff_wave_Linf) + "," + str(
                    diff_mfcc_L1) + "," + str(diff_mfcc_L2) + "," + str(diff_mfcc_Linf) + "\n")
    f.close()


main()
