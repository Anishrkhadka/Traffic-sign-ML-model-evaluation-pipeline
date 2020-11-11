# Evaluation.py

from Evaluation_helper import apply_noise_attack, plot_figure, anomaly_detection, \
    recoganise_traffic_signs, anomaly_traffic_sign_reconstruction, \
    noise_attacks, noise_attacks_name, labelNames, x_test, y_test, apply_pattern_attack, maskPath, anomaly_detection_per_class
import LogManager

import numpy as np


def run_test_noise():
    for pattern in range(len(noise_attacks)):
        LogManager.displayLog(f'Noise Type {noise_attacks_name[pattern]}',
                              'yellow')
        for noise in range(1, 6):
            LogManager.displayLog(f'Intensity value:{noise}', 'grey')
            normal, abnormal, size_type = apply_noise_attack(
                x_test, noise_attacks[pattern], noise, 1)
            LogManager.displayLog(f'Area of attack ({size_type} of {48 * 48}) = {size_type / 2304 * 100}%')
            plot_figure(abnormal[np.random.randint(0, abnormal.shape[0], 100)],
                        f'results/preview_attacked_{noise_attacks_name[pattern]}_{noise}.png')
            anomaly_detection(
                normal, abnormal,
                f'results/confuse_matrix_{noise_attacks_name[pattern]}_{noise}.png'
            )
            LogManager.displayLog(f'Before Reconstruction', 'red')
            recoganise_traffic_signs(abnormal, y_test)
            reconstruction = anomaly_traffic_sign_reconstruction(abnormal)
            LogManager.displayLog(f'After Reconstruction', 'yellow')
            recoganise_traffic_signs(reconstruction, y_test)


def run_test_pattern():
    for pattern in range(len(maskPath)):
        attack_name = maskPath[pattern].split('/')[-1].split('.')[-2]
        LogManager.displayLog(f'Pattern Attack Type {attack_name}', 'yellow')
        normal, abnormal, size_type = apply_pattern_attack(x_test, pattern)
        plot_figure(abnormal[np.random.randint(0, abnormal.shape[0], 100)],
                    f'results/preview_attacked_{attack_name}.png')
        anomaly_detection_per_class(normal, abnormal, f'results/confuse_mat_anomaly_before_{attack_name}')
        LogManager.displayLog(f'Before Reconstruction', 'red')
        recoganise_traffic_signs(abnormal, y_test, f'results/confuse_mat_recoganise_before_{attack_name}.png')
        reconstruction = anomaly_traffic_sign_reconstruction(abnormal)
        LogManager.displayLog(f'After Reconstruction', 'yellow')
        recoganise_traffic_signs(reconstruction, y_test, f'results/confuse_mat_recoganise_after_{attack_name}.png')


run_test_pattern()
