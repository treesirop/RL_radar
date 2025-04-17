#!/usr/bin/python
import numpy as np
import logging

def calc_CSI_reg(pred, true, threshold):
    # Ensure input arrays have matching shapes
    if pred.shape != true.shape:
        raise ValueError('y_pred and y_true shape not match!')

    # Create binary masks using vectorized operations
    pred_labels = (pred >= threshold).astype(np.float32)
    true_labels = (true >= threshold).astype(np.float32)

    # Compute metrics
    CSI, POD, FAR, TP, FP, FN, TN, acc, CI_acc, NCI_acc = get_CSI(pred_labels, true_labels)

    logging.info('CSI %5.3f, POD %5.3f, FAR  %5.3f, threshold %5.3f' % (CSI, POD, FAR, threshold))
    logging.info(' TP, FP, FN, TN: %d, %d, %d, %d' % (TP, FP, FN, TN))
    logging.info(' accuracy: %5.3f ' % acc)
    logging.info(' CI_accuracy: %5.3f ' % CI_acc)
    logging.info(' NCI_accuracy: %5.3f ' % NCI_acc)

    return np.array([CSI, POD, FAR])

def get_CSI(dec_labels, true_labels):
    '''Calculate the CSI, POD, FAR and return them'''
    # Compute confusion matrix elements using vectorized operations
    TP = float(np.sum((dec_labels == 1) & (true_labels == 1)))
    FP = float(np.sum((dec_labels == 1) & (true_labels == 0)))
    FN = float(np.sum((dec_labels == 0) & (true_labels == 1)))
    TN = float(np.sum((dec_labels == 0) & (true_labels == 0)))

    # Compute CSI, POD, FAR
    CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    POD = TP / (TP + FN) if (TP + FN) > 0 else 0
    FAR = FP / (FP + TP) if (FP + TP) > 0 else 0

    # Compute accuracies
    acc = (TP + TN) / (TP + TN + FP + FN)
    CI_acc = POD  # Same as POD
    NCI_acc = TN / (TN + FP) if (TN + FP) > 0 else 0

    return [CSI, POD, FAR, TP, FP, FN, TN, acc, CI_acc, NCI_acc]

def find_best_CSI(dec_valuee, labels, confident=0, dbz=False):
    '''Find the best CSI when POD >= 0.6'''
    if dec_valuee.shape[0] != labels.shape[0]:
        print('dimensions do not match')
        return

    dec_value = dec_valuee
    num = dec_value.shape[0]
    best_CSI = 0

    if confident == 0:
        # Search through confidence levels
        for temp_conf in np.linspace(0.1, 0.9, 9):
            dec_labels = np.ones(num, dtype=int)
            dec_labels[dec_value >= temp_conf if not dbz else dec_value < temp_conf] = 0
            CSI, POD, FAR, TP, FP, FN, TN, acc, CI_acc, NCI_acc = get_CSI(dec_labels, labels)

            if POD > 0 and CSI > best_CSI:
                best_CSI = CSI
                confident = temp_conf

        # Fine-tune around best confidence
        left_margin = max(0, confident - 0.2)
        right_margin = min(1, confident + 0.2)

        for temp_conf in np.arange(left_margin, right_margin, 0.001):
            dec_labels = np.ones(num, dtype=int)
            dec_labels[dec_value >= temp_conf if not dbz else dec_value < temp_conf] = 0
            CSI, POD, FAR, TP, FP, FN, TN, acc, CI_acc, NCI_acc = get_CSI(dec_labels, labels)

            if POD > 0 and CSI > best_CSI:
                best_CSI = CSI
                confident = temp_conf

    # Final evaluation with best confidence
    dec_labels = np.ones(num, dtype=int)
    dec_labels[dec_value >= confident if not dbz else dec_value < confident] = 0
    CSI, POD, FAR, TP, FP, FN, TN, acc, CI_acc, NCI_acc = get_CSI(dec_labels, labels)

    # Log results
    msg = 'CSI %5.3f, POD %5.3f, FAR  %5.3f, Confident %5.3f' % (CSI, POD, FAR, confident)
    print(msg)
    logging.info(msg)

    msg = ' TP, FP, FN, TN: %d, %d, %d, %d' % (TP, FP, FN, TN)
    print(msg)
    logging.info(msg)

    msg = ' accuracy: %5.3f' % acc
    print(msg)
    logging.info(msg)

    msg = ' CI_accuracy: %5.3f' % CI_acc
    print(msg)
    logging.info(msg)

    msg = ' NCI_accuracy: %5.3f' % NCI_acc
    print(msg)
    logging.info(msg)

    return [CSI, POD, FAR, confident]