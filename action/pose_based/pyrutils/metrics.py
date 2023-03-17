import numpy as np
from collections import Counter
from pyrutils.itertools import run_length_encoding
from pyrutils.utils import run_length_encoding_intervals
from sklearn.metrics import f1_score

def f1_at_k_single_example_v2(y_true, y_pred, num_classes, overlap):
    target_intervals = np.array(list(run_length_encoding_intervals(y_true)))
    target_ids = np.array(next(zip(*run_length_encoding(y_true))))
    true_positives = np.zeros(num_classes, dtype=np.float32)
    false_positives = np.zeros(num_classes, dtype=np.float32)
    used_true_segments = np.zeros(len(target_ids), dtype=np.float32)
    idx = 0
    for interval, id in zip(target_intervals, target_ids):
        left, right = interval[0], interval[1]
        pred_segment = y_pred[left:right]
        c = Counter(pred_segment)
        most_common_id = c.most_common(1)[0][0]
        # print('-------------------------')
        # print('most common:', most_common_id)
        # print('id:', id)
        ratio = c[most_common_id]/(right - left)
        # true prediction and Overlap > threshold
        if most_common_id == id and ratio >= overlap:
            true_positives[id] += 1
            used_true_segments[idx] = 1
        else:
            false_positives[most_common_id] += 1
        idx += 1

    # F1 score calculation
    true_positives = np.sum(true_positives).item()
    false_positives = np.sum(false_positives).item()

    # False negatives are any unused true segments.
    false_negatives = len(used_true_segments) - np.sum(used_true_segments).item()
    # print('-------------------------')
    # print('len target intervals:', len(target_intervals))
    # print('TP:', true_positives)
    # print('FP:', false_positives)
    # print('FN:', false_negatives)
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0.0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0.0
    
    return f1
def f1_at_k_v2(y_true, y_pred, num_classes: int, *, overlap: float, ignore_value: float = None) -> float:
    """Compute the F1@k metric between a (batch) target and a (batch) predicted segmentation.

    See f1_at_k_single_example for explanation of arguments. The only difference, the ignore_value argument, is
    meant to remove padding labels from the evaluation.
    """
    f1 = 0.0
    effective_examples = 0.0
    for y_t, y_p in zip(y_true, y_pred):
        if ignore_value is not None:
            y_t, y_p = np.array(y_t), np.array(y_p)
            indices = y_t != ignore_value
            y_t, y_p = y_t[indices], y_p[indices]
        # if y_t.size == 0:
        #     continue
        if len(y_t) == 0:
            continue
        f1 += f1_at_k_single_example_v2(y_t, y_p, num_classes, overlap=overlap)
        effective_examples += 1
    return f1 / effective_examples

def f1_at_k_single_example(y_true, y_pred, num_classes: int, *, overlap: float) -> float:
    """Compute the F1@k metric between a single target and a single predicted segmentation.

    To ignore some classes during computation of the F1@k metric, simply make sure their ID is at least as high as
    the number of classes.

    Arguments:
        y_true - Ground-truth labels. The segmentation is extracted from the identical consecutive labels.
        y_pred - Predicted labels. The segmentation is extracted from the identical consecutive labels.
        num_classes - Number of classes.
        overlap - The minimum overlap between ground-truth and predicted segments to count as a true positive. This is
            the 'k' in the function name. It must be a value between 0.0 and 1.0.
    Returns:
        The F1@k score between the ground-truth and predicted segmentations.
    """
    target_intervals = np.array(list(run_length_encoding_intervals(y_true)))
    target_ids = np.array(next(zip(*run_length_encoding(y_true))))
    # print(type(target_ids))

    # print(target_intervals)
    # print(y_true)
    output_intervals = np.array(list(run_length_encoding_intervals(y_pred)))
    output_ids = np.array(next(zip(*run_length_encoding(y_pred))))
    # print('target intervals:', target_intervals)
    # print('output intervals:', output_intervals)
    # We keep track of the per-class TPs and FPs, but in the end we just sum over them.
    true_positives = np.zeros(num_classes, dtype=np.float32)
    false_positives = np.zeros(num_classes, dtype=np.float32)
    used_true_segments = np.zeros(len(target_ids), dtype=np.float32)
    for output_interval, output_id in zip(output_intervals, output_ids):
        # Compute IoU of a predicted segment against all ground-truth segments.
        # print('---------------------------------')
        # print('output interval:', output_interval)
        # print('target intervals:', target_intervals)
        intersection = (np.minimum(output_interval[1], target_intervals[:, 1]) -
                        np.maximum(output_interval[0], target_intervals[:, 0]))
        # print('intersection:', intersection)
        union = (np.maximum(output_interval[1], target_intervals[:, 1]) -
                 np.minimum(output_interval[0], target_intervals[:, 0]))
        # print('union:', union)
        # print('ouput_id:', output_id)
        # print('target_id:', target_ids)
        iou = (intersection / union) * (output_id == target_ids)
        # print('iou:', iou)       
        idx = np.argmax(iou).item()
        if output_id >= num_classes:
            continue
        if iou[idx] >= overlap and not used_true_segments[idx]:
            true_positives[output_id] += 1
            used_true_segments[idx] = 1
        else:
            false_positives[output_id] += 1
    true_positives = np.sum(true_positives).item()
    false_positives = np.sum(false_positives).item()
    # False negatives are any unused true segments.
    false_negatives = len(used_true_segments) - np.sum(used_true_segments).item()
    # print('-------------------------')
    # print('TP:', true_positives)
    # print('FP:', false_positives)
    # print('FN:', false_negatives)
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0.0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0.0
    return f1


def f1_at_k(y_true, y_pred, num_classes: int, *, overlap: float, ignore_value: float = None) -> float:
    """Compute the F1@k metric between a (batch) target and a (batch) predicted segmentation.

    See f1_at_k_single_example for explanation of arguments. The only difference, the ignore_value argument, is
    meant to remove padding labels from the evaluation.
    """
    f1 = 0.0
    effective_examples = 0.0
    for y_t, y_p in zip(y_true, y_pred):
        if ignore_value is not None:
            y_t, y_p = np.array(y_t), np.array(y_p)
            indices = y_t != ignore_value
            y_t, y_p = y_t[indices], y_p[indices]
        # if y_t.size == 0:
        #     continue
        if len(y_t) == 0:
            continue
        f1 += f1_at_k_single_example(y_t, y_p, num_classes, overlap=overlap)
        effective_examples += 1
    return f1 / effective_examples * 5

def f1(y_true, y_pred):
    f1 = 0.0
    effective_examples = 0.0
    for y_t, y_p in zip(y_true, y_pred):
        if len(y_t) == 0:
            continue
        f1 += f1_score(y_t, y_p, average='macro')
        effective_examples += 1
    return f1 / effective_examples