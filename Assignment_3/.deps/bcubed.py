"""Minimal pure-Python implementation of bcubed metrics to satisfy vendor import."""

def precision(predicted, gold):
    correct = 0
    total = 0
    for item in predicted:
        pred_clusters = set(predicted[item])
        gold_clusters = set(gold[item])
        for other in predicted:
            if set(predicted[other]) & pred_clusters:
                total += 1
                if set(gold[other]) & gold_clusters:
                    correct += 1
    return correct / total if total > 0 else 0.0

def recall(predicted, gold):
    return precision(gold, predicted)

def fscore(p, r, beta=1.0):
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / ((beta**2 * p) + r)
