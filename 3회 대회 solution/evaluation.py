import argparse
import numpy as np
import sklearn.metrics

def evaluate(prediction_labels, gt_labels):
    f1 = sklearn.metrics.f1_score(gt_labels,prediction_labels, pos_label='0')
    return f1


def read_prediction_pt(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    pt = [list(l.replace("\n","").split(' '))[2] for l in lines]
    return pt


def read_prediction_gt(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    gt = [list(l.replace("\n", "").split(','))[2] for l in lines]
    return gt[1:]


def evaluation_metrics(prediction_file, testset_path):
    prediction_labels = read_prediction_pt(prediction_file)
    gt_labels = read_prediction_gt(testset_path)

    return evaluate(prediction_labels, gt_labels)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction_file', type=str, default='pred.txt')
    args.add_argument('--test_file', type=str, default='test_label')

    config = args.parse_args()

    print(evaluation_metrics(config.prediction_file, config.test_file))