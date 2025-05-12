import argparse
import pandas as pd

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from utils.util import ptb_tokenize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prediction_file",default='/home/bingxing2/ailab/caijinyu/image_captioning/experiments/resnet101_attention/resnet101_attention_b64_emd300_predictions.json', type=str)
    parser.add_argument("-r", "--reference_file",default="/home/bingxing2/ailab/caijinyu/image_captioning/data/caption.txt", type=str)
    parser.add_argument("-o", "--output_file",default="result.txt", type=str)
    args = parser.parse_args()
    prediction_df = pd.read_json(args.prediction_file)
    # [n, var_len]
    key_to_pred = dict(zip(prediction_df["img_id"], prediction_df["prediction"]))
    # [n, 5, var_len]
    captions = open(args.reference_file, "r").read().strip().split("\n")
    key_to_refs = {}
    for i, row in enumerate(captions):
        row = row.split("\t")
        row[0] = row[0][: len(row[0]) - 2]  # filename#0 caption
        if row[0] not in key_to_pred:
            continue
        if row[0] in key_to_refs:
            key_to_refs[row[0]].append(row[1])
        else:
            key_to_refs[row[0]] = [row[1]]

    scorers = [Bleu(n=4), Rouge(), Meteor(), Cider()]
    key_to_refs = ptb_tokenize(key_to_refs)
    key_to_pred = ptb_tokenize(key_to_pred)

    output = {"SPIDEr": 0}
    with open(args.output_file, "w") as writer:
        for scorer in scorers:
            score, scores = scorer.compute_score(key_to_refs, key_to_pred)
            method = scorer.method()
            output[method] = score
            if method == "Bleu":
                for n in range(4):
                    print("Bleu-{}: {:.3f}".format(n + 1, score[n]), file=writer)
            else:
                print(f"{method}: {score:.3f}", file=writer)
            if method in ["CIDEr", "SPICE"]:
                output["SPIDEr"] += score
        output["SPIDEr"] /= 2
        print(f"SPIDEr: {output['SPIDEr']:.3f}", file=writer)

