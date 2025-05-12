import argparse
import pandas as pd
import json
import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

from utils.util import ptb_tokenize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prediction_file", default='/home/bingxing2/ailab/caijinyu/image_captioning/experiments/resnet101_attention/resnet101_attention_b64_emd300_predictions.json', type=str)
    parser.add_argument("-r", "--reference_file", default="/home/bingxing2/ailab/caijinyu/image_captioning/data/caption.txt", type=str)
    parser.add_argument("-j", "--json_output_file", default="result.json", type=str)  # 输出 JSON 文件
    args = parser.parse_args()
    
    prediction_df = pd.read_json(args.prediction_file)
    key_to_pred = dict(zip(prediction_df["img_id"], prediction_df["prediction"]))
    
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

    # results = []
    # for img_id in key_to_pred.keys():
    #     result = {
    #         "img_id": img_id,
    #         "prediction": key_to_pred[img_id],
    #         "references": key_to_refs[img_id],
    #         "scores": {}
    #     }
    #     print('ref:',key_to_refs[img_id], 'pred:', key_to_pred[img_id])
    #     for scorer in scorers:
    #         score, _ = scorer.compute_score(key_to_refs, {img_id: key_to_pred[img_id]})
    #         method = scorer.method()
    #         result["scores"][method] = score
            
    #     results.append(result)

    # # 将所有结果保存到 JSON 文件
    # with open(args.json_output_file, "w") as json_writer:
    #     json.dump(results, json_writer, indent=4)
    
    scores_dict={}
    for scorer in scorers:
        score, scores = scorer.compute_score(key_to_refs, key_to_pred)
        
        print(np.array(scores).shape)
        
        method = scorer.method()
        if method == "Bleu":
            scores=np.array(scores).T
            for n in range(4):
                print("Bleu-{}: {:.3f}".format(n + 1, score[n]))
            scores_dict[method]=scores.tolist()
        else:

            print(f"{method}: {score:.3f}")
            for idx in range(len(scores)):
                img_id=list(key_to_refs.keys())[idx]  
            scores_dict[method]=scores
    results=[]
    for idx in range(len(scores)):
        img_id=list(key_to_refs.keys())[idx]  
        result = {
            "img_id": img_id,
            "prediction": key_to_pred[img_id],
            "references": key_to_refs[img_id],
            "scores": {}
        }
        for method in scores_dict.keys():
            result["scores"][method] = scores_dict[method][idx]
        results.append(result)
    with open(args.json_output_file, "w") as json_writer:
        json.dump(results, json_writer, indent=4)