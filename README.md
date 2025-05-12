安装 requirements.txt 中的包, 准备好环境。


跑 baseline:
```bash
sbatch run.sh
```

过程中如果提示缺少资源，则按照指示下载（例如 'punkt_tab'）:
```python
>>> import nltk
>>> nltk.download('punkt_tab')
```

注: 
evaluate.py 用于计算指标，预测结果 `prediction.json` 写成这样的形式:
```json
[
    {
        "img_id":"1386964743_9e80d96b05.jpg",
        "prediction":[
            "young boy is standing in a field of grass"
        ]
    },
    {
        "img_id":"3523559027_a65619a34b.jpg",
        "prediction":[
            "young boy is standing in a field of grass"
        ]
    },
    ......
]
```
调用方法：
```bash
python evaluate.py --prediction_file prediction.json \
                   --reference_file /lustre/home/acct-stu/stu168/data/image_captioning/flickr8k/caption.txt \
                   --output_file result.txt
```

myEvaluate.py 用于生成一个 json 文件，其中包含图片ID和对应的分数。
调用方法：
```bash
python myEvaluate.py --prediction_file prediction.json \
                   --reference_file /lustre/home/acct-stu/stu168/data/image_captioning/flickr8k/caption.txt \
                   --json_output_file result.json
```
sort_result.py 用于将 json 文件中的分数降序排列。
调用方法：
```bash
python sort_result.py 
```