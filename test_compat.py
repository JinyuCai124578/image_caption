from pycocoevalcap.spice.spice import Spice

scorers = [ Spice()]
for scorer in scorers:
    key_to_refs = {
        "image1": ["这是一个测试参考。"],
        "image2": ["这是另一个测试参考。"],
    }
    key_to_pred = {
        "image1": ["这是一个测试参考。"],
        "image2": ["这是一个错误的预测。"],
    }
    # hypo = list(key_to_pred.values())
    # # ref=list(key_to_refs.values())
    score, scores = scorer.compute_score(key_to_refs, key_to_pred)
    print('1')













# from pycocoevalcap.spice.spice import Spice
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# Spice()

# import json
# import urllib.request


# def baidu_search():
#     url = "https://www.baidu.com/s?"
#     data = {"wd": "啊哈"}
#     data = json.dumps(data).encode('UTF-8')
#     import ssl
#     ssl._create_default_https_context = ssl._create_unverified_context  # 如果不添加这两行，下一行报错
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#     }
#     request = urllib.request.Request(url, data, headers=headers)
#     response = urllib.request.urlopen(request)
#     content = response.read()
#     print(str(content))


# if __name__ == '__main__':
#     baidu_search()

# import requests

# def baidu_search():
#     url = "https://www.baidu.com/s?"
#     params = {"wd": "啊哈"}
#     response = requests.get(url, params=params)
#     print(response.text)

# if __name__ == '__main__':
#     baidu_search()