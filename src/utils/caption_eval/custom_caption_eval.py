"""
from
https://gist.github.com/kracwarlock/c979b10433fe4ac9fb97
"""

from src.utils.caption_eval.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from src.utils.caption_eval.pycocoevalcap.bleu.bleu import Bleu
# from src.utils.caption_eval.pycocoevalcap.meteor.meteor import Meteor
from src.utils.caption_eval.pycocoevalcap.rouge.rouge import Rouge
from src.utils.caption_eval.pycocoevalcap.cider.cider import Cider
# from src.utils.caption_eval.pycocoevalcap.spice.spice import Spice
# from src.utils.caption_eval.pycocoevalcap.wmd.wmd import WMD


class COCOEvalCap:
    def __init__(self, images, gts, res):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.params = {'image_id': images}
        self.gts = gts
        self.res = res

    def evaluate(self):
        imgIds = self.params['image_id']
        gts = self.gts
        res = self.res

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            # (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
            # (WMD(),   "WMD"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def calculate_metrics(rng, datasetGTS, datasetRES):
    imgIds = rng
    gts = {}
    res = {}

    imgToAnnsGTS = {ann['image_id']: [] for ann in datasetGTS['annotations']}
    for ann in datasetGTS['annotations']:
        imgToAnnsGTS[ann['image_id']] += [ann]

    imgToAnnsRES = {ann['image_id']: [] for ann in datasetRES['annotations']}
    for ann in datasetRES['annotations']:
        imgToAnnsRES[ann['image_id']] += [ann]

    for imgId in imgIds:
        gts[imgId] = imgToAnnsGTS[imgId]
        res[imgId] = imgToAnnsRES[imgId]

    evalObj = COCOEvalCap(imgIds, gts, res)
    evalObj.evaluate()
    return evalObj.eval


if __name__ == '__main__':
    rng = range(2)
    datasetGTS = {
        'annotations': [{'image_id': 0, 'caption': 'the man is playing a guitar'},
                        {'image_id': 0, 'caption': 'a man is playing a guitar'},
                        {'image_id': 1, 'caption': 'a woman is slicing cucumbers'},
                        {'image_id': 1, 'caption': 'the woman is slicing cucumbers'},
                        {'image_id': 1, 'caption': 'a woman is cutting cucumbers'}]
    }
    datasetRES = {
        'annotations': [{'image_id': 0, 'caption': 'man is playing guitar'},
                        {'image_id': 1, 'caption': 'a woman is cutting vegetables'}]
    }
    print(calculate_metrics(rng, datasetGTS, datasetRES))
