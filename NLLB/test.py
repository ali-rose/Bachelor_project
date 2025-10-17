import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import sacrebleu

# 样本数据
references = ["Gwen Tennyson ist die Cousine von Ben Tennyson und die Enkelin von Max Tennyson. In der Originalserie hatte Gwen grüne Augen und kurzes rotes Haar, das von einer blauen Haarspange gehalten wurde, und trug Saphirohrringe, ein ellbogenlanges blaues Raglanhemd mit Katzenlogo, weiße Caprihosen und weiße Turnschuhe mit dunkelblauen Streifen ohne Socken. Sie hat ihre Sommerferien auf einem Roadtrip mit Ben und Max verbracht, um Abenteuer zu erleben und gegen Bösewichte zu kämpfen, seien es Außerirdische oder Menschen."]
hypotheses = [" Gwen Tennyson ist die Cousine von Ben Tennyson und die Enkelin von Max Tennyson. In der Originalserie hatte Gwen grüne Augen und kurze rote Haare, die von einer blauen Haarschleife gehalten wurden, und trug Saphirohrringe, ein Ellenbogenlänge blaues Raglan-Shirt mit einem Katzenlogo, weiße Capri-Hosen und weiße Turnschuhe mit dunkelblauen Streifen ohne Socken. Sie verbrachte ihren Sommerurlaub auf einer Roadtrip mit "]

# 计算BLEU分数
def calculate_bleu(reference, hypothesis):
    smoothing_function = SmoothingFunction().method4
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothing_function)

# 计算METEOR分数
def calculate_meteor(reference, hypothesis):
    return meteor_score([reference], hypothesis)

# 计算CHR-F分数
def calculate_chrf(reference, hypothesis):
    return sacrebleu.corpus_chrf([hypothesis], [[reference]]).score

# 输出评估结果
for ref, hyp in zip(references, hypotheses):
    bleu = calculate_bleu(ref, hyp)
    # meteor = calculate_meteor(ref.split(), hyp.split())  # 这里进行分词处理
    chrf = calculate_chrf(ref, hyp)
    print(f"BLEU Score: {bleu}")
    # print(f"METEOR Score: {meteor}")
    print(f"CHR-F Score: {chrf}")
