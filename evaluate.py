import tensorflow as tf
import numpy as np
import math
import os
from absl import flags
from absl import app
from model import GPTConfig
from model import GPT
from utils import sample

flags.DEFINE_string('checkpoint_filename', default='saved_model', help='filename to save model checkpoints during training')
flags.DEFINE_integer('num_sampling', default=2000, help='token length to sample')
flags.DEFINE_float('temperature', default=0.9, help='low temperature -> accuracy up, high temperature -> random probabilty up')
flags.DEFINE_string('initial_input_text', default='KING RICH', help='initial input text to sample')

FLAGS = flags.FLAGS

# 설정값 지정
block_size = 128 

# Tiny Shakesphere 데이터셋을 위한 Dataset class 설정
class CharDataset:
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        # data_size : txt 파일에 존재하는 전체 chracter의 개수, vocab_size : vocab dict에 있는 unique한 character 개수
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }   # string to integer
        self.itos = { i:ch for i,ch in enumerate(chars) }   # integer to string
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __iter__(self):
        for _ in range(self.__len__()):
            # 전체 텍스트에서 random한 위치에서 block_size만큼의 charcter를 선택
            i = np.random.randint(0, len(self.data) - (self.block_size + 1))
            chunk = self.data[i:i+self.block_size+1]
            dix = [self.stoi[s] for s in chunk]
            x = tf.convert_to_tensor(dix[:-1], dtype=tf.int32)
            # LM(Language Modelling)의 정답 label : 한글자 뒤에올 character를 예측
            y = tf.convert_to_tensor(dix[1:], dtype=tf.int32)

            yield x, y
    
    __call__ = __iter__

# Tiny Shakesphere 데이터셋 불러오기
text = open('input.txt', 'r').read()
train_dataset_gen = CharDataset(text, block_size) 

# GPT 모델 설정값 지정후 인스턴스 선언
model_config = GPTConfig(
                  train_dataset_gen.vocab_size, \
                  train_dataset_gen.block_size, \
                  n_layer=12, \
                  n_head=12, \
                  n_embd=768)
gpt_model = GPT(model_config)

def main(_):
    # 학습된 weight가 있으면 불러옵니다.
    if os.path.exists(FLAGS.checkpoint_filename + '.index'):
        print('학습된 weight를 불러옵니다.')
        gpt_model.load_weights(FLAGS.checkpoint_filename)
    else:
        print('학습된 weight가 없습니다.')
        exit()

    # Tiny Shakesphere에 학습된 모델을 이용해서 Sampling을 진행합니다.
    # input : 주어진 input text    
    x = tf.convert_to_tensor([train_dataset_gen.stoi[s] for s in FLAGS.initial_input_text], dtype=tf.int32)[None,...]
    # y: 주어진 input text를 기반으로 num_sampling 개수만큼 GPT가 sampling한 text
    # 낮은 temperature 값은 더욱 정확한 텍스트를 생성합니다.
    # 높은 temperature 값은 더욱 다양한 텍스트를 생성합니다.
    y = sample(gpt_model, x, FLAGS.num_sampling, temperature=FLAGS.temperature, sample=True, top_k=5)[0]
    sampled_text = ''.join([train_dataset_gen.itos[int(i)] for i in y])

    # 샘플링된 텍스트 출력
    print(sampled_text)

if __name__ == '__main__':
    app.run(main)