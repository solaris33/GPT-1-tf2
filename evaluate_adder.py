import tensorflow as tf
import numpy as np
import os
from absl import flags
from absl import app
from model import GPTConfig
from model import GPT
from utils import sample

flags.DEFINE_string('checkpoint_filename', default='saved_model_adder', help='filename to save model checkpoints during training')

FLAGS = flags.FLAGS

# 덧셈(Adder) 데이터셋을 위한 Dataset class 설정
class AdditionDataset():    
    """
    GPT를 이용한 덧셈 문제 해결 학습을 위한 데이터셋

    (n)-자리수끼리의 덧셈은 (n+1) 자리수의 결과를 출력합니다.
    따라서 [(n)-자리수 숫자,(n)-자리수 숫자,(n+1)-자리수 숫자]를 이은 형태로 데이터를 구성합니다.
    
    예를 들어, 2-자리수 덧셈 형태로 데이터를 구성하면 아래와 같습니다.
    - 85 + 50 = 135 -> [8, 5, 5, 0, 1, 3, 5]
    - 6 + 39 = 45 -> [0, 6, 3, 9, 0, 4, 5]

    GPT를 이용해서 정답의 마지막 자리수의 숫자를 예측하도록 학습시킵니다.
    즉, [(n)-자리수 숫자, (n)-자리수 숫자, 정답의 (n)-자리수 숫자]는 입력으로 주어진다고 가정합니다.

    예를 들어, [0, 6, 3, 9, 0, 4]를 GPT 입력값으로 넣으면
    [0, 4, 5]를 GPT가 출력하길 예상합니다.    
    """

    def __init__(self, ndigit, split):
        self.split = split     # train/test 분리
        self.ndigit = ndigit   # 자리수 설정
        self.vocab_size = 10   # 10 = 숫자 0~9
        self.block_size = ndigit + ndigit + ndigit
        
        # 데이터를 traing set과 test set으로 나눕니다.
        num = (10**self.ndigit)**2      # 가능한 모든 조합의 경우의 수
        r = np.random.RandomState(1337) # deterministic하도록 random seed 값 고정
        perm = r.permutation(num)
        num_test = min(int(num*0.2), 1000) # 전체 데이터셋의 20%나 최대 1000개까지를 test dataset으로 지정
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def __len__(self):
        return self.ixes.size

    def __iter__(self):
        # 주어진 idx에서 a + b = c 형태의 데이터를 생성
        for idx in range(self.__len__()):
            idx = self.ixes[idx]
            # e.g.
            # idx : 1842
            # nd : 100
            # a : 18
            # b : 42
            # c : 70
            nd = 10**self.ndigit
            a = idx // nd
            b = idx %  nd
            c = a + b            
            render = f'%0{self.ndigit}d%0{self.ndigit}d%0{self.ndigit+1}d' % (a,b,c) # e.g. 03+25=28 becomes "0325028"
            dix = [int(s) for s in render] # convert string to integer

            # X는 [(n)-자리수 숫자,(n)-자리수 숫자, 정답의 앞 (n)-자리수 숫자]입니다.
            x = dix[:-1]
            y = dix[1:] # 마지막에 올 token을 예측합니다.            
            # 마지막 (n+1) 자리수만을 학습을 위한 타겟 y로 설정합니다.
            # -1은 masking으로 loss 계산에서 배제됩니다.
            # 예를 들어,
            # X: [8 1 1 6 0 9]
            # Y: [-1 -1 -1  0  9  7]
            # masking : [False False False True True True]
            y[:self.ndigit*2-1] = [-1] * (self.ndigit*2-1)
            x = tf.convert_to_tensor(x, dtype=tf.int32)
            y = tf.convert_to_tensor(y, dtype=tf.int32)

            yield x, y
            
    __call__ = __iter__

ndigit = 2
#train_dataset_gen = AdditionDataset(ndigit=ndigit, split='train')
test_dataset_gen = AdditionDataset(ndigit=ndigit, split='test')

test_dataset = tf.data.Dataset.from_generator(test_dataset_gen, (tf.int32,tf.int32))

model_config = GPTConfig(
                  test_dataset_gen.vocab_size, \
                  test_dataset_gen.block_size, \
                  n_layer=12, \
                  n_head=12, \
                  n_embd=768)
gpt_model = GPT(model_config)

# GPT를 이용한 덧셈(Adder) 예측 성능을 평가합니다.
def give_exam(dataset, model, batch_size=32, max_batches=-1):
    results = []
    
    loader = dataset.batch(batch_size)
    for batch_index, (x, y) in enumerate(loader):
        d1d2 = x[:, :ndigit*2]
        d1d2d3 = sample(model, d1d2, ndigit+1)
        d3 = d1d2d3[:, -(ndigit+1):]

        factors = tf.convert_to_tensor([[10**i for i in range(ndigit+1)][::-1]])
        # convert to integer
        d1i = tf.reduce_sum((d1d2[:,:ndigit] * factors[:,1:]),axis=1)
        d2i = tf.reduce_sum((d1d2[:,ndigit:ndigit*2] * factors[:,1:]),axis=1)
        d3i_pred = tf.reduce_sum((d3 * factors),axis=1)
        d3i_gt = d1i + d2i

        correct = (d3i_pred == d3i_gt)
        for i in range(x.shape[0]):
            results.append(int(correct[i]))
            judge = 'O' if correct[i] else 'X'
            print("GPT claims that %03d + %03d = %03d (gt is %03d; %s)" 
                    % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i], judge))
    
        if max_batches >= 0 and batch_index+1 >= max_batches:
            break

    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))

def main(_):
    # 학습된 weight가 있으면 불러옵니다.
    if os.path.exists(FLAGS.checkpoint_filename + '.index'):
        print('학습된 weight를 불러옵니다.')
        gpt_model.load_weights(FLAGS.checkpoint_filename)
    else:
        print('학습된 weight가 없습니다.')
        exit()

    give_exam(test_dataset, gpt_model, batch_size=32, max_batches=-1)

if __name__ == '__main__':
    app.run(main)