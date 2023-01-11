import tensorflow as tf
import numpy as np
import os
from absl import flags
from absl import app
from model import GPTConfig
from model import GPT


flags.DEFINE_integer('num_epochs', default=10, help='training epochs')
flags.DEFINE_string('checkpoint_filename', default='saved_model_adder', help='filename to save model checkpoints during training')

FLAGS = flags.FLAGS

# 설정값 지정
batch_size = 24         # batch size (GPU 메모리 사이즈에 따라 적절히 수정)
learning_rate = 2.5e-4  # 러닝 레이트

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
        r = np.random.RandomState(1337) # 데이터셋이 deterministic하도록 random seed 값 고정
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
train_dataset_gen = AdditionDataset(ndigit=ndigit, split='train')
#test_dataset_gen = AdditionDataset(ndigit=ndigit, split='test')

# GPT 모델 설정
model_config = GPTConfig(
                  train_dataset_gen.vocab_size, \
                  train_dataset_gen.block_size, \
                  n_layer=12, \
                  n_head=12, \
                  n_embd=768)
gpt_model = GPT(model_config)

# 데이터셋 설정
train_dataset = tf.data.Dataset.from_generator(train_dataset_gen, (tf.int32, tf.int32))
train_dataset = train_dataset.batch(batch_size)

# set optimizer & cross entropy loss
optimizer = tf.optimizers.Adam(learning_rate)
cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def train_step(inputs, model, optimizer):
    # 마지막 (n+1) 자리수만을 학습을 위한 타겟 y로 설정합니다.
    # -1은 masking으로 loss 계산에서 배제됩니다.
    # 예를 들어,
    # X: [8 1 1 6 0 9]
    # Y: [-1 -1 -1  0  9  7]
    X, Y = inputs

    with tf.GradientTape() as tape:
        logits = model(X, training=True) # batch_size x block_size x vocab_size

        num_labels = tf.shape(logits)[-1]     # vocab_size : 10
        label_mask = tf.math.logical_not(Y < 0)
        # label_mask : [False False False  True  True  True]
        label_mask = tf.reshape(label_mask,(-1,))

        # logits과 target_y에 masking 적용
        # tf.boolean_mask를 이용해서 mask가 True인 부분만 남기고 False인 부분 삭제
        logits = tf.reshape(logits, (-1, num_labels))
        pred_logits_masked = tf.boolean_mask(logits, label_mask)        
        target_y = tf.reshape(Y,(-1,))
        target_y_masked = tf.boolean_mask(target_y, label_mask)
        
        cross_entropy = cross_entropy_loss(target_y_masked, pred_logits_masked)
        loss = tf.reduce_sum(cross_entropy) / batch_size

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

    return loss

def main(_):
    # 학습된 weight가 있으면 불러옵니다.
    if os.path.exists(FLAGS.checkpoint_filename + '.index'):
        print('학습된 weight를 불러옵니다.')
        gpt_model.load_weights(FLAGS.checkpoint_filename)

    num_batch = len(list(train_dataset))
    # 지정된 epoch 횟수만큼 training을 진행합니다.
    for epoch in range(FLAGS.num_epochs):
        for iter, inputs in enumerate(train_dataset.as_numpy_iterator()):
            loss = train_step(inputs, gpt_model, optimizer)

            print(f'epoch: {epoch+1}/{FLAGS.num_epochs} iter: {iter+1}/{num_batch} loss: {loss}')
        
        # epoch이 끝날때마다 weight를 저장합니다.
        gpt_model.save_weights(FLAGS.checkpoint_filename)

if __name__ == '__main__':
    app.run(main)