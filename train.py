import tensorflow as tf
import math
import numpy as np
import os
from absl import flags
from absl import app
from model import GPTConfig
from model import GPT


flags.DEFINE_integer('num_epochs', default=10, help='training epochs')
flags.DEFINE_string('checkpoint_filename', default='saved_model', help='filename to save model checkpoints during training')

FLAGS = flags.FLAGS

# 설정값 지정
block_size = 128        # training시에 사용할 token 묶음 size (GPU 메모리 사이즈에 따라 적절히 수정)
batch_size = 12         # batch size (GPU 메모리 사이즈에 따라 적절히 수정)
learning_rate = 2.5e-4  # 러닝 레이트

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

# 데이터셋 설정
train_dataset = tf.data.Dataset.from_generator(train_dataset_gen, (tf.int32, tf.int32))
train_dataset = train_dataset.batch(batch_size)

# optimizer & cross entropy loss 설정
optimizer = tf.optimizers.Adam(learning_rate)
cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def train_step(inputs, model, optimizer):
    # X : block_size만큼의 character
    # Y : 한글자 뒤로 민 block_size만큼의 character
    X, Y = inputs

    with tf.GradientTape() as tape:
        # input data를 통한 prediction
        logits = model(X, training=True) # batch_size x block_size x vocab_size

        # reshape
        num_labels = tf.shape(logits)[-1]     
        pred_logits = tf.reshape(logits, (-1, num_labels)) # (batch_size x block_size) x vocab_size
        target_y = tf.reshape(Y,(-1,))    # batch_size x block_size
        
        # loss 계산
        cross_entropy = cross_entropy_loss(target_y, pred_logits)
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