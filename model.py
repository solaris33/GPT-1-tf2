import tensorflow as tf
import math

class GPTConfig:
    # GPT 설정값
    embedding_dropout_rate = 0.1
    residual_dropout_rate = 0.1
    attention_dropout_rate = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

def gelu(x):
    # Gaussian Error Linear Unit(GELU) activation 구현
    cdf = 0.5 * (1.0 + tf.tanh((math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))

    return x * cdf

class MultiHeadAttention(tf.keras.layers.Layer):
    # Multi-Head Attention 구현

    def __init__(self, d_model, num_heads, attention_dropout_rate, residual_dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        # query, key, value projections for all heads
        self.wq = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),name="query")
        self.wk = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),name="key")
        self.wv = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),name="value")
        # dropout 적용
        self.attention_dropout = tf.keras.layers.Dropout(rate=attention_dropout_rate)
        self.residual_dropout = tf.keras.layers.Dropout(rate=residual_dropout_rate)
        # output projection
        self.dense = tf.keras.layers.Dense(d_model, name="projection")

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask, training):
        batch_size = tf.shape(x)[0]

        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(x)  # (batch_size, seq_len, d_model)
        v = self.wv(x)  # (batch_size, seq_len, d_model)
        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # (..., seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        attention_weights = self.attention_dropout(attention_weights, training=training)
        # (..., seq_len_q, depth_v)
        scaled_attention = tf.matmul(attention_weights, v)
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        output = self.residual_dropout(output, training=training)

        return output

def point_wise_feed_forward_network(d_model, dff, residual_dropout_rate):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, 
                                                      activation=tf.keras.activations.get(gelu),
                                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)),
                                # (batch_size, seq_len, dff)
                                tf.keras.layers.Dense(d_model),
                                # (batch_size, seq_len, d_model)
                                tf.keras.layers.Dropout(residual_dropout_rate)
                                ])

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, attention_dropout_rate, residual_dropout_rate):
        super(TransformerLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, 
                                      num_heads,
                                      attention_dropout_rate, 
                                      residual_dropout_rate)
        self.ffn = point_wise_feed_forward_network(d_model, d_model * 4, residual_dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, mask, training):
        x = x + self.mha(self.layernorm1(x), mask, training=training)
        x = x + self.ffn(self.layernorm2(x), training=training)

        return x

# GPT 모델 구현
class GPT(tf.keras.Model):
    def __init__(self, config):
        super().__init__()

        # 인풋 토큰 embedding
        # vocab_size x embedding_size
        self.tok_emb = tf.keras.layers.Embedding(config.vocab_size,
                                                 config.n_embd,
                                                 embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        
        # positional embedding
        # block_size x embedding_size
        self.position_embedding = self.add_weight("position_embeddings",
                                       shape=[config.block_size,
                                              config.n_embd],
                                       initializer=tf.keras.initializers.Zeros(),
                                       dtype=tf.float32)
        
        # embedding dropout
        self.drop = tf.keras.layers.Dropout(config.embedding_dropout_rate)        

        # Transformer layer
        self.transformer_blocks = [TransformerLayer(
                                    config.n_embd,
                                    config.n_head,
                                    config.attention_dropout_rate,
                                    config.residual_dropout_rate)
                       for _ in range(config.n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        # vocab_size만큼 logits을 출력하는 최종 output layer
        self.head = tf.keras.layers.Dense(config.vocab_size, 
                                          use_bias=False,
                                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        # batch_size x block_size x vocab_size

        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer

    def call(self, inputs, training=False):
        # inputs: batch_size x block_size
        t = tf.shape(inputs)[1] # t: block_size

        # GPT 모델에 inputs 데이터 추가
        # input vector(vocab_size) -> embedding vector(embedding_size)
        token_embeddings = self.tok_emb(inputs)       # batch_size x block_size x embedding_size

        # We used learned position embeddings instead of the sinusoidal version proposed in the original work.
        position_embeddings = tf.expand_dims(tf.slice(self.position_embedding, [0, 0], [t, self.n_embd]),
                                             axis=0)
        # batch_size x block_size x embedding_size
        x = self.drop(token_embeddings + position_embeddings, training=training)

        # 더 뒤에 있는 data를 참조하지 않기 위한 masking 생성
        mask = 1 - tf.linalg.band_part(tf.ones((t, t)), -1, 0)
        # [[0. 1. 1. ... 1. 1. 1.]
        # [0. 0. 1. ... 1. 1. 1.]
        # [0. 0. 0. ... 1. 1. 1.]
        # ...
        # [0. 0. 0. ... 0. 1. 1.]
        # [0. 0. 0. ... 0. 0. 1.]
        # [0. 0. 0. ... 0. 0. 0.]]

        # Transformer
        for i in range(self.n_layer):
            x = self.transformer_blocks[i](x, mask, training=training)
        x = self.ln_f(x)

        # batch_size x block_size x vocab_size
        logits = self.head(x)

        return logits