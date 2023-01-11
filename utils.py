import tensorflow as tf
import math

def top_k_logits(logits, k):
    v, ix = tf.math.top_k(logits, k)  # 전체 vocab에서 top k 결과만 선택
    out = logits.numpy()
    out[out < v.numpy()[:, [-1]]] = -math.inf
    
    return out

def sample(model, x, num_sampling, temperature=1.0, sample=False, top_k=None):
    # 주어진 input text(=x)를 기반으로 새로운 text data를 sampling
    block_size = model.block_size
    # num_sampling 개수만큼 sampling
    for k in range(num_sampling):
        print(f'sampling progress : {k}/{num_sampling}')
        # input text가 block size보다 클경우 block size만큼 crop
        x_cond = x if x.shape[1] <= block_size else x[:, -block_size:]
        
        logits = model(x_cond)
        # prediction의 가장 마지막 block의 token 값을 가져오고 temperature값만큼 보정
        logits = logits[:, -1, :] / temperature

        # 전체 후보군 vocab 중에서 top_k개의 값만 선택되도록 보정
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        # softmax 함수를 적용해서 sum to 1 확률 값 형태로 변형
        probs = tf.nn.softmax(logits, axis=-1)

        if sample:
            # 각 분포에 따른 확률을 기반으로 random sampling
            sampled_token = tf.random.categorical(logits, 1,dtype=tf.int32)
        else:
            # 무조건 가장 확률값이 큰 token을 선택
            _, sampled_token = tf.math.top_k(probs, k=1)
        
        # sampling으로 추출된 character를 input text에 append
        x = tf.concat((x, sampled_token), axis=1)

    return x