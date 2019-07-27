import pickle
import tensorflow as tf
import numpy as np


def load_policy(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    # assert len(data.keys()) == 2
    nonlin_type = data['nonlin_type']
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]

    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

    # Keep track of input and output dims (i.e. observation and action dims) for the user

    # build_policy functions as forward pass that takes obs -> action

    def build_policy(obs):
        def read_layer(l):
            # Returns 'W' and 'b' from a layer
            assert list(l.keys()) == ['AffineLayer']
            assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
            return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)

        def apply_nonlin(x):
            if nonlin_type == 'lrelu':
                return tf.nn.lrelu(x, leak=.01)  # openai/imitation nn.py:233
            elif nonlin_type == 'tanh':
                return tf.tanh(x)
            else:
                raise NotImplementedError(nonlin_type)

        # Build the policy. First, observation normalization.
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        # print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)
        # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation
        normed_obs = (obs - obsnorm_mean) / (obsnorm_stdev + 1e-6)

        h = normed_obs

        # Hidden layers next
        assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            h = apply_nonlin(tf.matmul(h, W) + b)

        # Output layer
        W, b = read_layer(policy_params['out'])
        actions = tf.matmul(h, W) + b
        return actions

    return tf.function(build_policy)
