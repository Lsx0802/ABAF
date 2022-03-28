import tensorflow as tf


def W_variable(shape):
    initial = tf.ones(shape)
    return tf.Variable(initial)


def get_inputOp(filename, batch_size, capacity):
    def read_and_decode(filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={"label": tf.FixedLenFeature([], tf.int64),
                                                     "image": tf.FixedLenFeature([], tf.string), })
        img = tf.decode_raw(features["image"], tf.int16)
        img = tf.reshape(img, [16 * 16 * 16 * 1])
        max = tf.to_float(tf.reduce_max(img))
        img = tf.cast(img, tf.float32) * (1.0 / max)
        label = tf.cast(features["label"], tf.int32)
        return img, label

    im, l = read_and_decode(filename)
    l = tf.one_hot(indices=tf.cast(l, tf.int32), depth=2)
    data, label = tf.train.batch([im, l], batch_size=batch_size, capacity=capacity)
    return data, label


def S(LA, LB, beta):
    tmp = tf.cond(tf.greater(LB, LA), lambda: tf.to_float(tf.exp(beta * (LB - LA)) - 1.0),
                  lambda: 0.0)
    return tmp


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def Sensitivity_specificity(model_output, equal):
    positive_position = 1
    negative_position = 0
    staticity_T = [0, 0]
    staticity_F = [0, 0]

    for i in range(len(equal)):
        if equal[i] == True:
            staticity_T[model_output[i]] += 1
        else:
            staticity_F[model_output[i]] += 1

    sensitivity = staticity_T[positive_position] + 1e-6 / (
            staticity_T[positive_position] + staticity_F[negative_position] + 1e-6)
    specificity = staticity_T[negative_position] + 1e-6 / (
            staticity_T[negative_position] + staticity_F[positive_position] + 1e-6)
    return sensitivity, specificity


def pairwise_distance(embeddings, label, miu, tao, batch_size):
    N_ = 1.0 / tf.to_float(tf.pow(batch_size, 2))
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    # dot_product对角线部分就是 每个embedding的平方
    square_norm = tf.diag_part(dot_product)
    # |a-b|^2 = a^2 - 2ab + b^2
    # tf.expand_dims(square_norm, axis=1)是（batch_size, 1）大小的矩阵，减去 （batch_size, batch_size）大小的矩阵，相当于每一列操作
    a2 = tf.expand_dims(square_norm, axis=1)
    b2 = tf.expand_dims(square_norm, axis=0)
    distances = a2 - 2.0 * dot_product + b2
    distances = tf.maximum(distances, 0.0)  # 小于0的距离置为0
    distances = miu - distances
    mask_y_1 = tf.to_float(tf.argmax(label, axis=1))
    mask_y = tf.to_float(tf.equal(mask_y_1, 0.0)) * -1.0
    yij = tf.expand_dims(mask_y_1 + mask_y, axis=0)
    yij = tf.matmul(tf.transpose(yij), yij)
    distances = tf.multiply(yij, distances)
    distances = tao - distances
    distances = tf.maximum(distances, 0.0)
    distances = tf.reduce_sum(distances)
    distances = distances * N_
    return distances


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2_3D(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='SAME')


def ewa(now, new, beta=0.9):  # 指数加权平均
    return now * beta + new * (1 - beta)


def tsne_embedding(embeddings, reverse_dictionary, plot_only=40,
                   second=5, saveable=False, name='tsne', fig_idx=9862):
    """Visualize the embeddings by using t-SNE.

    Parameters
    ----------
    embeddings : a matrix
        The images.
    reverse_dictionary : a dictionary
        id_to_word, mapping id to unique word.
    plot_only : int
        The number of examples to plot, choice the most common words.
    second : int
        The display second(s) for the image(s), if saveable is False.
    saveable : boolean
        Save or plot the figure.
    name : a string
        A name to save the image, if saveable is True.
    fig_idx : int
        matplotlib figure index.
            Examples
    --------
    >>> final_embeddings = normalized_embeddings.eval()
    >>> tsne_embedding(final_embeddings, labels, reverse_dictionary,
    ...                   plot_only=500, second=5, saveable=False, name='tsne')
    """

    def plot_with_labels(low_dim_embs, labels, figsize=(8, 8), second=5,
                         saveable=True, name='tsne', fig_idx=9862):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        if saveable is False:
            plt.ion()
            plt.figure(fig_idx)
        plt.figure(figsize=figsize)  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        if saveable:
            plt.savefig(name + '.png', format='png')
        else:
            plt.draw()
            plt.pause(second)

    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        from six.moves import xrange

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels, second=second, saveable=saveable, \
                         name=name, fig_idx=fig_idx)
    except ImportError:
        print("Please install sklearn and matplotlib to visualize embeddings.")
