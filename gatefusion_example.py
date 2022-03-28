import tensorflow as tf
import time
from sklearn.metrics import roc_curve, auc  # curve曲线
from matplotlib import pyplot as plt
import os
import scipy.stats as stats
import numpy as np
import cv2
from scipy.misc import imresize
from PIL import Image


def get_inputOp(filename, batch_size, capacity):
    def read_and_decode(filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={"label": tf.FixedLenFeature([], tf.int64),
                                                     "image": tf.FixedLenFeature([], tf.string), })
        img = tf.decode_raw(features["image"], tf.int16)
        img = tf.reshape(img, [28 * 28 * 1])
        max = tf.to_float(tf.reduce_max(img))
        img = tf.cast(img, tf.float32) * (1.0 / max)
        label = tf.cast(features["label"], tf.int32)
        return img, label

    im, l = read_and_decode(filename)
    l = tf.one_hot(indices=tf.cast(l, tf.int32), depth=2)
    data, label = tf.train.batch([im, l], batch_size, capacity)
    return data, label


def _pairwise_distance(embeddings, label, miu, tao, batch_size):
    # 矩阵相乘,得到（batch_size, batch_size），因为计算欧式距离|a-b|^2 = a^2 -2ab + b^2,
    # 其中 ab 可以用矩阵乘表示
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
    # mask = tf.to_float(tf.equal(distances, 0.0))
    # distances = distances + mask * 1e-16
    # distances = tf.sqrt(distances)
    # distances = distances * (1.0 - mask)  # 0的部分仍然置为0
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


def correlation_distance(embeddings1, embeddings2, batch_size):
    def cd(embeddings):
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
        # dot_product对角线部分就是 每个embedding的平方
        square_norm = tf.diag_part(dot_product)
        # |a-b|^2 = a^2 - 2ab + b^2
        # tf.expand_dims(square_norm, axis=1)是（batch_size, 1）大小的矩阵，减去 （batch_size, batch_size）大小的矩阵，相当于每一列操作
        a2 = tf.expand_dims(square_norm, axis=1)
        b2 = tf.expand_dims(square_norm, axis=0)
        distances = a2 - 2.0 * dot_product + b2
        distances = tf.maximum(distances, 0.0)  # 小于0的距离置为0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)  # 0的部分仍然置为0
        return distances

    N_ = 1.0 / tf.to_float(batch_size)
    d1 = cd(embeddings1)
    d2 = cd(embeddings2)
    distances = tf.square(tf.subtract(d1, d2))
    distances = tf.reduce_sum(distances)
    distances = distances * N_
    return distances


def S(LA, LB, beta):
    tmp = tf.cond(tf.greater(LA, LB), lambda: tf.exp(beta * (LA - LB)) - 1, lambda: 0.0)
    return tmp


def get_inputOp(filename, batch_size, capacity):
    def read_and_decode(filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={"label": tf.FixedLenFeature([], tf.int64),
                                                     "image": tf.FixedLenFeature([], tf.string), })
        img = tf.decode_raw(features["image"], tf.int16)
        img = tf.reshape(img, [28 * 28 * 1])
        max = tf.to_float(tf.reduce_max(img))
        img = tf.cast(img, tf.float32) * (1.0 / max)
        label = tf.cast(features["label"], tf.int32)
        return img, label

    im, l = read_and_decode(filename)
    l = tf.one_hot(indices=tf.cast(l, tf.int32), depth=2)
    data, label = tf.train.batch([im, l], batch_size, capacity)
    return data, label


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def W_variable(shape):
    initial = tf.ones(shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


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

    sensitivity = staticity_T[positive_position] / (
            staticity_T[positive_position] + staticity_F[(positive_position + 1) % 2])
    specificity = staticity_T[negative_position] / (
            staticity_T[negative_position] + staticity_F[(negative_position + 1) % 2])
    return sensitivity, specificity


def plot(pic1, conv_output11, conv_grad11, conv_output12, conv_grad12, pic2, conv_output21, conv_grad21, conv_output22,
         conv_grad22, step, batch_size, img_path):
    def getcam(image1, output11, grads_val11, output12, grads_val12, image2, output21,
               grads_val21, output22, grads_val22):
        def grad(output, grads_val):
            weights = np.mean(grads_val, axis=(0, 1))  # alpha_k, [512]
            cam = np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]
            # Taking a weighted average
            for k, w in enumerate(weights):
                cam += w * output[:, :, k]
            cam = np.maximum(cam, 0)
            cam = cam / (np.max(cam) + 1e-16)  # scale 0 to 1.0
            # cam = imresize(cam, (28, 28))
            cam = np.array(Image.fromarray(cam).resize(size=(28, 28)))
            cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

            return cam_heatmap

        cam11 = grad(output11, grads_val11)
        cam12 = grad(output12, grads_val12)

        cam21 = grad(output21, grads_val21)
        cam22 = grad(output22, grads_val22)

        img1 = image1.astype(float)
        img1 -= np.min(img1)
        img1 /= img1.max()

        img2 = image2.astype(float)
        img2 -= np.min(img2)
        img2 /= img2.max()

        return img1, cam11, cam12, img2, cam21, cam22

        # return img, cam1

    for i in range(batch_size):  # batch number
        image1 = pic1[i, :, :, 0]
        image2 = pic2[i, :, :, 0]

        output11 = conv_output11[i]
        grads_val11 = conv_grad11[i]

        output12 = conv_output12[i]
        grads_val12 = conv_grad12[i]

        output21 = conv_output21[i]
        grads_val21 = conv_grad21[i]

        output22 = conv_output22[i]
        grads_val22 = conv_grad22[i]

        img1, cam11, cam12, img2, cam21, cam22 = getcam(image1, output11, grads_val11, output12,
                                                        grads_val12, image2, output21, grads_val21,
                                                        output22, grads_val22)

        # img1, cam1, cam2, cam3, cam4 = getcam(image1, output1, grads_val1)

        fig = plt.figure()
        ax = fig.add_subplot(251)
        plt.axis('off')
        plt.title('original1')
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(image1, cmap=plt.get_cmap('gray'))

        ax = fig.add_subplot(253)
        plt.axis('off')
        plt.title('conv11')
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(cam11)

        ax = fig.add_subplot(255)
        plt.axis('off')
        plt.title('conv12')
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(cam12)

        ax = fig.add_subplot(151)
        plt.axis('off')
        plt.title('original2')
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(image2, cmap=plt.get_cmap('gray'))

        ax = fig.add_subplot(153)
        plt.axis('off')
        plt.title('conv21')
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(cam21)

        ax = fig.add_subplot(155)
        plt.axis('off')
        plt.title('conv22', )
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(cam22)

        plt.savefig(img_path + '/{}.png'.format(str(step) + '_' + str(i), bbox_inches='tight'))


def Model(i, j, lr, ds, dr, t_step, batch_size):
    global_step = tf.Variable(0)
    keep_prob = tf.placeholder(tf.float32)

    x1 = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
    label1 = tf.placeholder(tf.float32, [None, 2])

    inputData_1 = tf.reshape(x1, [-1, 28, 28, 1])

    kernel_11 = weight_variable([3, 3, 1, 32])
    bias_11 = bias_variable([32])
    conv_11 = conv2d(inputData_1, kernel_11)
    conv_out_11 = tf.nn.relu(conv_11 + bias_11)
    pooling_out_11 = max_pool_2x2(conv_out_11)

    kernel_12 = weight_variable([3, 3, 32, 64])
    bias_12 = bias_variable([64])
    conv_12 = conv2d(pooling_out_11, kernel_12)
    conv_out_12 = tf.nn.relu(conv_12 + bias_12)
    pooling_out_12 = max_pool_2x2(conv_out_12)

    pooling_out_12 = tf.reshape(pooling_out_12, [-1, 7 * 7 * 64])

    w_fc_11 = weight_variable([7 * 7 * 64, 500])
    b_fc_11 = bias_variable([500])
    fc_out_11 = tf.nn.relu(tf.matmul(pooling_out_12, w_fc_11) + b_fc_11)
    drop11 = tf.nn.dropout(fc_out_11, keep_prob)

    w_fc_12 = weight_variable([500, 50])
    b_fc_12 = bias_variable([50])
    fc_out_12 = tf.nn.relu(tf.matmul(drop11, w_fc_12) + b_fc_12)

    ######################################   Coronal_Feature   ###################################

    x2 = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
    label2 = tf.placeholder(tf.float32, [None, 2])

    inputData_2 = tf.reshape(x2, [-1, 28, 28, 1])

    kernel_21 = weight_variable([3, 3, 1, 32])
    bias_21 = bias_variable([32])
    conv_21 = conv2d(inputData_2, kernel_21)
    conv_out_21 = tf.nn.relu(conv_21 + bias_21)

    pooling_out_21 = max_pool_2x2(conv_out_21)

    kernel_22 = weight_variable([3, 3, 32, 64])
    bias_22 = bias_variable([64])
    conv_22 = conv2d(pooling_out_21, kernel_22)
    conv_out_22 = tf.nn.relu(conv_22 + bias_22)
    pooling_out_22 = max_pool_2x2(conv_out_22)

    pooling_out_22 = tf.reshape(pooling_out_22, [-1, 7 * 7 * 64])

    w_fc_21 = weight_variable([7 * 7 * 64, 500])
    b_fc_21 = bias_variable([500])
    fc_out_21 = tf.nn.relu(tf.matmul(pooling_out_22, w_fc_21) + b_fc_21)
    drop21 = tf.nn.dropout(fc_out_21, keep_prob)

    w_fc_22 = weight_variable([500, 50])
    b_fc_22 = bias_variable([50])
    fc_out_22 = tf.nn.relu(tf.matmul(drop21, w_fc_22) + b_fc_22)

    ######################################   Sagittal_Feature   ###################################

    x3 = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
    label3 = tf.placeholder(tf.float32, [None, 2])

    inputData_3 = tf.reshape(x3, [-1, 28, 28, 1])

    kernel_31 = weight_variable([3, 3, 1, 32])
    bias_31 = bias_variable([32])
    conv_31 = conv2d(inputData_3, kernel_31)
    conv_out_31 = tf.nn.relu(conv_31 + bias_31)
    pooling_out_31 = max_pool_2x2(conv_out_31)

    kernel_32 = weight_variable([3, 3, 32, 64])
    bias_32 = bias_variable([64])
    conv_32 = conv2d(pooling_out_31, kernel_32)
    conv_out_32 = tf.nn.relu(conv_32 + bias_32)
    pooling_out_32 = max_pool_2x2(conv_out_32)

    pooling_out_32 = tf.reshape(pooling_out_32, [-1, 7 * 7 * 64])

    w_fc_31 = weight_variable([7 * 7 * 64, 500])
    b_fc_31 = bias_variable([500])
    fc_out_31 = tf.nn.relu(tf.matmul(pooling_out_32, w_fc_31) + b_fc_31)
    drop31 = tf.nn.dropout(fc_out_31, keep_prob)

    w_fc_32 = weight_variable([500, 50])
    b_fc_32 = bias_variable([50])
    fc_out_32 = tf.nn.relu(tf.matmul(drop31, w_fc_32) + b_fc_32)
    ######################################     Fusion      ####################################
    concat1 = tf.concat([fc_out_12, fc_out_22, fc_out_32], 1)

    w_fusion_11 = weight_variable([150, 50])
    b_fusion_11 = bias_variable([50])
    w_fusion_21 = weight_variable([150, 50])
    b_fusion_21 = bias_variable([50])
    w_fusion_31 = weight_variable([150, 50])
    b_fusion_31 = bias_variable([50])

    f11 = tf.matmul(concat1, w_fusion_11) + b_fusion_11
    f21 = tf.matmul(concat1, w_fusion_21) + b_fusion_21
    f31 = tf.matmul(concat1, w_fusion_31) + b_fusion_31

    feature_cat = tf.concat([f11, f21, f31], 1)
    w_fc_f1 = weight_variable([150, 50])
    b_fc_f1 = bias_variable([50])
    fc_out_f1 = tf.nn.relu(tf.matmul(feature_cat, w_fc_f1) + b_fc_f1)

    w_fc_f2 = weight_variable([50, 2])
    b_fc_f2 = bias_variable([2])
    mid = tf.matmul(fc_out_f1, w_fc_f2) + b_fc_f2
    prediction = tf.nn.softmax(mid)

    ################################################
    with tf.name_scope('loss'):

        loss_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=mid, labels=label1))
        tf.summary.scalar('loss_cross_entropy', loss_cross_entropy)

        total_loss = loss_cross_entropy
        tf.summary.scalar('total_loss', total_loss)

    learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps=ds, decay_rate=dr,
                                               staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

    with tf.name_scope('Accuracy'):
        output_position = tf.argmax(prediction, 1)
        label_position = tf.argmax(label1, 1)
        predict = tf.equal(output_position, label_position)
        Accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
        tf.summary.scalar('Accuracy', Accuracy)

    merge = tf.summary.merge_all()
    #####################################    Train    ##########################################

    sess.run(tf.global_variables_initializer())
    tf.reset_default_graph()

    board_path = '/home/public/june29/log/' + py_title
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    board_path = board_path + '/' + title
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    board_path = board_path + '/' + i
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    board_path = board_path + '/' + j
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    image_board_path = board_path + '/' + 'image'
    if (not (os.path.exists(image_board_path))):
        os.mkdir(image_board_path)

    test_board_path = board_path + '/' + 'test'
    if (not (os.path.exists(test_board_path))):
        os.mkdir(test_board_path)

    train_board_path = board_path + '/' + 'train'
    if (not (os.path.exists(train_board_path))):
        os.mkdir(train_board_path)

    test_writer = tf.summary.FileWriter(test_board_path + '/', tf.get_default_graph())
    train_writer = tf.summary.FileWriter(train_board_path + '/', tf.get_default_graph())

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    accmax = 0
    before = time.time()
    ########################################### ###
    for times in range(t_step):
        DMQ_Axial_dataTest_r, DMQ_Coronal_dataTest_r, DMQ_Sagittal_dataTest_r, \
        DMQ_Axial_labelTest_r, DMQ_Coronal_labelTest_r, DMQ_Sagittal_labelTest_r = sess.run(
            [DMQ_Axial_dataTest, DMQ_Coronal_dataTest, DMQ_Sagittal_dataTest, DMQ_Axial_labelTest,
             DMQ_Coronal_labelTest, DMQ_Sagittal_labelTest])
        DMQ_Axial_dataTrain_r, DMQ_Coronal_dataTrain_r, DMQ_Sagittal_dataTrain_r, \
        DMQ_Axial_labelTrain_r, DMQ_Coronal_labelTrain_r, DMQ_Sagittal_labelTrain_r = sess.run(
            [DMQ_Axial_dataTrain, DMQ_Coronal_dataTrain, DMQ_Sagittal_dataTrain,
             DMQ_Axial_labelTrain, DMQ_Coronal_labelTrain, DMQ_Sagittal_labelTrain])

        # ##########################  test  #######################
        #
        # if times % 10 == 0:
        #     img1r, att11r, grad11r, att12r, grad12r, att13r, grad13r, att14r, grad14r, att15r, grad15r, att16r, grad16r, \
        #     att17r, grad17r, att18r, grad18r, img2r, att21r, grad21r, att22r, grad22r, att23r, grad23r, att24r, grad24r, \
        #     att25r, grad25r, att26r, grad26r, att27r, grad27r, att28r, grad28r, img3r, att31r, grad31r, att32r, grad32r, \
        #     att33r, grad33r, att34r, grad34r, att35r, grad35r, att36r, grad36r, att37r, grad37r, att38r, grad38r, \
        #     summary, acc, output_position_r, label_position_r, predict_r, p, loss_cross_entropy_r, midr, mid1r, mid2r, mid3r = \
        #         sess.run([inputData_1, h_conv11, target_conv_layer_grad11, h_non_local_out11, target_conv_layer_grad12,
        #                   h_co_att_out12, target_conv_layer_grad13, h_co_att_out120, target_conv_layer_grad14, h_conv13,
        #                   target_conv_layer_grad15, h_non_local_out13, target_conv_layer_grad16, h_co_att_out14,
        #                   target_conv_layer_grad17, h_co_att_out141, target_conv_layer_grad18,
        #                   inputData_2, h_conv21, target_conv_layer_grad21, h_non_local_out21, target_conv_layer_grad22,
        #                   h_co_att_out22, target_conv_layer_grad23, h_co_att_out220, target_conv_layer_grad24, h_conv23,
        #                   target_conv_layer_grad25, h_non_local_out23, target_conv_layer_grad26, h_co_att_out24,
        #                   target_conv_layer_grad27, h_co_att_out241, target_conv_layer_grad28,
        #                   inputData_3, h_conv31, target_conv_layer_grad31, h_non_local_out31, target_conv_layer_grad32,
        #                   h_co_att_out32, target_conv_layer_grad33, h_co_att_out320, target_conv_layer_grad34, h_conv33,
        #                   target_conv_layer_grad35, h_non_local_out33, target_conv_layer_grad36, h_co_att_out34,
        #                   target_conv_layer_grad37, h_co_att_out341, target_conv_layer_grad38,
        #                   merge, Accuracy, output_position, label_position, predict, prediction, loss_cross_entropy,
        #                   mid, mid1, mid2, mid3],
        #                  feed_dict={x1: DMQ_Axial_dataTest_r, x2: DMQ_Coronal_dataTest_r,
        #                             x3: DMQ_Sagittal_dataTest_r,
        #                             label1: DMQ_Axial_labelTest_r, label2: DMQ_Coronal_labelTest_r,
        #                             label3: DMQ_Sagittal_labelTest_r, keep_prob: 1.0})
        ###########################  test  #######################
        if times % 10 == 0:
            summary, acc, output_position_r, label_position_r, predict_r, p, loss_cross_entropy_r = sess.run(
                [merge, Accuracy, output_position, label_position, predict, prediction, loss_cross_entropy],
                feed_dict={x1: DMQ_Axial_dataTest_r, x2: DMQ_Coronal_dataTest_r,
                           x3: DMQ_Sagittal_dataTest_r,
                           label1: DMQ_Axial_labelTest_r, label2: DMQ_Coronal_labelTest_r,
                           label3: DMQ_Sagittal_labelTest_r, keep_prob: 1.0})
            test_writer.add_summary(summary, times)
            sen, spe = Sensitivity_specificity(output_position_r, predict_r)
            fpr, tpr, thresholds = roc_curve(label_position_r, p[:, 1], drop_intermediate=False)
            AUC = auc(fpr, tpr)

            with open(board_path + '/' + 'result.txt', "a") as res_file:
                res_file.write(
                    str(times) + ',' + str(acc) + ',' + str(loss_cross_entropy_r) + ',' + str(sen) + ',' + str(
                        spe) + ',' + str(AUC) + ',' + str(fpr) + ',' + str(tpr) + ",,,,,,")

            ###################################### Grad-cam ################################
            if acc >= accmax:
                accmax = acc
                print(times,':',accmax)
        #         plot(img1r, att11r, grad11r, att12r, grad12r, att13r, grad13r, att14r, grad14r, att15r, grad15r,
        #              att16r, grad16r, att17r, grad17r, att18r, grad18r, times, batch_size, image_board_path1)
        #         plot(img2r, att21r, grad21r, att22r, grad22r, att23r, grad23r, att24r, grad24r, att25r, grad25r,
        #              att26r, grad26r, att27r, grad27r, att28r, grad28r, times, batch_size, image_board_path2)
        #         plot(img3r, att31r, grad31r, att32r, grad32r, att33r, grad33r, att34r, grad34r, att35r, grad35r,
        #              att36r, grad36r, att37r, grad37r, att38r, grad38r, times, batch_size, image_board_path3)
        #
        #         plot_only = 40  # 只画前500个点
        #         tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        #         # plot_only = 500  # 只画前500个点
        #         # 对中间层输出进行tsne降维
        #         low_dim_embs = tsne.fit_transform(midr[:plot_only, :])
        #         low_dim_embs1 = tsne.fit_transform(mid1r[:plot_only, :])
        #         low_dim_embs2 = tsne.fit_transform(mid2r[:plot_only, :])
        #         low_dim_embs3 = tsne.fit_transform(mid3r[:plot_only, :])
        #         # 对中间层输出进行tsne降维
        #         # 数据经过tsne以后是二维
        #         # 画图传递数据二维的，和真实类别
        #         labels = label_position_r[:plot_only]
        #         # plot_with_labels(low_dim_embs,labels,tsne_path,times)
        #         plot_o_l(low_dim_embs, labels, tsne_path, times)
        #         plot_o_l(low_dim_embs1, labels, tsne_path1, times)
        #         plot_o_l(low_dim_embs2, labels, tsne_path2, times)
        #         plot_o_l(low_dim_embs3, labels, tsne_path3, times)
        ###########################  train  #######################
        if times % 99 == 0:
            summary, _ = sess.run([merge, train_step],
                                  feed_dict={x1: DMQ_Axial_dataTrain_r, x2: DMQ_Coronal_dataTrain_r,
                                             x3: DMQ_Sagittal_dataTrain_r,
                                             label1: DMQ_Axial_labelTrain_r,
                                             label2: DMQ_Coronal_labelTrain_r,
                                             label3: DMQ_Sagittal_labelTrain_r, keep_prob: 0.5})
            train_writer.add_summary(summary, times)
        else:
            sess.run([train_step],
                     feed_dict={x1: DMQ_Axial_dataTrain_r, x2: DMQ_Coronal_dataTrain_r,
                                x3: DMQ_Sagittal_dataTrain_r,
                                label1: DMQ_Axial_labelTrain_r, label2: DMQ_Coronal_labelTrain_r,
                                label3: DMQ_Sagittal_labelTrain_r, keep_prob: 0.5})

    after = time.time()
    train_writer.close()
    test_writer.close()
    coord.request_stop()
    coord.join(thread)
    return acc, sen, spe, AUC, output_position_r, label_position_r, predict_r, p, after, before


py_title = 'concat'

lr = 1e-4
ds = 64
t_step = 10001
dr = 0.92
batch_size = 40

data_ = []
for i in range(1, 10):
    data_.append(str(i))

# data_ = ['85','9']

for k in range(1, 2):
    j = str(k)
    for i in data_:
        DMQ_Axial_dataTrain, DMQ_Axial_labelTrain = get_inputOp(
            "/home/public/june29/2D/DMQ/Axial/" + i + "/Train.tfrecords",
            batch_size, 1000)
        DMQ_Axial_dataTest, DMQ_Axial_labelTest = get_inputOp(
            "/home/public/june29/2D/DMQ/Axial/" + i + "/Test.tfrecords",
            batch_size, batch_size)
        DMQ_Coronal_dataTrain, DMQ_Coronal_labelTrain = get_inputOp(
            "/home/public/june29/2D/MMQ/Axial/" + i + "/Train.tfrecords",
            batch_size, 1000)
        DMQ_Coronal_dataTest, DMQ_Coronal_labelTest = get_inputOp(
            "/home/public/june29/2D/MMQ/Axial/" + i + "/Test.tfrecords",
            batch_size, batch_size)
        DMQ_Sagittal_dataTrain, DMQ_Sagittal_labelTrain = get_inputOp(
            "/home/public/june29/2D/PS/Axial/" + i + "/Train.tfrecords",
            batch_size, 1000)
        DMQ_Sagittal_dataTest, DMQ_Sagittal_labelTest = get_inputOp(
            "/home/public/june29/2D/PS/Axial/" + i + "/Test.tfrecords",
            batch_size, batch_size)

        sess = tf.InteractiveSession()

        title = str(lr) + '_' + str(ds) + '_' + str(dr)

        acc, sen, spe, AUC, output_position_r, label_position_r, predict_r, p, after, before \
            = Model(i, j, lr, ds, dr, t_step, batch_size)

        s = []
        for b in range(len(p)):
            s.append(p[b][0])
        r, pvalue = stats.pearsonr(s, label_position_r)
        total_time = (after - before) / 60

        print('train dada:  ' + i + ' round:  ' + j)
        print('Accuracy is: ' + str(acc) + "\n" + 'Sensitivity is: ' + str(sen) +
              "\n" + 'Specificity is: ' + str(spe))
        print('Auc : ' + str(AUC))
        print('pvalue: ' + str(pvalue))
        print('Total time is: ' + str(total_time) + ' minutes.' + "\n")

        sess.close()

# lr_ = [5e-5,1e-4,3e-4, 5e-4, 1e-3]
#
# for k in range(1,3):
#     j=str(k)
#     for i in data_:
#
#         t_step = 8001
#
#         for lr in lr_:
#             ds = 64
#             while ds <= 128:
#                 dr = 1.0
#
#                 while dr >= 0.95:
#
#                     DMQ_Axial_dataTrain, DMQ_Axial_labelTrain = get_inputOp(
#                         "/home/public/june29/2D/DMQ/Axial/" + i + "/Train.tfrecords",
#                         batch_size, 1000)
#                     DMQ_Axial_dataTest, DMQ_Axial_labelTest = get_inputOp(
#                         "/home/public/june29/2D/DMQ/Axial/" + i + "/Test.tfrecords",
#                         batch_size, batch_size)
#                     DMQ_Coronal_dataTrain, DMQ_Coronal_labelTrain = get_inputOp(
#                         "/home/public/june29/2D/DMQ/Coronal/" + i + "/Train.tfrecords",
#                         batch_size, 1000)
#                     DMQ_Coronal_dataTest, DMQ_Coronal_labelTest = get_inputOp(
#                         "/home/public/june29/2D/DMQ/Coronal/" + i + "/Test.tfrecords",
#                         batch_size, batch_size)
#                     DMQ_Sagittal_dataTrain, DMQ_Sagittal_labelTrain = get_inputOp(
#                         "/home/public/june29/2D/DMQ/Sagittal/" + i + "/Train.tfrecords",
#                         batch_size, 1000)
#                     DMQ_Sagittal_dataTest, DMQ_Sagittal_labelTest = get_inputOp(
#                         "/home/public/june29/2D/DMQ/Sagittal/" + i + "/Test.tfrecords",
#                         batch_size, batch_size)
#
#                     sess = tf.InteractiveSession()
#
#                     title = str(lr) + '_' + str(dr) + '_' + str(ds)
#
#                     acc, sen, spe, AUC, output_position_r, label_position_r, predict_r, p, after, before = Model(i,j, lr, ds, dr, t_step,batch_size)
#                     s = []
#                     for k in range(len(p)):
#                         s.append(p[k][0])
#                     r, pvalue = stats.pearsonr(s, label_position_r)
#                     sess.close()
#
#                     print(title)
#                     print('Accuracy is: ' + str(acc) + "\n" + 'Sensitivity is: ' + str(sen) + "\n" + 'Specificity is: ' + str(spe))
#                     print('Auc : ' + str(AUC))
#                     print('pvalue: ' + str(pvalue))
#                     print('Total time is: ' + str((after - before) / 60) + ' minutes.'+'\n')
#
#                     dr = dr - 0.02
#                 ds=ds+32
