import tensorflow as tf
from model import *
import scipy.stats as stats

py_title = 'ABAF'

lr = 1e-4
ds = 64
t_step = 10001
dr = 0.96
batch_size = 40

data_ = []
for _ in range(1, 4):
    data_.append(str(_))

for k in range(1, 6):
    j = str(k)
    for i in data_:

        AP_dataTrain, AP_labelTrain = get_inputOp(
            "/home/public/ABAF/3D/AP/" + i + "/Train.tfrecords", batch_size,
            1000)
        AP_dataTest, AP_labelTest = get_inputOp(
            "/home/public/ABAF/3D/AP/" + i + "/Train.tfrecords", batch_size,
            batch_size)
        PVP_dataTrain, PVP_labelTrain = get_inputOp(
            "/home/public/ABAF/3D/PVP/" + i + "/Train.tfrecords", batch_size,
            1000)
        PVP_dataTest, PVP_labelTest = get_inputOp(
            "/home/public/ABAF/3D/PVP/" + i + "/Train.tfrecords", batch_size,
            batch_size)
        PC_dataTrain, PC_labelTrain = get_inputOp(
            "/home/public/ABAF/3D/PC/" + i + "/Train.tfrecords", batch_size,
            1000)
        PC_dataTest, PC_labelTest = get_inputOp(
            "/home/public/ABAF/3D/PC/" + i + "/Train.tfrecords", batch_size,
            batch_size)

        sess = tf.InteractiveSession()

        title = str(lr) + '_' + str(ds) + '_' + str(dr)

        acc, sen, spe, AUC, output_position_r, label_position_r, predict_r, p, after, before \
            = Model(i, j, lr, ds, dr, t_step, batch_size, py_title, title, sess, AP_dataTrain,
                    AP_labelTrain, AP_dataTest, AP_labelTest,PVP_dataTrain, PVP_labelTrain,
                    PVP_dataTest, PVP_labelTest,PC_dataTrain, PC_labelTrain,PC_dataTest, PC_labelTest)

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

        # sess.close()
