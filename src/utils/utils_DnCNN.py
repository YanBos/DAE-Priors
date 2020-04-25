import tensorflow as tf

# constant values

TAR = 4.222982524343324
VAR = 4.1644589233398435
TN = 5630
VN = 350

TDS = "/mnt/data/lightfield/train_small.tfrecords"
VDS = "/mnt/data/lightfield/val_small.tfrecords"
SD = "/mnt/data/models/SC"
TBP = "/mnt/tf-logs"
GPU = 0

# parameters for the network

EPOCHS = 1000
FILTERS = 64
BSZ = 8
NSZ = 20
KS = 3
ACTIVATION = 'tanh'
PADDING = 'same'
LR = 1e-4
PSZ = 64

LNL = 0.0
UNL = 0.2

# save string
SAVE = ""
FW = None
# dataset dict

FEATURES = {'width': tf.io.FixedLenFeature((), tf.int64),
            'height': tf.io.FixedLenFeature((), tf.int64),
            "image_raw": tf.io.FixedLenFeature((), tf.string),
            "guidance_raw": tf.io.FixedLenFeature((), tf.string)}
FEATURES_ALT = {
            "noisy": tf.io.FixedLenFeature((), tf.string),
            "original": tf.io.FixedLenFeature((), tf.string)}

TL = tf.keras.metrics.Mean('loss')
VL = tf.keras.metrics.Mean('val_loss')
# callbacks

ES = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50, restore_best_weights=True)
TB = None
DC = None
MC = None
FP = ""
