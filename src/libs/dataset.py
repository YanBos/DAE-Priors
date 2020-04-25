import tensorflow as tf
import utils.utils as utils

guidance = False

@tf.function
def parse(example_proto):

    example = tf.io.parse_single_example(example_proto, utils.FEATURES)

    img = tf.io.decode_raw(example['disp'], out_type=tf.float32)
    img = tf.image.pad_to_bounding_box(tf.reshape(img, [example['height'], example['width'], 1]), 0, 0, 512,640)
    guide = tf.io.decode_raw(example['guidance'], out_type=tf.float32)
    guide = tf.image.pad_to_bounding_box(tf.reshape(guide, [example['height'], example['width'], 3]), 0, 0, 512,640)

    return {"img":img, "guidance":guide, "height":example['height'], "width":example['width']}

@tf.function

def crop_sparse(data):
    offset = lambda size, psz: tf.random.uniform(shape=[], minval=0, maxval=tf.cast(size, dtype=tf.int32)-psz, dtype=tf.int32)
    def func(data):
        x = offset(data[3], utils.PSZ)
        y = offset(data[2], utils.PSZ)
        img = tf.image.crop_to_bounding_box(data[0], offset_height=y, offset_width=x, target_height=utils.PSZ, target_width=utils.PSZ)
        guide = tf.image.crop_to_bounding_box(data[1], offset_height=y, offset_width=x, target_height=utils.PSZ, target_width=utils.PSZ)
        return img, guide
    data = data['img'], data['guidance'], data['height'], data['width']
    img, guide = func(data)
    level = tf.random.uniform(shape=[], minval=utils.LNL,
                      maxval=utils.UNL)
    sparse = tf.reshape(tf.random.categorical(tf.math.log([[0.5,0.5]]), tf.cast(utils.PSZ**2, dtype=tf.int32)), (utils.PSZ, utils.PSZ, 1))
    img = tf.cast(sparse, dtype=tf.float32)*img
    return {"input_1":img, "input_2":guide}, img

@tf.function
def crop(data):
    global guidance
    offset = lambda size, psz: tf.random.uniform(shape=[], minval=0, maxval=tf.cast(size, dtype=tf.int32)-psz, dtype=tf.int32)
    @tf.function
    def func(data):
        x = offset(data[3], utils.PSZ)
        y = offset(data[2], utils.PSZ)
        img = tf.image.crop_to_bounding_box(data[0], offset_height=y, offset_width=x, target_height=utils.PSZ, target_width=utils.PSZ)
        guide = tf.image.crop_to_bounding_box(data[1], offset_height=y, offset_width=x, target_height=utils.PSZ, target_width=utils.PSZ)
        return img, guide, data[2], data[3]
    data = data['img'], data['guidance'], data['height'], data['width']
    img, guide, _, _ = tf.map_fn(func, data)
    img_noisy = img + tf.random.normal(shape=(utils.BSZ, utils.PSZ, utils.PSZ, 1), stddev=(tf.reduce_max(img) - tf.reduce_min(img))*tf.random.uniform(shape=(), minval=utils.LNL, maxval=utils.UNL))
    out = {"input_1":img_noisy, "input_2":guide} if guidance else img_noisy
    return out, img

@tf.function
def crop_val(data):
    global guidance
    offset = lambda size, psz: tf.random.uniform(shape=[], minval=0, maxval=tf.cast(size, dtype=tf.int32)-psz, dtype=tf.int32)
    @tf.function
    def func(data):
        x = offset(data[3], utils.PSZ)
        y = offset(data[2], utils.PSZ)
        img = tf.image.crop_to_bounding_box(data[0], offset_height=y, offset_width=x, target_height=utils.PSZ, target_width=utils.PSZ)
        guide = tf.image.crop_to_bounding_box(data[1], offset_height=y, offset_width=x, target_height=utils.PSZ, target_width=utils.PSZ)
        return img, guide, data[2], data[3]
    data = data['img'], data['guidance'], data['height'], data['width']
    img, guide, _, _ = tf.map_fn(func, data)
    img_noisy = img + tf.random.normal(shape=(utils.BSZ, utils.PSZ, utils.PSZ, 1), stddev=(tf.reduce_max(img) - tf.reduce_min(img))*tf.random.uniform(shape=(), minval=utils.LNL+(utils.UNL - utils.LNL)/2, maxval=utils.LNL+(utils.UNL - utils.LNL)/2))
    out = {"input_1":img_noisy, "input_2":guide} if guidance else img_noisy
    return out, img

def prepare_dataset(gd):
    global guidance
    guidance = gd

    dataset = tf.data.TFRecordDataset(utils.TDS)
    dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_cb = dataset.repeat().batch(utils.BSZ).map(crop_val)
    dataset = dataset.shuffle(utils.CDS)
    ds = dataset.take(utils.CDS - utils.CDSS)
    dsv = dataset.skip(utils.CDS - utils.CDSS).take(utils.CDSS)
    ds = ds.cache()
    dsv = dsv.take(utils.BSZ*utils.BVS).batch(utils.BSZ).map(crop)
    dsv = dsv.cache()
    for _ in ds:
        pass
    for _ in dsv:
        pass
    dsv = dsv.repeat()
    ds = ds.shuffle(utils.CDS)
    ds = ds.repeat()
    ds = ds.batch(utils.BSZ, drop_remainder=True)
    ds = ds.map(crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds, dataset_cb, dsv
