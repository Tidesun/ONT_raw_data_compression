import tensorflow as tf
# for loading dataset 
buffer_size = 1024 * 200
batch_size = 1024
def parse_x(example_proto):
    feature_description = {
        'x': tf.io.FixedLenFeature([512], dtype=tf.float32),
        'y': tf.io.VarLenFeature(dtype=tf.int64)
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    x = tf.expand_dims(parsed_example['x'],1)
    return x
def parse_y(example_proto):
    feature_description = {
        'x': tf.io.FixedLenFeature([512], dtype=tf.float32),
        'y': tf.io.VarLenFeature(dtype=tf.int64)
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    y = tf.cast(tf.sparse.to_dense(parsed_example['y']), dtype=tf.int32)
    return y
def make_training_dataset(filenames):
    tf_rec_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size)
    all_ds = []
    for dataset_type in ['train','validation']:
        if dataset_type == 'train':
            ds = tf_rec_dataset.skip(batch_size*100)
        else:
            ds = tf_rec_dataset.take(batch_size*100)
        # Map the parsing function to the dataset
        x_dataset = ds.map(parse_x, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=batch_size)
        y_dataset = ds.map(parse_y, num_parallel_calls=tf.data.AUTOTUNE).ragged_batch(batch_size=batch_size).map(lambda y:y.to_sparse(), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        # Apply transformations
        if dataset_type == 'train':
            dataset = dataset.repeat(num_epochs)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        all_ds.append(dataset)
    
    return all_ds
def make_test_dataset(filenames):
    tf_rec_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size)
    # Map the parsing function to the dataset
    x_dataset = tf_rec_dataset.map(parse_x, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=batch_size)
    y_dataset = tf_rec_dataset.map(parse_y, num_parallel_calls=tf.data.AUTOTUNE).ragged_batch(batch_size=batch_size).map(lambda y:y.to_sparse(), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
    # Apply transformations
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset        
train_filenames = tf.data.Dataset.list_files('training.tfrecord')
test_filenames = tf.data.Dataset.list_files('test.tfrecord')

training_validation_dataset = make_training_dataset(train_filenames)
test_dataset = make_dataset(test_filenames)
[training_dataset,validation_dataset] = training_validation_dataset