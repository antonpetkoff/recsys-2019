# Lambda wrappers
def expand_dims_axis(axis):
  def expand_dims(x):
    return tf.expand_dims(x, axis=axis)
  return expand_dims

def reduce_sum_axis(axis):
  def reduce_sum(x):
    return tf.reduce_sum(x, axis)
  return reduce_sum

def model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys, see below
   params):  # Additional configuration
  
OneHotPlatform = layers.Lambda(lambda v: K.one_hot(v,
                                 num_classes=params['platforms_vocab']))
OneHotDevice = layers.Lambda(lambda v: K.one_hot(v,
                                 num_classes=params['devices_vocab']))
OneHotFilters = layers.Lambda(lambda v: K.one_hot(v,
                                 num_classes=params['filters_vocab']))
OneHotItems = layers.Lambda(lambda v: K.one_hot(v,
                                 num_classes=params['items_vocab']))
ReduceSum = layers.Lambda(reduce_sum_axis(1))
ReduceMean = layers.Lambda(lambda v: tf.reduce_mean(v, axis=1))

# Input layers
input_user = layers.Input(shape=(1,), dtype=tf.int32)
input_platform = layers.Input(shape=(1,), dtype=tf.int32)
input_device = layers.Input(shape=(1,), dtype=tf.int32)
input_filters = layers.Input(shape=(None,), dtype=tf.int32)
input_interacted_items = layers.Input(shape=(None,), dtype=tf.int32)
input_impression_items = layers.Input(shape=(None,), dtype=tf.int32)

users_embeddings = layers.Embedding(input_dim=params['users_vocab'],
                                    output_dim=params['users_embedding'])
items_embeddings = layers.Embedding(input_dim=params['items_vocab'],
                                    output_dim=params['items_embedding'])

user = layers.Flatten()(users_embeddings(input_user))
platform = layers.Flatten()(OneHotPlatform(input_platform))
device = layers.Flatten()(OneHotDevice(input_device))
# Summing them - taking into account all of the filters used.
filters = ReduceSum(OneHotFilters(input_filters))

interacted_items = ReduceMean(items_embeddings(input_interacted_items))
impression_items = ReduceMean(items_embeddings(input_impression_items))


# Dense layers
input_layer = layers.concatenate([user, platform, device, filters,
                                  interacted_items, impression_items],
                                 axis=1)
first_layer = layers.Dense(units=params['layer_1_units'],
                           activation=tf.keras.activations.relu)(input_layer)
second_layer = layers.Dense(units=params['layer_2_units'],
                            activation=tf.keras.activations.relu)(first_layer)
third_layer = layers.Dense(units=params['layer_3_units'],
                           activation=tf.keras.activations.relu)(second_layer)
last_layer = layers.Dense(units=params['items_vocab'],
                          activation=tf.keras.activations.softmax)(third_layer)

mask_impressions = ReduceSum(OneHotItems(input_impression_items))
predictions = layers.Multiply()([mask_impressions, last_layer])
predictions = layers.Softmax()(predictions)

inputs = [input_user, input_platform, input_device, input_filters,
          input_interacted_items, input_impression_items]

model = tf.keras.Model(inputs=inputs, outputs=last_layer)

# Set Optimizer
opt = RMSprop(lr=0.1, decay=1e-6)

model.compile(optimizer=opt,
              #loss=tf.keras.losses.CategoricalCrossentropy(),
              loss='sparse_categorical_crossentropy',
              # List of metrics to monitor
              metrics=[tf.keras.metrics.CategoricalAccuracy()],
             )


history = model.fit(x=dataset_train.make_one_shot_iterator(),
                    # y= target_items_batch,#input_onehot_labels_answer,
                    epochs=10,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=dataset_val.make_one_shot_iterator(),
                    validation_steps=validation_steps,
                    callbacks=[custom_metrics]
                   )