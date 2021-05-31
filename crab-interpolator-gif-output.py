import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from celluloid import Camera

#get more datapoints and set the radius to be larger so it doesn't decay too small that NaNs show up and break it

# reshape data into images
numberofdatapoints = 50
dim_1 = 1
data_set = np.load('crabdataset.npy')
small_data_set = data_set[342:numberofdatapoints+342, :]
data = small_data_set.reshape(numberofdatapoints, 28, 28).astype('float32') / 255 #type necessary to go through autoencoder

# # Plot initial data
# fig, ax = plt.subplots(4, 4)
# for i in range(numberofdatapoints):
#     ax = plt.subplot(4, 4, i + 1)
#     plt.imshow(data[i])
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

# initial parameters and plot dimensions
initial_learningrate = 0.04
iterations = 5000
# initial_radius = max(dim_1, dim_2)/2
initial_radius = 45
time_constant = iterations / 4 * np.log10(initial_radius)


def compute_dist(data, weights):
    return tf.math.reduce_sum((data - weights) ** 2, axis=1, keepdims=True)


def decay_radius(initial_radius1, iteration_number, time):
    return initial_radius1 * np.exp(-iteration_number / time)


def decay_learningrate(learningrate1, iteration_number, total_iterations):
    return (learningrate1 * np.exp(-iteration_number / total_iterations))


def radius_influence(distance1, rad):
    return tf.math.exp(-distance1 / (2 * rad ** 2))


def point_dist(index_min, indices):
    return (indices - index_min) ** 2

#index array is a 1D
index_array = tf.range(numberofdatapoints)

#upload trained model
autoencoder = tf.keras.models.load_model('model_autoencoder')
#encode all images
encoded_img = autoencoder.encoder(data)# 10x64

# weights_array = tf.random.uniform(shape=[tf.shape(encoded_img)[1], dim_1, dim_2], minval=0, maxval=1, dtype=tf.float32)
# weights = tf.random.uniform(shape = [numberofdatapoints, 28, 28], minval=0, maxval=1, dtype=tf.float32)
# weights_array = autoencoder.encoder(weights)
weights_array = tf.random.uniform(shape=[tf.shape(encoded_img)[1]], minval=0, maxval=1, dtype=tf.float32)

# # Plot initial weights
# fig, ax = plt.subplots(4, 4)
# for i in range(numberofdatapoints):
#     ax = plt.subplot(4, 4, i + 1)
#     plt.imshow(weights[i])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

for iter in range(iterations):

    radius = decay_radius(initial_radius, iter, time_constant)
    learningrate = decay_learningrate(initial_learningrate, iter, iterations)

    #choose a random image and make a tensor of the image (for multiplication)
    rand_img_idx = tf.random.uniform(shape=[], minval=0, maxval=numberofdatapoints-1, dtype=tf.int64)
    random_datapoint = encoded_img[rand_img_idx, :]
    r0 = tf.expand_dims(random_datapoint, axis=0)
    random_data_array = tf.keras.backend.repeat_elements(r0, numberofdatapoints, axis=0)

    # #print random data array to check
    # print(tf.expand_dims(encoded_img[rand_img_idx], axis=0))
    # test = autoencoder.decoder(tf.expand_dims(encoded_img[rand_img_idx], axis=0))
    # plt.imshow(tf.transpose(test, perm=[1, 2, 0]))
    # plt.show()
    # print(random_data_array)

    #find min distance for BMU
    dist = compute_dist(random_data_array, weights_array) #shape: 1 x 10
    min_index_array = np.argmin(dist, axis=None)

    # find the distances between all data points and bmu to determine influence and change to weight tensor
    loc_dist = point_dist(min_index_array, index_array)
    influence0 = radius_influence(loc_dist, radius)
    influence1 = tf.expand_dims(influence0, axis=1)
    influence = tf.broadcast_to(influence1, [numberofdatapoints, tf.shape(encoded_img)[1]])

    # #weights array and random data array need to be cast as double tensors so they can multiply
    w_a = tf.cast(weights_array, dtype = tf.double)
    rda = tf.cast(random_data_array, dtype = tf.double)
    add_value = learningrate * influence * (rda - w_a)

    # use a boolean matrix to set the nearest neighbors that get updated, then update the weights tensor
    array_mask0 = loc_dist <= radius ** 2
    array_mask1 = tf.expand_dims(array_mask0, axis=1)
    array_mask = tf.broadcast_to(array_mask1, [numberofdatapoints, tf.shape(encoded_img)[1]])
    add1 = tf.cast(array_mask, dtype=tf.float64)*add_value
    # check = add1 + w_a
    # print('check', tf.math.reduce_sum(rda - check, axis=1))

    #put weights_array back as float32 to go through the for loop again
    weights_array = tf.cast(w_a + add1, dtype=tf.float32)
    # print(weights_array)

    #convert the updated encoded values back into images
    decoded_imgs = autoencoder.decoder(weights_array)

    # plt.imshow(updated_imgs[rand_img_idx])
    # plt.show()
    # test = autoencoder.decoder(encoded_img)
    #
    # plt.imshow(test[rand_img_idx])
    # plt.show()
    saveimgs = []

    if iter % 1000 == 0:
        print(iter)


fig = plt.figure()
camera = Camera(fig)

for i in range(numberofdatapoints-10):
    plt.imshow(decoded_imgs[i])
    plt.gray()
    camera.snap()

animation = camera.animate()
animation.save('crab_animation3-2.gif')
#get image for each drawing and use external software to save movie file