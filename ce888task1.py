## IMPORT NEEDED PACKAGES
from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.datasets import cifar10

# WARNING: BECAUSE OF HUGE AMOUNT OF DATA THE EXECUTION OF THE CODE MAY BE A LITTLE BIT SLOW


# READ THE MNIST DATASET AND MANIPULATE THE PIXELS OF x_train and x_test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#  GAIN A FIRST INSIGHT OF HOW THE X_TRAIN DATA LOOKS
# print("This is x_train type", type(x_train))
# print("This is x_train dimensions", x_train.shape[0])
# print("This is the shape of every 'matrix' it has as values", x_train[0].shape)

#  GAIN A FIRST INSIGHT OF HOW THE X_TEST DATA LOOKS
# print("This is x_test type", type(x_test))
# print("This is x_test dimensions", x_test.shape[0])
# print("This is the shape of every 'matrix' it has as values", x_test[0].shape)


# STARTING WITH THE X_TRAIN
mnist_train_shuffled1 = np.empty([x_train.shape[0], x_train.shape[1], x_train.shape[2]])
mnist_train_shuffled2 = np.empty([x_train.shape[0], x_train.shape[1], x_train.shape[2]])
mnist_train_shuffled3 = np.empty([x_train.shape[0], x_train.shape[1], x_train.shape[2]])

# OBTAIN THE FIRST TASK OF THE TRAIN SET BY RESHUFFLING
for i in range(0, x_train.shape[0]):
    temp_shuffled = np.empty([x_train.shape[1], x_train.shape[2]])
    for j in range(0, x_train.shape[1]):
        np.random.shuffle(x_train[i][j])
        temp_shuffled[j] = x_train[i][j]
    mnist_train_shuffled1[i] = temp_shuffled

# OBTAIN THE SECOND TASK OF THE TRAIN SET BY RESHUFFLING
for i in range(0, x_train.shape[0]):
    temp_shuffled = np.empty([x_train.shape[1], x_train.shape[2]])
    for j in range(0, x_train.shape[1]):
        np.random.shuffle(x_train[i][j])
        temp_shuffled[j] = x_train[i][j]
    mnist_train_shuffled2[i] = temp_shuffled

# OBTAIN THE THIRD TASK OF THE TRAIN SET BY RESHUFFLING
for i in range(0, x_train.shape[0]):
    temp_shuffled = np.empty([x_train.shape[1], x_train.shape[2]])
    for j in range(0, x_train.shape[1]):
        np.random.shuffle(x_train[i][j])
        temp_shuffled[j] = x_train[i][j]
    mnist_train_shuffled3[i] = temp_shuffled

# CONTINUE WITH THE X_TEST

mnist_test_shuffled1 = np.empty([x_test.shape[0], x_test.shape[1], x_test.shape[2]])
mnist_test_shuffled2 = np.empty([x_test.shape[0], x_test.shape[1], x_test.shape[2]])
mnist_test_shuffled3 = np.empty([x_test.shape[0], x_test.shape[1], x_test.shape[2]])

# OBTAIN THE FIRST TASK OF THE TEST SET BY RESHUFFLING
for i in range(0, x_test.shape[0]):
    temp_shuffled = np.empty([x_test.shape[1], x_test.shape[2]])
    for j in range(0, x_test.shape[1]):
        np.random.shuffle(x_test[i][j])
        temp_shuffled[j] = x_test[i][j]
    mnist_test_shuffled1[i] = temp_shuffled

# OBTAIN THE SECOND TASK OF THE TEST SET BY RESHUFFLING
for i in range(0, x_test.shape[0]):
    temp_shuffled = np.empty([x_test.shape[1], x_test.shape[2]])
    for j in range(0, x_test.shape[1]):
        np.random.shuffle(x_test[i][j])
        temp_shuffled[j] = x_test[i][j]
    mnist_test_shuffled2[i] = temp_shuffled

# OBTAIN THE THIRD TASK OF THE TEST SET BY RESHUFFLING
for i in range(0, x_test.shape[0]):
    temp_shuffled = np.empty([x_test.shape[1], x_test.shape[2]])
    for j in range(0, x_test.shape[1]):
        np.random.shuffle(x_test[i][j])
        temp_shuffled[j] = x_test[i][j]
    mnist_test_shuffled3[i] = temp_shuffled

# # CODE FOR RANDOM SANITY CHECK
# # CHECK IF THE RESULTED PERMUTTED SET OF IMAGES OF THE X_TRAIN IS EQUAL TO THE STARTING ONE
# if(np.array_equal(mnist_train_shuffled1, x_train) == True):
#     print("Your set of images of the x_train is the same,you should reshuffle")
# else:
#     print("Your shuffled set of images of the x_train is different from the your starting one. You can continue.")
#
# ## CHECK IF THE RESULTED PERMUTTED SET OF IMAGES OF THE X_TEST IS EQUAL TO THE STARTING ONE
# if(np.array_equal(mnist_test_shuffled1, x_test) == True):
#     print("Your set of images of the x_test is the same,you should reshuffle")
# else:
#     print("Your shuffled set of images of the x_test is different from the your starting one. You can continue.")


# READ THE CIFAR10 DATASET AND MANIPULATE THE PIXELS OF THE x_trainCifar and x_testCifar
(x_trainCifar, y_trainCifar), (x_testCifar, y_testCifar) = cifar10.load_data()

# # GAIN A FIRST INSIGHT OF HOW THE X_TRAIN_CIFAR10 DATA LOOKS
# print("This is x_trainCifar type", type(x_trainCifar))
# print("This is x_trainCifar dimensions", x_testCifar.shape)

x_trainCifar_shuffled1 = np.empty([int(x_trainCifar.shape[0]/100), x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])
x_trainCifar_shuffled2 = np.empty([int(x_trainCifar.shape[0]/100), x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])
x_trainCifar_shuffled3 = np.empty([int(x_trainCifar.shape[0]/100), x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])

# OBTAIN THE FIRST TASK OF THE CIFAR10 TRAIN SET BY RESHUFFLING
for i in range(0, int(x_trainCifar.shape[0]/100)):
    temp1 = np.empty([x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])
    for j in range(0, x_trainCifar.shape[1]):
        temp2 = np.empty([x_trainCifar.shape[2], x_trainCifar.shape[3]])
        for z in range(0, x_trainCifar.shape[2]):
            np.random.shuffle(x_trainCifar[i][j][z])
            temp2[z] = x_trainCifar[i][j][z]
        temp1[j] = temp2
    x_trainCifar_shuffled1[i] = temp1

# OBTAIN THE SECOND TASK OF THE CIFAR10 TRAIN SET BY RESHUFFLING
for i in range(0, int(x_trainCifar.shape[0]/100)):
    temp1 = np.empty([x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])
    for j in range(0, x_trainCifar.shape[1]):
        temp2 = np.empty([x_trainCifar.shape[2], x_trainCifar.shape[3]])
        for z in range(0, x_trainCifar.shape[2]):
            np.random.shuffle(x_trainCifar[i][j][z])
            temp2[z] = x_trainCifar[i][j][z]
        temp1[j] = temp2
    x_trainCifar_shuffled2[i] = temp1

# OBTAIN THE THIRD TASK OF THE CIFAR10 TRAIN SET BY RESHUFFLING
for i in range(0, int(x_trainCifar.shape[0]/100)):
    temp1 = np.empty([x_trainCifar.shape[1], x_trainCifar.shape[2], x_trainCifar.shape[3]])
    for j in range(0, x_trainCifar.shape[1]):
        temp2 = np.empty([x_trainCifar.shape[2], x_trainCifar.shape[3]])
        for z in range(0, x_trainCifar.shape[2]):
            np.random.shuffle(x_trainCifar[i][j][z])
            temp2[z] = x_trainCifar[i][j][z]
        temp1[j] = temp2
    x_trainCifar_shuffled3[i] = temp1

# CONTINUE WITH THE X_TEST

x_testCifar_shuffled1 = np.empty([int(x_testCifar.shape[0]/100), x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]])
x_testCifar_shuffled2 = np.empty([int(x_testCifar.shape[0]/100), x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]])
x_testCifar_shuffled3 = np.empty([int(x_testCifar.shape[0]/100), x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]])

# OBTAIN THE FIRST TASK OF THE CIFAR10 TEST SET BY RESHUFFLING
for i in range(0, int(x_testCifar.shape[0]/100)):
    temp1 = np.empty(([x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]]))
    for j in range(0, x_testCifar.shape[1]):
        temp2 = np.empty([x_testCifar.shape[2], x_testCifar.shape[3]])
        for z in range(0, x_testCifar.shape[2]):
            np.random.shuffle(x_testCifar[i][j][z])
            temp2[z] = x_testCifar[i][j][z]
        temp1[j] = temp2
    x_testCifar_shuffled1[i] = temp1

# OBTAIN THE SECOND TASK OF THE CIFAR10 TEST SET BY RESHUFFLING
for i in range(0, int(x_testCifar.shape[0]/100)):
    temp1 = np.empty(([x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]]))
    for j in range(0, x_testCifar.shape[1]):
        temp2 = np.empty([x_testCifar.shape[2], x_testCifar.shape[3]])
        for z in range(0, x_testCifar.shape[2]):
            np.random.shuffle(x_testCifar[i][j][z])
            temp2[z] = x_testCifar[i][j][z]
        temp1[j] = temp2
    x_testCifar_shuffled2[i] = temp1

# OBTAIN THE THIRD TASK OF THE CIFAR10 TEST SET BY RESHUFFLING
for i in range(0, int(x_testCifar.shape[0]/100)):
    temp1 = np.empty(([x_testCifar.shape[1], x_testCifar.shape[2], x_testCifar.shape[3]]))
    for j in range(0, x_testCifar.shape[1]):
        temp2 = np.empty([x_testCifar.shape[2], x_testCifar.shape[3]])
        for z in range(0, x_testCifar.shape[2]):
            np.random.shuffle(x_testCifar[i][j][z])
            temp2[z] = x_testCifar[i][j][z]
        temp1[j] = temp2
    x_testCifar_shuffled3[i] = temp1

print("Data was read successfully")





