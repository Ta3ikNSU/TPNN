from pathlib import Path
from Utils import *
import matplotlib.pyplot as plt


class GAN:
    def __init__(self, numbers, epochs=100, batch_size=64, input_layer_size_g=100, hidden_layer_size_g=128,
                 hidden_layer_size_d=128, learning_rate=1e-3, decay_rate=1e-4, image_size=28):

        self.numbers = numbers
        self.epochs = epochs
        self.batch_size = batch_size
        self.nx_g = input_layer_size_g
        self.nh_g = hidden_layer_size_g
        self.nh_d = hidden_layer_size_d
        self.lr = learning_rate
        self.dr = decay_rate
        self.image_size = image_size


        self.image_dir = Path('../gan-results')
        self.filenames = []

        # Initialise weights

        # Generator
        self.weightsGenerator0 = np.random.randn(self.nx_g, self.nh_g) * np.sqrt(2. / self.nx_g)
        self.biasGenerator0 = np.zeros((1, self.nh_g))

        self.weightsGenerator1 = np.random.randn(self.nh_g, self.nh_g) * np.sqrt(2. / self.nh_g)
        self.biasGenerator1 = np.zeros((1, self.nh_g))

        self.weightsGenerator2 = np.random.randn(self.nh_g, self.image_size ** 2) * np.sqrt(2. / self.nh_g)
        self.biasGenerator2 = np.zeros((1, self.image_size ** 2))

        # Discriminator
        self.weightsDiscriminator0 = np.random.randn(self.image_size ** 2, self.nh_d) * np.sqrt(
            2. / self.image_size ** 2)
        self.biasDiscriminator0 = np.zeros((1, self.nh_d))

        self.weightsDiscriminator1 = np.random.randn(self.nh_d, 1) * np.sqrt(2. / self.nh_d)
        self.biasDiscriminator1 = np.zeros((1, 1))

    def forward_generator(self, z):

        self.z0_g = np.dot(z, self.weightsGenerator0) + self.biasGenerator0
        self.a0_g = lrelu(self.z0_g, alpha=0)
        self.z1_g = np.dot(self.a0_g, self.weightsGenerator1) + self.biasGenerator1
        self.a1_g = np.tanh(self.z1_g)
        self.z2_g = np.dot(self.a1_g, self.weightsGenerator2) + self.biasGenerator2
        self.a2_g = np.tanh(self.z2_g)
        return self.z2_g, self.a2_g

    def forward_discriminator(self, x):

        self.z0_d = np.dot(x, self.weightsDiscriminator0) + self.biasDiscriminator0
        self.a0_d = lrelu(self.z0_d)
        self.z1_d = np.dot(self.a0_d, self.weightsDiscriminator1) + self.biasDiscriminator1
        self.a1_d = sigmoid(self.z1_d)
        return self.z1_d, self.a1_d

    def backward_discriminator(self, x_real, z1_real, a1_real, x_fake, z1_fake, a1_fake):

        da1_real = -1. / (a1_real + 1e-8)

        dz1_real = da1_real * d_sigmoid(z1_real)
        derivativeWeightsRealImage1 = np.dot(self.a0_d.T, dz1_real)
        derivativeBiasRealImage1 = np.sum(dz1_real, axis=0, keepdims=True)

        da0_real = np.dot(dz1_real, self.weightsDiscriminator1.T)
        dz0_real = da0_real * d_lrelu(self.z0_d)
        derivativeWeightsRealImage0 = np.dot(x_real.T, dz0_real)
        derivativeBiasRealImage0 = np.sum(dz0_real, axis=0, keepdims=True)

        da1_fake = 1. / (1. - a1_fake + 1e-8)

        dz1_fake = da1_fake * d_sigmoid(z1_fake)
        derivativeWeightsFakeImage1 = np.dot(self.a0_d.T, dz1_fake)
        derivativeBiasFakeImage1 = np.sum(dz1_fake, axis=0, keepdims=True)

        da0_fake = np.dot(dz1_fake, self.weightsDiscriminator1.T)
        dz0_fake = da0_fake * d_lrelu(self.z0_d, alpha=0)
        derivativeWeightsFakeImage0 = np.dot(x_fake.T, dz0_fake)
        derivativeBiasFakeImage0 = np.sum(dz0_fake, axis=0, keepdims=True)

        derivativeWeights1 = derivativeWeightsRealImage1 + derivativeWeightsFakeImage1
        derivativeBias1 = derivativeBiasRealImage1 + derivativeBiasFakeImage1

        derivativeWeights0 = derivativeWeightsRealImage0 + derivativeWeightsFakeImage0
        derivativeBias0 = derivativeBiasRealImage0 + derivativeBiasFakeImage0

        # Update gradients using
        self.weightsDiscriminator0 -= self.lr * derivativeWeights0
        self.biasDiscriminator0 -= self.lr * derivativeBias0

        self.weightsDiscriminator1 -= self.lr * derivativeWeights1
        self.biasDiscriminator1 -= self.lr * derivativeBias1

    def backward_generator(self, z, z1_fake, a1_fake):
        da1_d = -1.0 / (a1_fake + 1e-8)

        dz1_d = da1_d * d_sigmoid(z1_fake)
        da0_d = np.dot(dz1_d, self.weightsDiscriminator1.T)
        dz0_d = da0_d * d_lrelu(self.z0_d)
        dx_d = np.dot(dz0_d, self.weightsDiscriminator0.T)

        dz2_g = dx_d * d_tan(self.z2_g)
        derivativeWeights2 = np.dot(self.a0_g.T, dz2_g)
        derivativeBias2 = np.sum(dz2_g, axis=0, keepdims=True)
        da1_g = np.dot(dz2_g, self.weightsGenerator2.T)
        dz1_g = da1_g * d_lrelu(self.z0_g, alpha=0)
        derivativeWeights1 = np.dot(self.a1_g.T, dz1_g)
        derivativeBias1 = np.sum(dz1_g, axis=0, keepdims=True)
        da0_g = np.dot(dz1_g, self.weightsGenerator1.T)
        dz0_g = da0_g * d_lrelu(self.z0_g, alpha=0)
        derivativeWeights0 = np.dot(z.T, dz0_g)
        derivativeBias0 = np.sum(dz0_g, axis=0, keepdims=True)

        # Update gradients
        self.weightsGenerator0 -= self.lr * derivativeWeights0
        self.biasGenerator0 -= self.lr * derivativeBias0

        self.weightsGenerator1 -= self.lr * derivativeWeights1
        self.biasGenerator1 -= self.lr * derivativeBias1

        self.weightsGenerator2 -= self.lr * derivativeWeights2
        self.biasGenerator2 -= self.lr * derivativeBias2

    def filling_data(self, x, y):
        x_train = []
        y_train = []

        for i in range(y.shape[0]):
            if y[i] in self.numbers:
                x_train.append(x[i])
                y_train.append(y[i])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        num_batches = x_train.shape[0] // self.batch_size
        x_train = x_train[: num_batches * self.batch_size]
        y_train = y_train[: num_batches * self.batch_size]

        # Flatten the images (_,28,28)->(_, 784)
        x_train = np.reshape(x_train, (x_train.shape[0], -1))

        # Normalise the data to the range [-1,1]
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        # Shuffle the data
        idx = np.random.permutation(len(x_train))
        x_train, y_train = x_train[idx], y_train[idx]
        return x_train, y_train, num_batches

    def train(self, x, y):
        discriminator_losses = []
        generator_losses = []

        x_train, _, num_batches = self.filling_data(x, y)

        for epoch in range(self.epochs):
            for i in range(num_batches):
                # init
                x_real = x_train[i * self.batch_size: (i + 1) * self.batch_size]
                z = np.random.normal(0, 1, size=[self.batch_size, self.nx_g])

                # Front propagation
                z1_g, x_fake = self.forward_generator(z)

                z1_d_real, a1_d_real = self.forward_discriminator(x_real)
                z1_d_fake, a1_d_fake = self.forward_discriminator(x_fake)

                discriminator_loss = np.mean(-np.log(a1_d_real) - np.log(1 - a1_d_fake))
                discriminator_losses.append(discriminator_loss)

                generator_loss = np.mean(-np.log(a1_d_fake))
                generator_losses.append(generator_loss)

                # Backprop
                self.backward_discriminator(x_real, z1_d_real, a1_d_real, x_fake, z1_d_fake, a1_d_fake)
                self.backward_generator(z, z1_d_fake, a1_d_fake)

            if epoch == 0 or epoch // 5 > 0 and epoch % 5 == 0:
                images = np.reshape(x_fake, (self.batch_size, self.image_size, self.image_size))
                plt.figure(figsize=(56, 56))
                plt.imshow(images[0] * 127.5 + 127.5, cmap='gray')
                plt.axis('off')
                current_epoch_filename = self.image_dir.joinpath(f"gan_epoch_{epoch}.png")
                self.filenames.append(current_epoch_filename)
                plt.savefig(current_epoch_filename)
                plt.close()

            self.lr = self.lr * (1.0 / (1.0 + self.dr * epoch))
