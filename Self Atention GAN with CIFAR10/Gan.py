import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import tensorflow_docs.vis.embed as embed
import imageio.v2 as imageio
import glob


# Self-Attention katmanı tanımı
class SelfAttention(layers.Layer):
    def __init__(self, channels, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        # Aşağıdaki katmanlar, self-attention mekanizmasının farklı bileşenlerini temsil eder
        self.theta = layers.Conv2D(channels // 8, kernel_size=1, padding='same', use_bias=False)
        self.phi = layers.Conv2D(channels // 8, kernel_size=1, padding='same', use_bias=False)
        self.g = layers.Conv2D(channels // 2, kernel_size=1, padding='same', use_bias=False)
        self.o = layers.Conv2D(channels, kernel_size=1, padding='same', use_bias=False)
        self.gamma = self.add_weight(name='gamma', shape=[], initializer='zeros')

    # Self-Attention mekanizmasının uygulanması
    def call(self, inputs, **kwargs):
        batch_size, h, w, c = inputs.shape
        batch_size = batch_size if batch_size is not None else -1

        # Giriş tensörünün farklı temsillerini hesapla
        theta = self.theta(inputs)
        theta = tf.reshape(theta, (batch_size, h * w, c // 8))

        phi = self.phi(inputs)
        phi = tf.reshape(phi, (batch_size, h * w, c // 8))
        phi = tf.transpose(phi, perm=[0, 2, 1])

        g = self.g(inputs)
        g = tf.reshape(g, (batch_size, h * w, c // 2))

        # Attention skorlarını hesapla
        attention = tf.nn.softmax(tf.matmul(theta, phi), axis=-1)
        # Skorlarla g temsilini ağırlıklı olarak topla
        out = tf.matmul(attention, g)
        out = tf.reshape(out, (batch_size, h, w, c // 2))
        out = self.o(out)

        # Girişle ağırlıklı çıktıyı toplayarak sonuç tensörünü oluştur
        return inputs + self.gamma * out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'channels': self.channels,
        })
        return config


# Üreteç (generator) modelini tanımla
def define_generator(latent_dim):
    model = Sequential([
        # Giriş boyutunu (latent_dim) 4x4x256 tensör boyutuna genişlet
        layers.Dense(256 * 4 * 4, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((4, 4, 256)),
        # Transposed convolution katmanları ile görüntü boyutunu artır
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        # Self-attention mekanizmasını ekleyin
        SelfAttention(channels=128),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        # Çıkış katmanı, 3 kanallı (RGB) görüntü üretir
        layers.Conv2D(3, (3, 3), activation='tanh', padding='same')
    ])
    return model

    # Ayırt edici (discriminator) modelini tanımla


def define_discriminator():
    model = Sequential([
        # Giriş katmanı, 32x32x3 boyutlu görüntü alır
        layers.Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)),
        layers.LeakyReLU(alpha=0.2),
        # Convolution katmanları ile öznitelik haritalarını küçült
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        # Self-attention mekanizmasını ekleyin
        SelfAttention(channels=256),
        layers.LeakyReLU(alpha=0.2),
        # Son çıktıyı düzleştir ve dropout uygula
        layers.Flatten(),
        layers.Dropout(0.4),
        # Tek nöronlu çıktı katmanı
        layers.Dense(1)
    ])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=opt, metrics=['accuracy'])
    return model

    # Üreteç ve ayırt edici modellerini birleştirerek GAN modelini tanımla


def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential([
        g_model,
        d_model
    ])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=opt)
    return model


# Veri önişleme işlemi
def preprocess_data(data):
    return (data - 127.5) / 127.5


# Görüntüleri kaydetme işlemi
def save_plot(examples, epoch, n=7):
    examples = (examples + 1) / 2.0  # Piksel değerlerini [0, 1] aralığına dönüştür

    # n x n şeklinde bir ızgara üzerinde görüntüleri göster
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])

    # Görüntüleri dosyaya kaydet
    filename = f'generated_plot_e{epoch + 1:03d}.png'
    plt.savefig(filename)
    plt.close()


# Model performansını özetleme işlemi
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # Gerçek görüntüler ve etiketleri seç
    X_real = dataset[np.random.randint(0, dataset.shape[0], n_samples)]
    y_real = tf.ones((n_samples, 1))
    # Gerçek görüntülerde ayırt edici modelin doğruluğunu hesapla
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

    # Rastgele noktalar üret ve bunları kullanarak sahte görüntüler oluştur
    latent_points = np.random.randn(n_samples, latent_dim)
    X_fake = g_model.predict(latent_points)
    y_fake = tf.zeros((n_samples, 1))
    # Sahte görüntülerde ayırt edici modelin doğruluğunu hesapla
    _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)

    # Performans özeti
    print(f'Accuracy real: {acc_real * 100:.0f}%, fake: {acc_fake * 100:.0f}%')
    # Görüntüleri kaydet
    save_plot(X_fake, epoch)
    # Üreteç modelini kaydet
    g_model.save(f'generator_model_{epoch + 1:03d}.h5')


# Eğitim işlemi
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=1024, epoch=0):
    (X_train, _), (_, _) = dataset
    X_train = preprocess_data(X_train.astype(np.float32))

    # Eğitim döngüsü (epochs)
    for epoch in range(epoch, n_epochs):
        # Mini-batch döngüsü
        for i in range(X_train.shape[0] // n_batch):
            # Gerçek görüntüler ve etiketleri seç
            X_real = X_train[np.random.randint(0, X_train.shape[0], n_batch)]
            y_real = tf.ones((n_batch, 1))
            # Gerçek görüntüler üzerinde ayırt edici modeli eğit
            d_loss_real = d_model.train_on_batch(X_real, y_real)

            # Rastgele noktalar üret ve bunları kullanarak sahte görüntüler oluştur
            latent_points = np.random.randn(n_batch, latent_dim)
            X_fake = g_model.predict(latent_points)
            y_fake = tf.zeros((n_batch, 1))
            # Sahte görüntüler üzerinde ayırt edici modeli eğit
            d_loss_fake = d_model.train_on_batch(X_fake, y_fake)

            # GAN modelini eğit
            gan_y = tf.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(latent_points, gan_y)

            print(
                f'Epoch: {epoch + 1}, Batch: {i + 1}, d_loss_real: {d_loss_real[0]:.3f}, d_loss_fake: {d_loss_fake[0]:.3f}, g_loss: {g_loss:.3f}')

        # Her 10 epoch'ta bir performans özeti ve görüntü kaydet
        if (epoch + 1) % 10 == 0:
            summarize_performance(epoch, g_model, d_model, X_train, latent_dim)


latent_dim = 100
g_model = define_generator(latent_dim)
d_model = define_discriminator()
gan_model = define_gan(g_model, d_model)
dataset = cifar10.load_data()

train(g_model, d_model, gan_model, dataset, latent_dim)

# eğitime yarıda kalırsa devam etme durumu
# g_model = load_model('generator_model_130.h5', custom_objects={'SelfAttention': SelfAttention})
g_model = define_generator(latent_dim)

d_model = define_discriminator()
gan_model = define_gan(g_model, d_model)
dataset = cifar10.load_data()

# eğitime yarıda kalırsa devam etme durumu
# train(g_model, d_model, gan_model, dataset, latent_dim,epoch=130)
train(g_model, d_model, gan_model, dataset, latent_dim)


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def create_plot(examples, n, epoch):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, :])
    plt.savefig("./images/image_at_epoch_{:04d}.png".format(epoch))
    plt.show()


def generate_and_save_images(model, epoch, test_input):
    latent_points = generate_latent_points(100, 100)
    X = model.predict(latent_points)
    X = (X + 1) / 2.0
    create_plot(X, 10, epoch)


# load model
model = load_model('generator_model_130.h5', custom_objects={'SelfAttention': SelfAttention})
# generate images
generate_and_save_images(model, 130, 100)

anim_file = 'sagan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('generated*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

embed.embed_file(anim_file)
