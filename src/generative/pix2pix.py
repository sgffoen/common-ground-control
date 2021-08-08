import tensorflow as tf
import os
from matplotlib import pyplot as plt
import time
from IPython import display


class ML():
    def __init__(self,
                 ld,
                 BUFFER_SIZE,
                 BATCH_SIZE,
                 IMG_WIDTH,
                 IMG_HEIGHT,
                 OUTPUT_CHANNELS,
                 LAMBDA):
        # LearningData
        self.ld = ld
        # The facade training set consist of 400 images
        self.BUFFER_SIZE = BUFFER_SIZE
        # The batch size of 1 produced better results for-
        # the U-Net in the original pix2pix experiment
        self.BATCH_SIZE = BATCH_SIZE
        # amount of data
        self.data_num = int(BUFFER_SIZE / 2)
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        # BUILD THE GENERATOR (modified U-Net)
        self.OUTPUT_CHANNELS = OUTPUT_CHANNELS
        # DEFINE the generator loss
        self.LAMBDA = LAMBDA
        self.loss_object = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.summaru_writer = None

    def load(self, img_path, save=False):
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)

        # Split each image tensor into two tensors:
        # - one with a real building facade image
        # - one with an architecture label image
        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, :w, :]
        real_image = image[:, w:, :]

        # Convert both images to float32 tensors
        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        if save:
            # save process
            self.save_fig(path=(self.ld.plots_dir+'/input.png'),
                          img=input_image/255.)
            self.save_fig(path=(self.ld.plots_dir+'/real.png'),
                          img=real_image/255.)
            # inspect some of the preprocessed image
            self.preprocess_sample(input_image,
                                   real_image,
                                   path=(self.ld.plots_dir+'/preprocess.png'))

        return input_image, real_image

    def save_fig(self, path, img, disc=False):
        plt.figure()
        plt.axis()
        if disc:
            plt.imshow(img, vmin=-20, vmax=20, cmap='RdBu_r')
            plt.colorbar
        else:
            plt.imshow(img)
        plt.savefig(path)

    def preprocess_sample(self, inp, re, path):
        plt.figure(figsize=(6, 6))
        for i in range(4):
            rj_inp, rj_re = self.random_jitter(inp, re)
            plt.subplot(2, 2, i + 1)
            plt.imshow(rj_inp / 255.0)
            plt.axis('off')
        plt.savefig(path)

    @tf.function()
    def random_jitter(self, input_image, real_image):
        # Resizing to 286x286
        input_image, real_image = self.resize(input_image,
                                              real_image,
                                              286,
                                              286)

        # Random cropping back to 256x256
        input_image, real_image = self.random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def resize(self, input_image, real_image, height, width):
        input_image = tf.image.resize(input_image,
                                      [height, width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image,
                                     [height, width],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return input_image, real_image

    def random_crop(self, input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image,
                                             size=[2,
                                                   self.IMG_HEIGHT,
                                                   self.IMG_WIDTH,
                                                   3])
        return cropped_image[0], cropped_image[1]

    '''helpers'''

    # Normalizing the images to [-1, 1]
    def normalize(self, input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image

    def load_image_train(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def load_image_test(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.resize(input_image,
                                              real_image,
                                              self.IMG_HEIGHT,
                                              self.IMG_WIDTH)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    '''encoder
    (  Convolution
    -> Batch normalization
    -> Leaky ReLU) '''

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
                tf.keras.layers.Conv2D(filters,
                                       size,
                                       strides=2,
                                       padding='same',
                                       kernel_initializer=initializer,
                                       use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())

        return result

    '''decoder
    (  Transposed convolution
    -> Batch normalization
    -> Dropout (applied to the first 3 blocks)
    -> ReLU) '''

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
                tf.keras.layers.Conv2DTranspose(filters, size,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())

        return result

    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        down_stack = [
                    self.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
                    self.downsample(128, 4),                        # (batch_size, 64, 64, 128)
                    self.downsample(256, 4),                        # (batch_size, 32, 32, 256)
                    self.downsample(512, 4),                        # (batch_size, 16, 16, 512)
                    self.downsample(512, 4),                        # (batch_size, 8, 8, 512)
                    self.downsample(512, 4),                        # (batch_size, 4, 4, 512)
                    self.downsample(512, 4),                        # (batch_size, 2, 2, 512)
                    self.downsample(512, 4),                        # (batch_size, 1, 1, 512)
                    ]

        up_stack = [
                    self.upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
                    self.upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
                    self.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
                    self.upsample(512, 4),                      # (batch_size, 16, 16, 1024)
                    self.upsample(256, 4),                      # (batch_size, 32, 32, 512)
                    self.upsample(128, 4),                      # (batch_size, 64, 64, 256)
                    self.upsample(64, 4),                       # (batch_size, 128, 128, 128)
                    ]

        initializer = tf.random_normal_initializer(0., 0.02)

        # (batch_size, 256, 256, 3)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS,
                                               4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        generator = tf.keras.Model(inputs=inputs, outputs=x)
        return generator

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output),
                                    disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    '''build discriminator'''

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        # (batch_size, 256, 256, channels*2)
        x = tf.keras.layers.concatenate([inp, tar])

        down1 = self.downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        # (batch_size, 34, 34, 256)
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)

        # (batch_size, 31, 31, 512)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        # (batch_size, 33, 33, 512)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

        # (batch_size, 30, 30, 1)
        last = tf.keras.layers.Conv2D(1,
                                      4,
                                      strides=1,
                                      kernel_initializer=initializer)(zero_pad2)

        discriminator = tf.keras.Model(inputs=[inp, tar], outputs=last)
        return discriminator

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output),
                                     disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output),
                                          disc_generated_output)
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def generate_images(self, model, test_input, tar, plot_dir, step):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        # plt.show()
        fname = os.path.join(plot_dir,
                             'predicted_img_{}.png'.format(step))
        plt.savefig(fname)

    @tf.function
    def train_step(self, generator, discriminator, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target],
                                             training=True)
            disc_generated_output = discriminator([input_image, gen_output],
                                                  training=True)

            (gen_total_loss,
             gen_gan_loss,
             gen_l1_loss) = self.generator_loss(disc_generated_output,
                                                gen_output,
                                                target)
            disc_loss = self.discriminator_loss(disc_real_output,
                                                disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                 generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                     discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

    def fit(self,
            generator,
            discriminator,
            train_ds,
            test_ds,
            checkpoint,
            ckpt_dir,
            model_dir,
            plot_dir,
            steps):
        start = time.time()
        example_input, example_target = next(iter(test_ds.take(1)))
        checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step) % 1000 == 0:
                display.clear_output(wait=True)

                if step != 0:
                    print(f'Time taken for 1000 steps: {(time.time()-start)/60} min\n')

                self.generate_images(generator,
                                     example_input,
                                     example_target,
                                     plot_dir,
                                     step)
                print(f"Step: {step//1000}k")

            self.train_step(generator,
                            discriminator,
                            input_image,
                            target,
                            step)

            # Training step
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)

            # Save (checkpoint) the model every 1k steps
            if (step + 1) % 1000 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

        generator.compile(optimizer=self.generator_optimizer,
                          loss=self.loss_object,
                          metrics=None,
                          loss_weights=None,
                          weighted_metrics=None,
                          run_eagerly=None,
                          steps_per_execution=None)
        generator.save(model_dir)

    def load_model(self, model_dir):
        loaded_model = tf.keras.models.load_model(model_dir)
