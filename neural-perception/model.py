from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras import Model
from .data_loader import TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH


class PoseRegress(Model):
    def __init__(self):
        super(PoseRegress, self).__init__()

        self.conv1 = Conv2D(16, kernel_size=3, padding='same', activation='relu',
                            input_shape=(TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3))
        self.pool1 = MaxPool2D(pool_size=(2, 2))

        self.conv2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2, 2))

        self.conv3 = Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.pool3 = MaxPool2D(pool_size=(2, 2))

        self.res_conv1 = Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.res_conv2 = Conv2D(128, kernel_size=1, padding='same', activation='relu')
        self.res_conv3 = Conv2D(128, kernel_size=3, padding='same', activation='relu')

        self.res_skip = Conv2D(128, kernel_size=1, activation='relu')

        self.flatter = Flatten()

        self.localizer = Dense(512, activation='relu')
        self.drop1 = Dropout(0.5)

        self.regressor_1 = Dense(256, activation='relu')
        self.drop2 = Dropout(0.25)

        self.regressor_2 = Dense(32, activation='relu')

        self.out = Dense(2, activation='linear')

    def call(self, x, training=False, **kwargs):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        res = self.pool3(x)

        x = self.res_conv1(res)
        x = self.res_conv2(x)
        x = self.res_conv3(x)

        res = self.res_skip(res) + x

        x = self.flatter(res)

        x = self.localizer(x)
        if training:
            x = self.drop1(x, training=training)

        x = self.regressor_1(x)

        if training:
            x = self.drop2(x, training=training)

        x = self.regressor_2(x)

        return self.out(x)
