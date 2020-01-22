import numpy as np
import keras
from keras.utils import to_categorical
from keras_applications.mobilenet import MobileNet

class DataGenerator(keras.utils.Sequence):
    # Initialization - Sequence 상속 - Multi Processing
    # dim, n_channel 등의 옵션
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    # at both-edge of each epoch
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        # 각 에폭마다 새로운 order 만듦, 단순 index의 셔플
        # 셔플을 통해 각 배치마다 동일한 데이터셋 학습을 방지함~ robust

    def normalize(img):

    # *** data batch generating ***
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]
        # keras.utils.to_categorical 함수 => y의 숫자 label 을 binary form으로 변환
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes) # arcFace 맞추어 수정

    # Custom; Model([input, y], output)
    def flow(self, X_list, Y_list):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        if y is not None:
            for i, (X_list,Y_list) in enumerate(zip(X_list, Y_list)):
                X[i] = X_list
                y[i] = Y_list
            return X, to_categorical(y, num_classes=self.n_classes)
        else:
            for i, X_list in enumerate(X_list):
                X[i] = X_list
            return X

    # 각 call request -> batch index 0~max size
    # model이 한 epoch 당 전체 training 데이터 대다수 훑는 것과 같은 효과 !
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # index 따라 batch processing 호출되면 generator가 호출하는 함수
    # batch size의 entry 계산해서 반환함(index 따라 카운팅)
    def __getitem__(self, index):
        'Generate one batch of data'
        # 배치 인덱스 생성 -> ID 리스트 찾고 -> 데이터 생성
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)
        return X, y