import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import heapq

# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

np.set_printoptions(threshold=np.nan)

EPOCHS = 25

class FasNet:
    def __init__(self, dataset, class_num, batch_size, input_size, fine_tune_model_file='imagenet'):
        self.class_num = class_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.dataset = dataset
        self.fine_tune_model_file = fine_tune_model_file
        self.model = self.__create_model()
        
    def __dense(self, feature):
        feature = tf.keras.layers.Flatten()(feature)
        feature = tf.keras.layers.Dense(units=128)(feature)
        feature = tf.keras.layers.BatchNormalization()(feature)
        feature = tf.keras.layers.Activation(activation=tf.nn.leaky_relu)(feature)
        feature = tf.keras.layers.Dropout(0.5)(feature)
        
        feature = tf.keras.layers.Dense(units=128)(feature)
        feature = tf.keras.layers.BatchNormalization()(feature)
        feature = tf.keras.layers.Activation(activation=tf.nn.leaky_relu)(feature)
        feature = tf.keras.layers.Dropout(0.5)(feature)
        
        return feature
    
    def __extract_feature(self, model, name, input):
        model._name = name
        for layer in model.layers:
            layer.trainable = False
        return model(input)
    
    
    def __create_model(self):
        input_hsv = tf.keras.layers.Input(name='hsv_input', shape=(self.input_size, self.input_size, 3))
        input_yuv = tf.keras.layers.Input(name='yuv_input', shape=(self.input_size, self.input_size, 3))
    
        vgg_hsv = tf.keras.applications.VGG16(weights=self.fine_tune_model_file, include_top=False)
        vgg_yuv = tf.keras.applications.VGG16(weights=self.fine_tune_model_file, include_top=False)
    
        feature_hsv = self.__extract_feature(vgg_hsv, 'vgg_hsv', input_hsv)
        feature_yuv = self.__extract_feature(vgg_yuv, 'vgg_yuv', input_yuv)
    
        feature = tf.keras.layers.concatenate([feature_hsv, feature_yuv])
        feature = self.__dense(feature)
    
        output = tf.keras.layers.Dense(name='output', units=self.class_num, activation=tf.nn.softmax)(feature)
    
        model = tf.keras.Model(inputs=[input_hsv, input_yuv], outputs=output)
    
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='categorical_crossentropy')
        return model

    
    def train(self, model_file, checkpoint_dir, log_dir, max_epoches=EPOCHS, load_weight=True):
        self.model.summary()
        
        # tf.keras.utils.plot_model(self.model, to_file='model.png')
        
        if load_weight:
            self.model.load_weights(model_file)
        else:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             period=2,
                                                             verbose=1)
            earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            mode='min',
                                                            min_delta=0.001,
                                                            patience=3,
                                                            verbose=1)

            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

            input_name_list = ['hsv_input', 'yuv_input']
            output_name_list = ['output']
            self.model.fit_generator(generator=self.dataset.train_data_generator(input_name_list, output_name_list, 'train.txt'),
                                     epochs=max_epoches,
                                     steps_per_epoch=self.dataset.train_num() // self.batch_size,
                                     validation_data=self.dataset.train_data_generator(input_name_list, output_name_list, 'val.txt'),
                                     validation_steps=self.dataset.val_num() // self.batch_size,
                                     callbacks=[cp_callback, earlystop_cb, tb_callback],
                                     max_queue_size=10,
                                     workers=1,
                                     verbose=1)

            self.model.save(model_file)

    def predict(self):
        input_name_list = ['hsv_input', 'yuv_input']
        output_name_list = ['output']
        predictions = self.model.predict_generator(generator=self.dataset.test_data_generator(input_name_list, output_name_list, 'test.txt', shuffle=False),
                                                   steps=self.dataset.test_num() // self.batch_size,
                                                   verbose=1)
        preds = predictions
        print(preds)
        test_data = self.dataset.test_data_generator(input_name_list, output_name_list, 'test.txt', shuffle=False)
        correct = 0
        steps = self.dataset.test_num() // self.batch_size
        
        total = steps * self.batch_size
        
        for step in range(steps):
            _, test_batch_y = next(test_data)
            # print(test_batch_y)
            real_batch = test_batch_y['output']
            # print(real_batch)
            for i, real in enumerate(real_batch):
                pred_idx = np.argmax(preds[step * self.batch_size + i])
                # print(pred_idx)
                if real[pred_idx]:
                    correct += 1

        
        print("fas==> correct:{}, total:{}, correct_rate:{}".format(correct, total, 1.0 * correct / total))
        
        return predictions
    
    def test_online(self, face_imgs):

        batch_x_hsv = np.array(face_imgs[0]['hsv'], dtype=np.float32)
        batch_x_ycrcb = np.array(face_imgs[0]['yuv'], dtype=np.float32)

        batch_x_hsv = np.expand_dims(batch_x_hsv, 0)
        batch_x_ycrcb = np.expand_dims(batch_x_ycrcb, 0)
        
        predictions = self.model.predict({'hsv_input': batch_x_hsv, 'yuv_input': batch_x_ycrcb}, batch_size=1)
        predictions = np.asarray(predictions)
        return predictions
