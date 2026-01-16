import keras
from keras.layers import Input, Dense, Activation, Embedding, Flatten, Reshape, Layer, Dropout, BatchNormalization, AveragePooling2D, Bidirectional, TimeDistributed, GlobalMaxPooling1D,GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D
from keras.layers import SimpleRNN, GRU, LSTM
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers import Add, Concatenate
from keras.layers import LocallyConnected2D, LayerNormalization, MultiHeadAttention
from keras.models import Model, Sequential
from metrics import rmse, mape, mae, MyReshape, MyInverseReshape, get_model_save_path, matrixLayer, MyInverseReshape2, matrixLayer2
from LookupConv import Lookup, LookUpSqueeze
import tensorflow as tf
import numpy as np
from keras.backend import squeeze
from keras.layers import Lambda
from model_utils import vehicle_embedding_layer, Patches, PatchEncoder, transformer_layers, mlp

class Factory(object):

    def get_model(self, conf, arm_shape):
        print("use model ", conf.model_name)
        model = conf.model_name
        function_name = "self.{}_model(conf, arm_shape)".format(conf.model_name)
        return eval(function_name)

    def RNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 2))
        output = MyReshape(conf.batch_size)(input_x)
        # output = SimpleRNN(32, return_sequences=True)(output)
        output = SimpleRNN(conf.observe_length)(output)
        # output = Dropout(0.1)(output)
        output = Dense(conf.predict_length, activation="tanh")(output)
        output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    def BiRNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 1))
        output = MyReshape(conf.batch_size)(input_x)
        # output = SimpleRNN(32, return_sequences=True)(output)
        output1 = SimpleRNN(conf.observe_length)(output)
        output2 = SimpleRNN(conf.observe_length, go_backwards=True)(output)
        output = Add()([output1, output2])
        # output = Dropout(0.1)(output)
        output = Dense(conf.predict_length, activation="tanh")(output)
        output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    def GRU_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 2))
        output = MyReshape(conf.batch_size)(input_x)
        output = GRU(conf.observe_length)(output)
        output = Dense(conf.predict_length)(output)
        output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    def LSTM_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 2))
        output = MyReshape(conf.batch_size)(input_x)
        output = LSTM(conf.observe_length)(output)
        output = Dense(conf.predict_length)(output)
        output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=input_x, outputs=output)
        return model 

    def DCNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 2))
        output = Conv2D(32, (5, 2), strides=(1, 1), padding="same")(input_x)
        output = Conv2D(32, (5, 2), strides=(1, 1), padding="same")(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Activation(activation="sigmoid")(output)

        output = Conv2D(16, (2, 2), strides=(1, 1), padding="same")(output)
        output = Conv2D(16, (2, 2), strides=(1, 1), padding="same")(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Activation(activation="sigmoid")(output)

        output = Conv2D(1, (2, 2), strides=(1, 1), padding="same")(output)
        output = Conv2D(1, (2, 2), strides=(1, 1), padding="same")(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Activation(activation="tanh")(output)

        output = Dense(conf.predict_length)(output)
        print('output.shape activation',output.shape)
        output = Reshape((road_num, conf.predict_length))(output)
        inputs = [input_x]
        model = Model(inputs=inputs, outputs=output)
        return model

    def CRNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 1))
        output = Conv2D(32, (2, 2), strides=(1, 1), padding="same")(input_x)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Activation(activation="relu")(output)
        output = Conv2D(16, (2, 2), strides=(1, 1), padding="same")(output)
        # pool2 = AveragePooling2D(pool_size=(1,2))(conv2)
        # pool2 = Activation(activation="sigmoid")(conv2)
        # conv3 = Conv2D(4, (2, 2), strides=(1, 1), padding="same")(pool2)
        # pool3 = AveragePooling2D(pool_size=(1, 2))(conv3)
        output = Activation(activation="relu")(output)
        output = MyReshape(conf.batch_size)(output)
        output = SimpleRNN(5)(output)
        output = Dense(conf.predict_length)(output)
        output = MyInverseReshape(conf.batch_size)(output)
        # f = Flatten()(pool3)
        # output = Dense(road_num * conf.predict_length, activation="sigmoid")(f)
        # output = Reshape((road_num, conf.predict_length))(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    def LCRNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 2))
        input_ram = Input(arm_shape)
        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)

        output = MyReshape(conf.batch_size)(output)
        output = SimpleRNN(5)(output)
        inputs = [input_x, input_ram]

        if conf.use_externel:
            output = Dense(conf.predict_length, activation="relu")(output)
            output = MyInverseReshape2(conf.batch_size)(output)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                outputs = [matrixLayer()(output)]
                outputs.append(matrixLayer()(output_e))
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation("tanh")(output)
        else:
            output = Dense(conf.predict_length, activation="tanh")(output)
            output = MyInverseReshape2(conf.batch_size)(output)

        model = Model(inputs=inputs, outputs=output)
        return model

    def __E_input_output(self, conf, arm_shape, activation="tanh"):
        road_num = arm_shape[0]
        if conf.observe_p != 0:
            input_x1 = Input((road_num, conf.observe_p))
            output1 = MyReshape(conf.batch_size)(input_x1)
            output1 = Dense(conf.observe_p + 1, activation="relu")(output1)

        
        if conf.observe_t != 0:
            input_x2 = Input((road_num, conf.observe_t))
            output2 = MyReshape(conf.batch_size)(input_x2)
            output2 = Dense(conf.observe_t + 1, activation="relu")(output2)
        
        if conf.observe_p != 0:
            if conf.observe_t != 0:
                output = Concatenate()([output1, output2])
                input_x = [input_x1, input_x2]
            else:
                output = output1
                input_x = input_x1
        else:
            output = output2
            input_x = input_x2

        output = Dense(conf.predict_length, activation=activation)(output)
        output = MyInverseReshape2(conf.batch_size)(output)
        print('output shape', output.shape)
        
        if conf.use_externel:
            input_x3 = Input((conf.predict_length, 34))  # 34 is externel dim with meteorol
        else:
            input_x3 = Input((conf.predict_length, 22))  # 22 is externel dim (21 vec and 1 holiday)

        if isinstance(input_x, list):
            input_x += [input_x3]
        else:
            input_x = [input_x, input_x3]

        output_3 = MyReshape(conf.batch_size)(input_x3)
        output_3 = Dense(road_num-1, activation=activation)(output_3)
        output_3 = MyInverseReshape(conf.batch_size)(output_3)
        print('output_3 shape',output_3.shape)
        print('road num', road_num)
        output_3 = Reshape((road_num-1, conf.predict_length))(output_3)
        print('output_3 reshaped', output_3.shape)
        output = Add()([output, output_3])
        print('output shape after add',output.shape)
        return input_x, output

    def E_model(self, conf, arm_shape):
        input_x, output = self.__E_input_output(conf, arm_shape)
        model = Model(inputs=input_x, output=output)
        return model

    def LCRNNBN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)
        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)
        
        
        print('output.shape', output.shape)
        output = MyReshape(conf.batch_size)(output)
        print('output.shape', output.shape)
        output = SimpleRNN(5)(output)
        #output = MyReshape(conf.batch_size)(output)
        inputs = [input_x, input_ram]
        if conf.use_externel:
            output = Dense(conf.predict_length, activation="relu")(output)
            output = MyInverseReshape2(conf.batch_size)(output)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                outputs = [matrixLayer()(output)]
                outputs.append(matrixLayer()(output_e))
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation("tanh")(output)
        else:
            output = Dense(conf.predict_length, activation="tanh")(output)
            output = MyInverseReshape2(conf.batch_size)(output)
           
        output = Dense(conf.predict_length, activation="tanh")(output)
        print('final layer', output.shape)
        model = Model(inputs=inputs, outputs=output)
        return model

    def LCNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)
        # input_effective = Input((arm_shape[0],))
        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)
        # output = Effective()([output, input_effective])
        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)
        # output = Effective()([output, input_effective])

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 8), activation="relu")(output)
        output = LookUpSqueeze()(output)
        output = Lambda(lambda y: squeeze(y, 0))(output)
        inputs = [input_x, input_ram]

        if conf.use_externel:
            output = Dense(1, activation='relu')(output)
            print('dense.output.shape',output.shape)
            output = MyInverseReshape2(conf.batch_size)(output)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                outputs = [matrixLayer()(output)]
                print('outputs.e',output_e)
                outputs.append(matrixLayer2()(output_e))
                print('outputs.shape',outputs)
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation('tanh')(output)
        else:
            output = Dense(1, activation='tanh')(output)
            output = MyInverseReshape2(conf.batch_size)(output)

        model = Model(inputs=inputs, outputs=output)
        return model


    def RESNET_model(self, conf, arm_shape):
        n_feature_maps = 8 #when you have large dataset increase it to 16 or 32 or 64 , experiment
        road_num = arm_shape[0]
        A = arm_shape[1]
        
        input_x = Input((road_num, conf.observe_length, 2))
        print('inpu+++++++++++++++++++++++++++++', input_x.shape)
        input_ram = Input(arm_shape)
        print('input_x.shape', input_x.shape)
        print('input_ram.shape', input_ram.shape)
        print('road_num.shape', road_num)
        # BLOCK 1 
        conv_x = Lookup(conf.batch_size)([input_x, input_ram])
        conv_x = Conv3D(n_feature_maps, (1, A, 2), activation='relu')(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = LookUpSqueeze()(conv_x)

        conv_y = Lookup(conf.batch_size)([conv_x, input_ram])
        conv_y = Conv3D(n_feature_maps, (1, A, 2), activation='relu')(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = LookUpSqueeze()(conv_y)

        conv_z = Lookup(conf.batch_size)([conv_y, input_ram])
        conv_z = Conv3D(n_feature_maps, (1, A, 2))(conv_z)
        conv_z = BatchNormalization()(conv_z)
        conv_z = LookUpSqueeze()(conv_z)
        
        # expand channels for the sum 
        shortcut_y = Lookup(conf.batch_size)([input_x, input_ram])
        shortcut_y = Conv3D(n_feature_maps, (1, A, 4),activation='relu')(shortcut_y)
        shortcut_y = BatchNormalization()(shortcut_y)
        shortcut_y = LookUpSqueeze()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = Activation('relu')(output_block_1)

        # BLOCK 2
        conv_x = Lookup(conf.batch_size)([output_block_1, input_ram])
        conv_x = Conv3D(n_feature_maps*2, (1, A, 2), activation='relu')(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = LookUpSqueeze()(conv_x)

        conv_y = Lookup(conf.batch_size)([conv_x, input_ram])
        conv_y = Conv3D(n_feature_maps*2, (1, A, 2),activation='relu')(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = LookUpSqueeze()(conv_y)

        conv_z = Lookup(conf.batch_size)([conv_y, input_ram])
        conv_z = Conv3D(n_feature_maps*2, (1, A, 2))(conv_z)
        conv_z = BatchNormalization()(conv_z)
        conv_z = LookUpSqueeze()(conv_z)

        # expand channels for the sum 
        shortcut_y = Lookup(conf.batch_size)([output_block_1, input_ram])
        shortcut_y = Conv3D(n_feature_maps*2, (1, A, 4),activation='relu')(shortcut_y)
        shortcut_y = BatchNormalization()(shortcut_y)
        shortcut_y = LookUpSqueeze()(shortcut_y)

        print('conv_z.shape', conv_z.shape)
        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3
        conv_x = Lookup(conf.batch_size)([output_block_2, input_ram])
        conv_x = Conv3D(n_feature_maps*2, (1, A, 2), activation='relu')(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = LookUpSqueeze()(conv_x)

        conv_y = Lookup(conf.batch_size)([conv_x, input_ram])
        conv_y = Conv3D(n_feature_maps*2, (1, A, 2),activation='relu')(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = LookUpSqueeze()(conv_y)

        conv_z = Lookup(conf.batch_size)([conv_y, input_ram])
        conv_z = Conv3D(n_feature_maps*2, (1, A, 2))(conv_z)
        conv_z = BatchNormalization()(conv_z)
        conv_z = LookUpSqueeze()(conv_z)

        # need to expand
        shortcut_y = Lookup(conf.batch_size)([output_block_2, input_ram])
        shortcut_y = Conv3D(n_feature_maps*2, (1, A, 4),activation='relu')(shortcut_y)
        shortcut_y = BatchNormalization()(shortcut_y)
        shortcut_y = LookUpSqueeze()(shortcut_y)

        output_block_3 = keras.layers.add([shortcut_y,conv_z])
        output_block_3 = Activation('relu')(output_block_3)

        output = Lookup(conf.batch_size)([output_block_3, input_ram])
        if conf.use_externel:
            output = Conv3D(n_feature_maps,(1, A, 38),activation='relu')(output)
        else:
            output = Conv3D(n_feature_maps,(1, A, 1),activation='relu')(output)
        output = LookUpSqueeze()(output)
        output = Lambda(lambda y: squeeze(y, 0))(output)

        inputs = [input_x, input_ram]
        if conf.use_externel:
            output = Dense(1, activation='relu')(output)
            print('dense.output.shape',output.shape)
            output = MyInverseReshape2(conf.batch_size)(output)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                print('Fuse, output before matrixlayer', output.shape )
                outputs = [matrixLayer()(output)]
                print('outputs.e',output_e)
                outputs.append(matrixLayer2()(output_e))
                print('outputs.shape',outputs)
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation('tanh')(output)
        else:
            output = Dense(1, activation='tanh')(output)
            output = MyInverseReshape2(conf.batch_size)(output)
           
        output = Dense(conf.predict_length, activation='tanh')(output)
        print('final layer', output.shape)
        model = Model(inputs=inputs, outputs=output)
        return model

    def RESNET_BILSTM_model(self, conf, arm_shape):
        n_feature_maps = 8 #when you have large dataset increase it to 16 or 32 or 64 , experiment
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 2))#changed here
        input_ram = Input(arm_shape)

        input_veh_type = Input((road_num, conf.observe_length, 1))
        input_engine = Input((road_num, conf.observe_length, 1))
        input_weight = Input((road_num, conf.observe_length, 1))
        print('input_x.shape', input_x.shape)
        print('input_ram.shape', input_ram.shape)
        print('road_num.shape', road_num)
        # BLOCK 1 

        veh_type_embd = Embedding(5, 3, mask_zero=False)(input_veh_type)
        engine_embd = Embedding(63, 10, mask_zero=False)(input_engine)
        weight_embd = Embedding(10, 5, mask_zero=False)(input_weight)
        squeezer = Lambda(lambda x: squeeze(x, axis=-2) )
        veh_type_embd = squeezer(veh_type_embd)
        engine_embd = squeezer(engine_embd)
        weight_embd = squeezer(weight_embd)

        concat_x = Concatenate()([input_x, veh_type_embd, engine_embd, weight_embd])

        conv_x = Lookup(conf.batch_size)([concat_x, input_ram])
        conv_x = Conv3D(n_feature_maps, (1, A, 2), activation='relu')(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = LookUpSqueeze()(conv_x)

        conv_y = Lookup(conf.batch_size)([conv_x, input_ram])
        conv_y = Conv3D(n_feature_maps, (1, A, 2), activation='relu')(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = LookUpSqueeze()(conv_y)

        conv_z = Lookup(conf.batch_size)([conv_y, input_ram])
        conv_z = Conv3D(n_feature_maps, (1, A, 2))(conv_z)
        conv_z = BatchNormalization()(conv_z)
        conv_z = LookUpSqueeze()(conv_z)
        
        # expand channels for the sum 
        shortcut_y = Lookup(conf.batch_size)([concat_x, input_ram])
        shortcut_y = Conv3D(n_feature_maps, (1, A, 4),activation='relu')(shortcut_y)
        shortcut_y = BatchNormalization()(shortcut_y)
        shortcut_y = LookUpSqueeze()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = Activation('relu')(output_block_1)

        # BLOCK 2
        conv_x = Lookup(conf.batch_size)([output_block_1, input_ram])
        conv_x = Conv3D(n_feature_maps*2, (1, A, 2), activation='relu')(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = LookUpSqueeze()(conv_x)

        conv_y = Lookup(conf.batch_size)([conv_x, input_ram])
        conv_y = Conv3D(n_feature_maps*2, (1, A, 2),activation='relu')(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = LookUpSqueeze()(conv_y)

        conv_z = Lookup(conf.batch_size)([conv_y, input_ram])
        conv_z = Conv3D(n_feature_maps*2, (1, A, 2))(conv_z)
        conv_z = BatchNormalization()(conv_z)
        conv_z = LookUpSqueeze()(conv_z)

        # expand channels for the sum 
        shortcut_y = Lookup(conf.batch_size)([output_block_1, input_ram])
        shortcut_y = Conv3D(n_feature_maps*2, (1, A, 4),activation='relu')(shortcut_y)
        shortcut_y = BatchNormalization()(shortcut_y)
        shortcut_y = LookUpSqueeze()(shortcut_y)

        print('conv_z.shape', conv_z.shape)
        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3
        conv_x = Lookup(conf.batch_size)([output_block_2, input_ram])
        conv_x = Conv3D(n_feature_maps*2, (1, A, 2), activation='relu')(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = LookUpSqueeze()(conv_x)

        conv_y = Lookup(conf.batch_size)([conv_x, input_ram])
        conv_y = Conv3D(n_feature_maps*2, (1, A, 2),activation='relu')(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = LookUpSqueeze()(conv_y)

        conv_z = Lookup(conf.batch_size)([conv_y, input_ram])
        conv_z = Conv3D(n_feature_maps*2, (1, A, 2))(conv_z)
        conv_z = BatchNormalization()(conv_z)
        conv_z = LookUpSqueeze()(conv_z)

        # need to expand
        shortcut_y = Lookup(conf.batch_size)([output_block_2, input_ram])
        shortcut_y = Conv3D(n_feature_maps*2, (1, A, 4),activation='relu')(shortcut_y)
        shortcut_y = BatchNormalization()(shortcut_y)
        shortcut_y = LookUpSqueeze()(shortcut_y)

        output_block_3 = keras.layers.add([shortcut_y,conv_z])
        output_block_3 = Activation('relu')(output_block_3)
        print('output_block_3', output_block_3.shape)
        to_lstm = Lookup(conf.batch_size)([output_block_3, input_ram])
        print('to_lstm BEFORE EXTERNAL #############', to_lstm.shape)
        if conf.use_externel:
            to_lstm = Conv3D(n_feature_maps,(1, A, 38),activation='relu')(to_lstm)
        else:
            to_lstm = Conv3D(n_feature_maps,(1, A, 1),activation='relu')(to_lstm)
        to_lstm = LookUpSqueeze()(to_lstm)
        to_lstm = Lambda(lambda y: squeeze(y, 0))(to_lstm)
        # output_block_3 = Activation('relu')(output_block_3)

        #FINAL
        # gap_layer = GlobalAveragePooling2D()(output_block_3)
        # gap_layer = Dropout(rate=.25)(gap_layer)
        # gap_layer = Dense(50, activation='sigmoid')(gap_layer)
        # gap_layer = Dropout(rate=.25)(gap_layer)

        # time_distibuted = TimeDistributed(output_block_3)
        print('to_lstm.shape', to_lstm.shape)
        # output = SimpleRNN(5)(to_lstm)
        # output = MyReshape(conf.batch_size)(gap_layer)
        output = Bidirectional(LSTM(10, return_sequences=True, dropout=0.5, recurrent_dropout=0.2))(to_lstm)
        print('lstm out.shape', output.shape)
        #output = MyReshape(conf.batch_size)(output)
        inputs = [input_x, input_ram, input_veh_type, input_engine, input_weight]

        if conf.use_externel:
            output = Dense(1, activation='relu')(output)
            
            output = MyInverseReshape2(conf.batch_size)(output)
            print('dense.output.shape',output.shape)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            print('output_e shape',output_e.shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                outputs = [matrixLayer()(output)]
                print('outputs',outputs)
                print('outputs.e',output_e)
                outputs.append(matrixLayer2()(output_e))
                print('outputs.shape',outputs)
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation('tanh')(output)
        else:
            output = Dense(1, activation='tanh')(to_lstm)
            output = MyInverseReshape(conf.batch_size)(output)
           
        output = Dense(conf.predict_length, activation='tanh')(output)
        print('final layer', output.shape)
        model = Model(inputs=inputs, outputs=output)
        return model

    def VIT_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 2))
        
        # Reshaping road segments as batch size 
        # and we are only looking at last last two dimensions of data
        reshaped_input_x = MyReshape(conf.batch_size)(input_x)
        
        # Currently the last two dimensions are of shape (.., .. , 12, 2)
        # So they can be reshaped into patch_sizes as needed
        # If patch_size is changed, num_patches should also be changed accordingly
        # For example, if the patch_size is changed to 4, num_patches should be 12 * 2 / 4 = 6
        patch_size = 2 
        num_patches = int(reshaped_input_x.shape[-1] * reshaped_input_x.shape[-2] / patch_size)
        
        # Parameters - change to experiment
        projection_dim = 12
        num_transformer_layers = 2
        num_heads = 2
        transformer_units = [projection_dim * 2, projection_dim] 
        mlp_head_units = [64, 32]

        # Convert to patches
        patches = tf.reshape(reshaped_input_x, [road_num, num_patches, patch_size])
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
        # Add transformer layers
        t_layers = transformer_layers(encoded_patches, num_transformer_layers, num_heads, 
                                      projection_dim, transformer_units)
        # Normalize and flatten.
        representation = LayerNormalization(epsilon=1e-6)(t_layers)
        representation = Flatten()(representation)
        # Add final MLP layer.
        features = mlp(representation, hidden_units=mlp_head_units)
        # Classify and reshape outputs.
        output = Dense(conf.predict_length, activation='tanh')(features)
        output = MyInverseReshape(conf.batch_size)(output)
        # Create the Keras model.
        model = Model(inputs=[input_x], outputs=output)
        return model

    def CVT(self, conf, arm_shape):
        pass

factory = Factory()

