#https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
#Model Dependencies
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Lambda, Flatten, Conv2DTranspose
from tensorflow.keras.layers import Dropout, GaussianNoise, Input, UpSampling2D, Concatenate
from tensorflow.keras.models import Model, Sequential, load_model
#Inference Dependencies
from tensorflow.python.compiler.tensorrt import trt_convert
import tensorrt as trt

#Set Configs
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

def build_model(lossfunc="mean_squared_error"):
    tf.keras.backend.clear_session()
    deltainp = Input((25*25*5), dtype=tf.float32, batch_size=32) #Delta Mask
    gofinp = Input((25,25,1), batch_size=32) #Game of Life Board
    #Increasing Filters to Capture Relationships
    conv1 = Conv2D(filters=32, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(gofinp)
    max1 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same")(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(max1)
    max2 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same")(conv2)
    #Reducing Filters to Final Output
    conv3 = Conv2D(filters=32, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(max2)
    max3 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same")(conv3)
    final = Conv2D(filters=5, kernel_size=(7,7), strides=(1,1), padding="same", activation=None)(max3)
    #Evaluate Required Board
    final = tf.keras.layers.Activation(activation="sigmoid")(Flatten()(final))
    final = Flatten()(tf.math.multiply(final, deltainp))
    model = Model(inputs=[gofinp, deltainp], outputs=final)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3), loss=lossfunc)
    return model

#PB File Generation with TF Saved_Model
model = build_model()
model.load_weights("./model2MSE.h5")
tf.saved_model.save(model, "./")

#Perform Graph Optimization for TRT
conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(precision_mode='FP16')
converter = trt_convert.TrtGraphConverterV2(input_saved_model_dir="./")
converter.convert()

#Input for Offline Engine Building
def my_input_fn():
    Inp1 = np.random.normal(size=(32,3125)).astype(np.float32)
    Inp2 = np.random.normal(size=(32,25,25,1)).astype(np.float32)
    yield  (Inp1, Inp2)
converter.build(input_fn=my_input_fn)
converter.save("./")
