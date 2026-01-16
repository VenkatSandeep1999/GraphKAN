from keras.backend import squeeze
from keras.layers import Input, Add, MultiHeadAttention, Embedding, Lambda, Concatenate, Dense, Dropout, Layer, LayerNormalization
import tensorflow as tf

def vehicle_embedding_layer(conf, road_num, input_x):
    input_veh_type = Input((road_num, conf.observe_length, 1))
    input_engine = Input((road_num, conf.observe_length, 1))
    input_weight = Input((road_num, conf.observe_length, 1))
    print('input_x.shape', input_x.shape)
    print('road_num.shape', road_num)

    veh_type_embd = Embedding(5, 3, mask_zero=False)(input_veh_type)
    engine_embd = Embedding(63, 10, mask_zero=False)(input_engine)
    weight_embd = Embedding(10, 5, mask_zero=False)(input_weight)

    squeezer = Lambda(lambda x: squeeze(x, axis=-2))
    
    veh_type_embd = squeezer(veh_type_embd)
    engine_embd = squeezer(engine_embd)
    weight_embd = squeezer(weight_embd)

    concat_x = Concatenate()(
        [input_x, veh_type_embd, engine_embd, weight_embd])
    
    return [concat_x, input_veh_type, input_engine, input_weight]

def mlp(x, hidden_units, activation='relu'):
    for units in hidden_units:
        x = Dense(units, activation=activation)(x)
    return x

def transformer_layers(x, num_layers, num_heads, projection_dim, transformer_units):
    for _ in range(num_layers):
        x1 = LayerNormalization(epsilon=1e-6)(x)
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim
            )(x1, x1)
        x2 = Add()([attention_output, x])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units)
        x = Add()([x3, x2])
    return x

class Patches(Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
