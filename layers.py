from keras import backend as K
from keras.engine.topology import Layer


class PreprocessPRN(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)


    def call(self, inputs, **kwargs):
        keypoints, filters = inputs

        boxes, scores, labels = filters

        for box, score, label in zip(boxes[0], scores[0], labels[0]):

            if (score < 0.5):
                break

            x1,y1,x2,y2 = boxes

        # shape = K.cast(keras.backend.shape(image), keras.backend.floatx())
        # if keras.backend.image_data_format() == 'channels_first':
        #     height = shape[2]
        #     width = shape[3]
        # else:
        #     height = shape[1]
        #     width = shape[2]
        # x1 = backend.clip_by_value(boxes[:, :, 0], 0, width)
        # y1 = backend.clip_by_value(boxes[:, :, 1], 0, height)
        # x2 = backend.clip_by_value(boxes[:, :, 2], 0, width)
        # y2 = backend.clip_by_value(boxes[:, :, 3], 0, height)
        #
        # return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]