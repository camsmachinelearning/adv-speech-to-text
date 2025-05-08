import numpy as np
from scipy.optimize import linear_sum_assignment


class Association:

    def __init__(self,
                 consideration_threshold: float | np.float32,
                 switch_penalty: float | np.float32,
                 voice_blend_weight: float | np.float32 = 0.8,
                 face_blend_weight: float | np.float32 = 0.5,
                 confidence_decay: float | np.float32 = 0.05):
        """
        :param consideration_threshold: confidence threshold to even consider the association
        :param switch_penalty: added cost for switching at full confidence
        :param voice_blend_weight: voice prediction blending weight
        :param face_blend_weight: face prediction blending weight
        :param confidence_decay: rate at which old considerations lose importance. Must be in the interval (0, 1]
        """

        self._correlation = np.zeros((0, 0), dtype=np.float32)
        self._rows = np.zeros(0, dtype=np.int32)
        self._columns = np.zeros(0, dtype=np.int32)
        self._cost = 0.0

        # for streamed covariance calculation
        self._outer_product_matrix = np.zeros((0, 0), dtype=np.float32)
        self._voice_probabilities = np.zeros(0, dtype=np.float32)
        self._voice_means = np.zeros(0, dtype=np.float32)
        self._voice_squared_means = np.zeros(0, dtype=np.float32)
        self._face_probabilities = np.zeros(0, dtype=np.float32)
        self._face_means = np.zeros(0, dtype=np.float32)
        self._face_squared_means = np.zeros(0, dtype=np.float32)

        self.consideration_threshold = np.float32(consideration_threshold)
        self.switch_penalty = np.float32(switch_penalty)
        self.voice_blend_weight = np.float32(voice_blend_weight)
        self.face_blend_weight = np.float32(face_blend_weight)
        self.confidence_decay = np.float32(confidence_decay)

    def update(self, p_voice: np.ndarray[np.float32], p_face: np.ndarray[np.float32]):
        # voice axis = row
        # face axis = column
        assert p_voice.shape[0] >= self._correlation.shape[0], "a voice profile was deleted"
        assert p_face.shape[0] >= self._correlation.shape[1], "a face profile was deleted"

        # resize if needed
        resize_voice: bool = p_voice.shape[0] > self._correlation.shape[0]
        resize_face: bool = p_face.shape[0] > self._correlation.shape[1]

        if resize_voice:
            extension = np.zeros(1, dtype=np.float32)
            self._voice_probabilities = np.hstack((self._voice_probabilities, extension))
            self._voice_means = np.hstack((self._voice_probabilities, extension))
            self._voice_squared_means = np.hstack((self._voice_probabilities, extension))
        if resize_face:
            extension = np.zeros(1, dtype=np.float32)
            self._face_probabilities = np.hstack((self._face_probabilities, extension))
            self._face_means = np.hstack((self._face_probabilities, extension))
            self._face_squared_means = np.hstack((self._face_probabilities, extension))

        if resize_voice or resize_face:
            new_shape = ((0, p_voice.shape[0] - self._correlation.shape[0]),
                         (0, p_face.shape[0] - self._correlation.shape[1]))

            self._correlation = np.pad(self._correlation,
                                       new_shape,
                                       mode='constant',
                                       constant_values=0)

            self._outer_product_matrix = np.pad(self._outer_product_matrix,
                                                new_shape,
                                                mode='constant',
                                                constant_values=0)

        # update probability streams with a low pass filter
        self._voice_probabilities = np.interp(p_voice, self._voice_probabilities, self.voice_blend_weight)
        self._face_probabilities = np.interp(p_face, self._face_probabilities, self.face_blend_weight)

        # streamed covariance
        self._voice_means = np.interp(self._voice_probabilities, self._voice_means, self.confidence_decay)
        self._face_means = np.interp(self._face_probabilities, self._face_means, self.confidence_decay)

        self._voice_squared_means = np.interp(np.square(self._voice_probabilities), self._voice_squared_means,
                                              self.confidence_decay)
        self._face_squared_means = np.interp(np.square(self._face_probabilities), self._face_squared_means,
                                             self.confidence_decay)

        self._outer_product_matrix = np.interp(np.outer(self._voice_probabilities, self._face_probabilities),
                                               self._outer_product_matrix,
                                               self.confidence_decay)

        # compute switch penalties before computing correlation
        cost_matrix = np.zeros_like(self._correlation)
        cost_matrix[self._rows, self._columns] = (self.switch_penalty *
                                                  self._correlation[self._rows, self._columns])

        # compute correlation
        self._correlation = ((self._outer_product_matrix - np.outer(self._voice_means, self._face_means)) /
                             np.outer(np.reciprocal(np.sqrt(self._voice_squared_means - np.square(self._voice_means))),
                                      np.reciprocal(np.sqrt(self._face_squared_means - np.square(self._face_means)))))

        cost_matrix = -np.clip(cost_matrix + self._correlation, self.consideration_threshold, 1. + self.switch_penalty)

        # run Hungarian Matching Algorithm
        self._rows, self._columns = linear_sum_assignment(cost_matrix)

        # compute cost
        self._cost = cost_matrix[self._rows, self._columns].sum()

    def get_cost(self) -> float:
        return self._cost

    def __getitem__(self, index) -> tuple[int, int, float]:
        """
        Association[index] -> tuple[int, int, float]
        :param index: index of the element in the association
        :return: voice index, face index, confidence
        """
        assert (index < len(self._rows))
        return self._rows[index], self._columns[index], self._correlation[self._rows[index], self._columns[index]]
