import abc
import logging

import numpy
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.transform import resize

from vgg19_fer_net import vgg_emotion_model


class FacialEmotionClassifier(abc.ABC):
    @abc.abstractmethod
    def eval_img_from_cropped_face(self, cropped_face):
        pass

    @staticmethod
    def get_facial_emotion_classifier(model_name, model_path):
        if model_name == "GitHubModel":
            return FacialEmotionClassifierGitHub(
                model_path=model_path
            )
        else:
            raise NotImplementedError(f"the model {model_name} doesn't exist")


class FacialEmotionClassifierGitHub(FacialEmotionClassifier):
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def __init__(self, model_path: str):
        self._model_path = model_path

        self._cut_size = 44
        self._transform_test = transforms.Compose([
            transforms.TenCrop(self._cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])

        self._load_model()
        self._logger = logging.getLogger(FacialEmotionClassifierGitHub.__name__)

    def _load_model(self):
        self._model = vgg_emotion_model.VGG('VGG19')
        checkpoint = torch.load(self._model_path)
        self._model.load_state_dict(checkpoint['net'])
        self._model.cuda()
        self._model.eval()

        print("MODEL SUMMARY:")
        print(self._model)

        self._intermediate_layer = self._model.features

    def _rgb2gray(self, rgb):
        return numpy.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def eval_img_from_cropped_face(self, cropped_face):
        image_array = numpy.array(cropped_face)
        gray = self._rgb2gray(image_array)
        gray = resize(gray, (48, 48), mode='symmetric').astype(numpy.uint8)

        img = gray[:, :, numpy.newaxis]

        img = numpy.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)

        # preprocess
        with torch.no_grad():
            image_tensor = self._transform_test(img)

            # add the batch dimension
            # image_tensor = image_tensor.unsqueeze(0)

            # could this be replaced by unqueeze?
            ncrops, c, h, w = numpy.shape(image_tensor)
            inputs = image_tensor.view(-1, c, h, w)

            inputs = inputs.cuda()

            # according to pytorch docs this is deprecated and has
            # no effect. It has been replaced by torch.no_grad()
            # inputs = Variable(inputs, volatile=True)

            try:
                outputs = self._intermediate_layer(inputs)
                outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

            except (RuntimeError, ValueError) as err:
                self._logger.error("couldn't evaluate due to \"{}\"\n".format(err))
                raise

        return outputs_avg
