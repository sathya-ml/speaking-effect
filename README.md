# Improving the Accuracy of Automatic Facial Expression Recognition in Speaking Subjects With Deep Learning

This repository contains the code used to produce the results presented in the paper [Improving the accuracy of automatic facial expression recognition in speaking subjects with deep learning](https://www.mdpi.com/2076-3417/10/11/4002).

When automatic facial expression recognition is applied to video sequences of speaking subjects, the recognition accuracy has been noted to be lower than with video sequences of still subjects. This effect known as the **speaking effect** arises during spontaneous conversations, and along with the affective expressions the speech articulation process influences facial configurations. We question whether, aside from facial features, other cues relating to the articulation process would increase emotion recognition accuracy when added in input to a deep neural network model.

## Environment

- Python 3.8
- Dependencies are listed in `requirements.txt` (Note: specific package versions are not specified except for tensorflow)

## Project Structure

- `classify/`: Scripts for training the five models
- `config/`: Configuration files for different experiments
- `lipnet/`: LipNet implementation for lip movement feature extraction
- `models/`: Definition of the CNN and RNN models
- `preprocess/`: Data preprocessing scripts and utilities
- `util/`: Common utility functions
- `vgg19_fer_net/`: VGG19-based facial expression recognition network

## Third-Party Code and Models

This project uses third-party code and components including LipNet, a facial expression recognition model, and the RAVDESS dataset. When using these components, please ensure compliance with their respective licenses:

- [LipNet](https://github.com/rizkiarm/LipNet/blob/master/LICENSE)
- [Facial Expression Recognition Model (VGG19)](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch/blob/master/LICENSE)
- [RAVDESS](https://zenodo.org/records/1188976)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bursic2020improving,
  title={Improving the accuracy of automatic facial expression recognition in speaking subjects with deep learning},
  author={Bursic, Sathya and Boccignone, Giuseppe and Ferrara, Alfio and D'Amelio, Alessandro and Lanzarotti, Raffaella},
  journal={Applied Sciences},
  volume={10},
  number={11},
  pages={4002},
  year={2020},
  publisher={MDPI}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2020 [PHuSe Lab](https://phuselab.di.unimi.it/)
