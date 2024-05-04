# Indian-Sign-Language-Recognition

## Theoretical Background

Sign languages, including Indian Sign Language (ISL), are rich and complex systems of communication that rely on visual and spatial elements such as hand shapes, movements, and facial expressions. Unlike spoken languages, sign languages convey meaning through gestures and expressions, making them a vital mode of communication for the deaf and hard of hearing community.

### Indian Sign Language

Indian Sign Language (ISL) is the primary sign language used by deaf and hard of hearing individuals in India. ISL is influenced by the diverse linguistic and cultural landscape of India, with regional variations existing across different parts of the country.

![image](https://github.com/Riddhi9570/Indian-Sign-Language-Recognition/assets/72887868/65b47805-b88b-4e3b-a318-80b3e9f652d1)

Fig. Alphabets Chart (ISL)


![image](https://github.com/Riddhi9570/Indian-Sign-Language-Recognition/assets/72887868/2806f242-d8b8-443b-b85e-6dc35a2c7171)

Fig. Digits Chart (ISL)


### Convolutional Neural Network (CNN)

A Convolutional Neural Network (CNN) is a type of artificial neural network commonly used in image recognition and computer vision tasks. It's designed to automatically and adaptively learn spatial hierarchies of features from input images. CNNs use layers that perform operations like convolution and pooling to automatically learn features from images. Non-linear activation functions such as ReLU (Rectified Linear Unit) are applied element-wise to the output of convolutional and pooling layers, introducing non-linearity to the network and enabling it to learn complex mappings between inputs and outputs. After several convolutional and pooling layers, CNNs often end with one or more fully connected layers, which perform classification or regression based on the learned features extracted from the input images. These learned features help the network make accurate predictions about the content of the images it's given.

## Objective

The primary objective of this project is to design, develop, and implement an Indian Sign Language Recognition (ISLR) system using Convolutional Neural Networks (CNNs) to accurately interpret ISL gestures from video input in real-time. By accurately interpreting ISL gestures from video input, the system aims to enhance communication accessibility for individuals who use ISL.

Key components of the project objective include:
-	Accuracy: Developing algorithms and models that achieve high accuracy in recognizing ISL gestures, allowing for reliable interpretation of sign language communication.
-	Real-time Performance: Implementing the ISLR system to process video input in real-time, enabling seamless and instantaneous recognition of ISL gestures during communication.
-	Inclusivity: Foster inclusivity and equal access to information, education, employment, and social opportunities for the deaf and hard of hearing community by bridging communication barriers through ISL recognition technology.


## Problem Analysis

The project revolves around developing an Indian Sign Language Recognition (ISLR) system capable of interpreting ISL gestures from video input in real-time, addressing the complexities and variability inherent in sign language communication.

- Gesture Variability: Sign languages rely on visual and spatial components like hand shapes, movements, and facial expressions, leading to significant variability in gestures. This variability poses a challenge in developing robust recognition algorithms.
- Data Limitations: Limited availability of standardized datasets and annotated data for Indian Sign Language hampers the training process. Existing datasets often suffer from small sample sizes and biases, hindering the generalization capabilities of machine learning models.
- Real-Time Processing: Achieving real-time recognition is essential for seamless communication between users. This requires optimizing algorithms and efficiently utilizing computational resources to process video input promptly.

## Software Requirements

- Python
-	OpenCV
-	TensorFlow
-	OS
-	Matplotlib
-	NumPy
-	Scikit-Learn

NOTE: Also need a webcam for data acquisition

## Design

![image](https://github.com/Riddhi9570/Indian-Sign-Language-Recognition/assets/72887868/8b37f57e-6e61-421a-8e6f-27c87a980e72)

Fig. Workflow - CNN Model Layers


## Dataset Collection

![image](https://github.com/Riddhi9570/Indian-Sign-Language-Recognition/assets/72887868/e7bc6c3b-d067-43f3-af40-984d8c6c3886)

Fig. Dataset Collection (Digits)


![image](https://github.com/Riddhi9570/Indian-Sign-Language-Recognition/assets/72887868/6f4f22fa-8135-4e1b-8e09-e9acb846c5c2)

Fig. Dataset Collection (Alphabets)


## Experimental Result

![image](https://github.com/Riddhi9570/Indian-Sign-Language-Recognition/assets/72887868/0374a434-2584-42d0-8977-5fc08a9e72d0)

Fig. Convolutional Neural Network Model – Sequential Outcomes


![image](https://github.com/Riddhi9570/Indian-Sign-Language-Recognition/assets/72887868/09697426-4e0e-4363-b649-1f374006f110)

Fig. Model Epochs

![image](https://github.com/Riddhi9570/Indian-Sign-Language-Recognition/assets/72887868/7ad53680-52c3-49bd-bcc5-fbd92ea469bc)

Fig. Experimental Result 1 (Digit)

Recognizing the digit ‘2’ as per the Indian Sign Language. In this example, the model has correctly identified the digit and given the output as “Two”.


![image](https://github.com/Riddhi9570/Indian-Sign-Language-Recognition/assets/72887868/303a3694-14b4-4d69-9175-e56e86305c57)

Fig. Experimental Result 2 (Alphabet)

Recognizing the character ‘B’ as per the Indian Sign Language. In this example, the model has correctly identified the character and given the output as ‘B’.


![image](https://github.com/Riddhi9570/Indian-Sign-Language-Recognition/assets/72887868/861355f9-f611-40d9-a5be-a33273967a2e)

Fig. Training and Validation Accuracy

The model has achieved a high accuracy of 99.18% and high validation accuracy of 98.54%.


![image](https://github.com/Riddhi9570/Indian-Sign-Language-Recognition/assets/72887868/6e47173e-dd5d-4ed7-a3f8-b16f78d00638)

Fig. Training and Validation Loss

The model has achieved a low loss of 2.17% and low validation loss of 4.39%.

## Conclusion and Future Scope

The development of an Indian Sign Language Recognition (ISLR) system using Convolutional Neural Networks (CNNs) marks a significant step towards fostering inclusivity and accessibility for individuals within the Indian Sign Language (ISL) community. By integrating computer vision techniques and machine learning algorithms, the ISLR system provides ISL users with an effective means of communication across various contexts, including education, employment, and social interactions.

The project began with an in-depth exploration of ISL theoretical foundations and state-of-the-art ISL recognition technology through a comprehensive literature review. This informed the development and implementation of the ISLR system, which utilizes a computer vision-based approach to process live video feeds from webcams or integrated laptop cameras. Trained on a diverse dataset of ISL alphabets and digits, the CNN model exhibited promising results in accurately interpreting hand gestures captured in real-time.

The methodology encompassed data collection, preprocessing, model training, system integration, testing, and documentation, ensuring the robustness and reliability of the ISLR system. While the project represents a significant advancement, challenges such as dataset availability, environmental variability, and model generalization remain areas for improvement and future research.

Moving forward, future research and development efforts could focus on expanding and diversifying the dataset, integrating user feedback for personalized adaptation, exploring multimodal fusion approaches, and conducting real-world deployment trials to evaluate effectiveness and usability.

In conclusion, the ISLR project holds immense promise in enhancing accessibility and inclusivity for the ISL community. Leveraging technology to break down communication barriers, the ISLR system exemplifies the potential for innovation to promote understanding and connectivity in society. As efforts continue to refine and optimize the system, its impact in real-world scenarios is expected to grow significantly.

## Contributors

- [Kumar Yash](https://github.com/kumaryash18)
- [Aryan](https://github.com/AryanYuva)
- [Riddhi](https://github.com/riddhi9570)
