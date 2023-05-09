<!DOCTYPE html>
<html>
  <body>
    <h1>Mammogram Analysis using Deep Learning</h1>
    <p>Breast cancer is a leading cause of death among women worldwide. Early detection through mammogram screening has been shown to significantly improve survival rates. Deep learning models have emerged as a promising approach for mammogram image classification, with various architectures, such as Densenet, CNN, VGG16, VGG19, Xception, and Mobilenet, being widely used for feature extraction. This repository includes the implementations of these models.</p>
    <h2>DenseNet Model</h2>
    <img src="imgs/densenet_arch.png" alt="GAN Image">
    <p style="text-align: center;">Figure 1 : Densenet Architecture</p>
    <p>DenseNet169 is a powerful convolutional neural network architecture that has shown impressive results in image classification tasks. It has been used in various applications, including medical imaging, where it has shown promising results in detecting and diagnosing diseases such as breast cancer.In mammography, DenseNet169 can be used for feature extraction, which involves using the pre-trained model to extract a set of features from the mammography images. These features can then be used as inputs to train a machine learning classifier to distinguish between benign and malignant breast tumors.</p>
    <h2>InceptionResNet V2 Model</h2>
    <img src="imgs/inet.png" alt="GAN Image">
    <p style="text-align: center;">Figure 2 : InceptionResNet V2 Architecture</p>
    <p>InceptionResNetV2 is a deep neural network architecture that combines the Inception architecture with residual connections, resulting in a more efficient and accurate model.To use InceptionResNetV2 for feature extraction in mammography images, the first step is to create a model using pre-trained weights on ImageNet. The model is then used to extract features from the mammography images by passing them through the model and capturing the output of the final convolutional layer. These features arethen   used   as   inputs   to   train  a  machine  learning  classifier  to  distinguish between  benign  and  malignant  breast  tumors.</p>
     <h2>MobileNet Model</h2>
    <img src="imgs/mnet.png" alt="GAN Image">
    <p style="text-align: center;">Figure 3 : MobileNet V2 Architecture</p>
    <p>MobileNet is a lightweight deep neural network architecture that has been specifically designed for mobile devices with limited computational resources. This architecture achieves a good trade-off between accuracy and complexity, making it suitable for applications where computational resources are limited, such as mobile devices or embedded systems.
To use MobileNet for feature extraction in mammography images, the first step is to create a model using pre-trained weights on ImageNet. The model is then used to extract features from the mammography images by passing them through the model and capturing the output of the final convolutional layer. These features are then used as inputs to train a machine learning classifier to distinguish between benign and malignant breast tumors.</p>
    <h2>VGG19 Model</h2>
    <img src="imgs/vgg19.png" alt="GAN Image">
    <p style="text-align: center;">Figure 4 : VGG19 Architecture</p>
    <p>The VGG19 model is a popular convolutional neural network architecture that has been widely used for image classification tasks. Itconsists    of    19    layers,    including    16 convolutional layers and 3 fully connected  layers.In this context, the VGG19 model is being used for feature extraction in mammography images. The model is pre-trained on the ImageNet dataset, which contains millions of images of various classes. By using transfer learning, the pre-trained VGG19 model can be used to extract meaningful features from the mammography images.The extracted features are then fed into various classifiers such as KNeighborsClassifier(), SVC(), RandomForestClassifier(), AdaBoostClassifier(), and XGBClassifier() to determine if the image is benign or malignant. These classifiers are trained on the extracted features and then used to predict the class of the test images.</p>
     <h2>InceptionResNet V2 Model</h2>
    <img src="imgs/inet.png" alt="GAN Image">
    <p style="text-align: center;">Figure 2 : InceptionResNet V2 Architecture</p>
    <p>InceptionResNetV2 is a deep neural network architecture that combines the Inception architecture with residual connections, resulting in a more efficient and accurate model.To use InceptionResNetV2 for feature extraction in mammography images, the first step is to create a model using pre-trained weights on ImageNet. The model is then used to extract features from the mammography images by passing them through the model and capturing the output of the final convolutional layer. These features arethen   used   as   inputs   to   train  a  machine  learning  classifier  to  distinguish between  benign  and  malignant  breast  tumors.</p>
    <h2>Convolutional Neural Networks (CNN)</h2>
    <img src="imgs/malware_cnn.png" alt="CNN Image">
    <p style="text-align: center;">Figure 2 : CNN Architecture</p>
    <p>Convolutional Neural Networks (CNN) is a deep learning model used for image classification, and is another model used for malware analysis in this repository.The CNN model used for this project consists of several convolutional layers, followed by max pooling layers and fully connected layers. The model is trained on the dataset using backpropagation and gradient descent to minimize the cross-entropy loss.The code for this model can be found in the CNN folder.</p>
    <h2>Random Forest</h2>
    <img src="imgs/malware_rf.png" alt="RF Image">
    <p style="text-align: center;">Figure 3 : Random Forest Architecture</p>
    <p>The Random Forest model used for this project consists of multiple decision trees, each trained on a subset of the dataset. The model is trained on the dataset using the Random Forest algorithm, which generates predictions by aggregating the predictions of multiple decision trees.</p>
    <h2>Support Vector Machine (SVM)</h2>
    <img src="imgs/malware_svm.png" alt="SVM Image">
    <p style="text-align: center;">Figure 4 : SVM Architecture</p>
    <p>In the DL-SVM classifier we use three models for malware classification: MLP-SVM, GRU-SVM, and CNN-SVM. MLP-SVM combines a multilayer perceptron (MLP) neural network with a  SVM classifier and similarly the other models.In all three models, the dataset is divided into training and testing sets, and the model is trained using the training set. The model is then evaluated on the testing set using metrics such as accuracy, precision, and recall.</p>
    <h1>Results</h1>
    <h3>GAN Results</h3>
    <p>The exisiting malware samples are changed by adding noise and certain parameters.These samples are then tested against various models to test the model's capabilities, the parameters such as LR and Optimizer can also be changed to better underastand the functioning of the model.</p>
    <p>Index:<br>Blue : RandomForest<br>Pink: Logisitic Regression<br>Yellow: Decision Tree<br>White: MultiLayerPerceptron</p>
    <p>Detector Loss </p>
    <img src="imgs/gan_result1.png" alt="GAN Image">
    <p>Generator Loss </p>
    <img src="imgs/gan_result2.png" alt="GAN Image">
    <h3>CNN Results</h3>
    <p>Accuracy of the model</p>
    <img src="imgs/results_cnn.png" alt="CNN Image">
    <h3>SVM Results</h3>
    <p>Accuracy of the 3 models</p>
    <img src="imgs/results_svm.png" alt="CNN Image">
    <h3>Random Forest Results</h3>
    <p>Accuracy in percentages</p>
    <img src="imgs/results_rf.png" alt="CNN Image">
  </body>
</html>
