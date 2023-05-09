<!DOCTYPE html>
<html>
  <body>
    <h1>Mammogram Analysis using Deep Learning</h1>
    <p>Breast cancer is a leading cause of death among women worldwide. Early detection through mammogram screening has been shown to significantly improve survival rates. Deep learning models have emerged as a promising approach for mammogram image classification, with various architectures, such as Densenet, CNN, VGG16, VGG19, Xception, and Mobilenet, being widely used for feature extraction. This repository includes the implementations of these models.</p>
    <h2></h2>
    <img src="imgs/malware_gan.png" alt="GAN Image">
    <p style="text-align: center;">Figure 1 : GAN Architecture</p>
    <p>Generative Adversarial Networks (GAN) is a deep learning model used for generating synthetic data, and is one of the models used for malware analysis in this repository. The          code for this model can be found in the GAN folder.The idea is to use a generative adversarial network (GAN) based algorithm to generate adversarial malware examples, which are able to bypass black-box machine learning based detection models.Figure 1 shows the adversarial malware generator’s training architecture.</p>
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
