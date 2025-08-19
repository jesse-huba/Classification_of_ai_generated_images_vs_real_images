# Classification_of_ai_generated_images_vs_real_images
📌 Project Overview

This project implements a deep learning pipeline to classify AI-generated images versus real images. Using a ResNet50-based CNN with transfer learning, the model leverages dataset normalization, augmentation, and robust evaluation metrics to ensure high accuracy and generalization.

In today’s digital landscape, the rise of generative AI tools has made it increasingly difficult to distinguish between synthetic and authentic media. This project addresses that challenge by training a model capable of detecting subtle differences between real and AI-generated visuals.

The notebook covers the entire workflow — from dataset preparation and preprocessing to model training, evaluation, and visualization. By incorporating GPU acceleration, early stopping, and learning rate scheduling, the implementation is both efficient and scalable.

This project serves as a practical foundation for applications in digital forensics, misinformation detection, and content authenticity verification.

📎**Dataset:**

[CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data): Real and AI-Generated Synthetic Images: CIFAKE is a dataset that contains 60,000 synthetically-generated images and 60,000 real images (collected from CIFAR-10).

This dataset is available for free in kaggle. ![](https://storage.googleapis.com/kaggle-datasets-images/3041726/5227744/64b8381e45aef2060808e31584ed141f/dataset-cover.png?t=2023-03-24-13-29-07)

🛠️**Tools and Libraries:**

Used Tensorflow for model training, the model is trained on T4 GPU.

🚀**DataPreparation:**

- Data augmentation is integrated into the training pipeline to enhance the diversity of the dataset.
- The augmentation process includes random horizontal and vertical flipping of the images. This simulates various orientations and perspectives, making the model more robust to such variations in real-world scenarios.
- The images are resized to a uniform size of 32x32 pixels
- Normalization is applied to the images by scaling pixel values to the range [0, 1], which helps in faster computation and more efficient training of the model.
- The dataset is then split into three subsets: 70% for training, 20% for validation, and 10% for testing. This ensures a good balance between training data and data used for model validation and testing.

**Model Architecture:**

- **Pre-trained ResNet50 Base:** The ResNet50 model, known for its deep architectureand residual connections that facilitate training of deeper networks, is utilized with weights pre-trained on ImageNet. This provides a strong feature extraction foundation.
- **Customization for Specific Task:** The ResNet50 base is modified to suit the specific task of distinguishing between real and AI-generated images. This involves adjusting the input shape to (32, 32, 3) to match the dataset's image dimensions.
- **Inclusion of Additional Layers:** Following the ResNet50 base, the model includes additional layers to further process the extracted features:
- A Flatten layer to convert the 2D feature maps into a 1D vector.

- Dense layers with 256 and 128 units, both utilizing ReLU activation, to learn higher-level representations and patterns in the data.
- Batch normalization is applied after the first dense layer to stabilize and speed up training.
- A dropout layer with a rate of 0.5 is included to reduce the risk of overfitting by randomly setting a fraction of input units to 0 at each update during training.
- The final dense layer with a single unit and sigmoid activation function is tailored for binary classification.
- **Adapting to Binary Classification:** The model's final layer uses sigmoid activation, making it suitable for binary classification tasks, like distinguishing between two classes of images (real vs. AI-generated).
- ![](RackMultipart20231125-1-5qp7kr_html_a4e4f2610fa15366.png)

**Model Training and Evaluation:**

- The model learns from pictures to recognize patterns. It's trained with two sets of pictures: training and validation.
- Training pictures help it learn, while validation pictures check how well it's learning.
- The model runs10 epochs , each time improving.
- There's an 'early stopping' tool that stops training if the model stops getting better after trying 3 more times.
- This helps to pick the best learning moment. After training, the model is tested with new pictures to see how accurate it is. This process helps the model to recognize images well.
- The model is evaluated using a test set, a separate group of images it hasn't seen before.
- This test checks how accurately the model can identify and classify these new images.
