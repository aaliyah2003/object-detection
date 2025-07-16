###  *🔍 Real-Time Custom Object Detection using TensorFlow 2.10*

<sub>A complete end-to-end object detection pipeline built with TensorFlow 2.10, Keras, and SSD MobileNet. This project allows you to train your own detection model on custom images, run real-time inference, and leverage GPU acceleration using CUDA and cuDNN.</sub>

#### *Features*

- <sub>Real-time object detection</sub>
- <sub>Custom model training using your own dataset</sub>
- <sub>TensorFlow 2 Object Detection API integration</sub>
- <sub>Environment setup guide with CUDA & cuDNN</sub>
- <sub>Training and evaluation scripts with live loss monitoring</sub>

#### *🛠️ Setup Instructions*

##### Step 1 : 🔁 Clone the Repository

<small>git clone https://github.com/your-username/your-repo.git
cd your-repo </small>

#### Step 2: 🐍 Create & Activate Virtual Environment

##### Create
<small>python -m venv objectdetection</small>

##### Activate
##### On Windows
<small>.\objectdetection\Scripts\activate</small>

##### On Linux/macOS
<small>source objectdetection/bin/activate</small>

#### Step 3: 📦 Install Dependencies

<small>python -m pip install --upgrade pip</small>

<small>pip install ipykernel</small>

<small>python -m ipykernel install --user --name=objectdetection</small>

#### Step 4: 📸 Image Collection & Dataset Preparation

1. Open Image Collection.ipynb to gather your training images.
2. Manually split the collected data into:
- <small>Tensorflow/workspace/images/train</small>
- <small>Tensorflow/workspace/images/test</small>

3. Ensure annotations and images are aligned correctly in both folders

#### Step 5: 🔧 GPU Setup (CUDA & cuDNN Configuration)
<sub>To leverage GPU acceleration during model training, ensure your system is properly configured with compatible versions of CUDA and cuDNN:</sub>

#### Step 6: ✅ Compatibility Requirements

- <small>TensorFlow Version: 2.10.0</small>
- <small>Required CUDA Version: 11.2</small>
- <small>Required cuDNN Version: 8.1</small>

##### 📥 Step-by-Step Installation
<small>a. Download:</small>

- <sub>[CUDA 11.2 (NVIDIA official site)](https://developer.nvidia.com/cuda-11.2.0-download-archive)</sub>

- <sub>[cuDNN 8.1 for CUDA 11.2 (NVIDIA official site)](https://developer.nvidia.com/rdp/cudnn-archive)</sub>

<small>b. After downloading:</small>
- <sub>Extract and copy contents (bin, include, lib folders) and paste it into your local CUDA directory (e.g., C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2)</sub>

<small>c. Set the corresponding CUDA and cuDNN paths in Environment Variables</small>

<small>d. Run the TensorFlow verification script — ensure it ends with an OK status.</small>

#### Step7: 🏋️ Model Training

Run the following command in your terminal (inside activated virtual environment):

python Tensorflow/models/research/object_detection/model_main_tf2.py \
--model_dir=Tensorflow/workspace/models/my_ssd_mobnet \
--pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config \
--num_train_steps=2000

###### 💡 Note: If you encounter compatibility errors during training, try reinstalling pycocotools and ensure all dependencies match TensorFlow 2.10 compatibility.

####  Step 8: 📈 Evaluation & TensorBoard (Optional)
<small>To evaluate your trained model:</small>

tensorboard --logdir=Tensorflow/workspace/models/my_ssd_mobnet/eval

#### Step 9: 👀 Real-Time Detection

<small>Once your model is trained and exported:</small>

- <sub>Load the model in your detection script or notebook (Training.ipynb)</sub>

- <sub>Start detecting objects in real-time via webcam or image feed.</sub>


## ⚠️ Final Note

<SMALL>Due to hardware limitations on my local machine, I was unable to fully train and deploy the model. However, the complete pipeline and guidance have been documented thoroughly in this README for seamless setup and training on capable systems (preferably with a compatible NVIDIA GPU).</SMALL>

<SUB>If you'd like access to the dataset used for this project, feel free to reach out via email:</SUB>

<SUB>📩 **aaliyahshaikh10@gmail.com**</SUB>

<SMALL>You're welcome to train the model yourself using the steps above and contribute improvements or feedback via GitHub. Happy Building! 🚀</SMALL>

