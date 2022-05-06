# Deep Learning with Azure ML and ML.NET (Global Azure 2022 Torino Conference)

In this repository you can find the slides and demo for **Deep Learning with Azure ML and ML.NET** session, presented (in Italian) at [Global Azure 2022 Torino Conference](https://globalazuretorino.welol.it/) on May 7th, 2022.

Abstract:

Machine Learning and Deep Learning are more and more utilized at all levels, from embedded devices to web browsers. In this session, we will see how we can leverage our .NET expertise and tools to develop applications that utilize AI models: with a practical approach, we explore a different way to use the most common machine learning/deep learning frameworks to train models on Azure and score them using ML.NET.

Speakers:

- [Clemente Giorio](https://www.linkedin.com/in/clemente-giorio-03a61811/) (Deltatre, Microsoft MVP)
- [Gianni Rosa Gallina](https://www.linkedin.com/in/gianni-rosa-gallina-b206a821/) (Deltatre, Microsoft MVP)

---

## Setup local environment

Software requirements:

- Windows 10 21H2 or Windows 11
- Visual Studio 2022
- .NET 6 SDK
- Visual Studio Code
- Python 3.8.x

To setup a local copy, just clone the repository.  
You can find the training notebook in the `notebooks` folder, while training scripts and scoring demo app are in `src` folder.

### Ball & Person Detector Onnx Demo

Before executing the application, you need to:

- place all the images you want to process in the `src\BallDetectorOnnxDemo\Deltatre.BallDetector.Onnx.Demo.CLI\SampleData` folder
- download pre-trained YOLOv5 ONNX models from [Ultralytics GitHub repository](https://github.com/ultralytics/yolov5/releases) and place them in the `src\BallDetectorOnnxDemo\Deltatre.BallDetector.Onnx.Demo.YoloModel\Assets\ModelWeights` folder

To start the scoring application, set the `Deltatre.BallDetector.Onnx.Demo.CLI` project as *Startup project*, and launch a debug session. It will load the configured YOLOv5 pre-trained model and score all the images in the `SampleData` folder. All results will be saved in the `Outputs` folder.

If you want to customize the folder where to look for images to process and where to store the results, you can edit the `Program.cs` file and change the following lines:

```csharp
var datasetRelativePath = @"../../../";
string datasetPath = GetAbsolutePath(datasetRelativePath);
var imagesFolder = Path.Combine(datasetPath, "SampleData");
var outputFolder = Path.Combine(datasetPath, "SampleData", "Outputs");
```

### Model Fine Tuning Demo

**TODO: TO BE UPDATED**  
**ADD REQUIREMENTS FOR AZURE**

Before executing the training scripts, you need to place your set of images you want to use in a dataset folder and change paths accordingly. The dataset folder must be setup as following:

There you have a `Training` folder and a `Test` folder. Within each folder, you may put images in different sub-folders, representing the "label" of the contained images.

An example of the dataset folder structure is:

```bash
└── src
    └── ModelFineTuningDemo
        └── SampleData
            ├── MLModels
            ├── Training
            │   ├── Class1
            │   │   ├── image1.png
            │   │   ├── image2.png
            │   │   ├── ...
            │   │   └── imageN.png
            │   ├── Class2
            │   │   ├── image1.png
            │   │   ├── image2.png
            │   │   ├── ...
            │   │   └── imageN.png
            │   ├── ...
            │   └── ClassN
            │       ├── image1.png
            │       ├── image2.png
            │       ├── ...
            │       └── imageN.png
            └── Test
                ├── Class1
                │   ├── image1.png
                │   ├── image2.png
                │   ├── ...
                │   └── imageN.png
                ├── Class2
                │   ├── image1.png
                │   ├── image2.png
                │   ├── ...
                │   └── imageN.png
                ├── ...
                └── ClassN
                    ├── image1.png
                    ├── image2.png
                    ├── ...
                    └── imageN.png
```

For the best results, each class should contain almost the same number of representative samples, and usually, the more samples you have, the more the quality of the fine-tuned model improves. But you need to test and verify on your own dataset the metrics and decide the proper actions to take to fulfill your requirements.

### Export YOLOv5 in ONNX model format

**TODO: TO BE UPDATED, as shown in the session**

Taking into account that the training times for YOLOv5n/s/m/l/x are 1/2/4/6/8 days on an [NVIDIA V100 GPU](https://www.nvidia.com/en-us/data-center/a100/), the fastest way is to download pre-trained models from PyTorch and convert them into the [ONNX](https://onnx.ai/) model format.

The first step is to clone [Ultralytics](https://ultralytics.com/)'s YOLOv5 repository.

```ps
git clone https://github.com/ultralytics/yolov5
cd yolov5
```

Optionally, but suggested, create and activate a [Python >= 3.7.0 Virtual Environment](https://docs.python.org/3/library/venv.html). Using Virtual Environments allows you to avoid installing Python packages globally which could break system tools or other projects. 

```ps
python -m pip install --upgrade pip
python -m pip install --user virtualenv
python -m venv .venv
.venv\Scripts\activate
```

>For this demo we used Python 3.9.6  

Install all the requirements.

```ps
python -m pip install --upgrade pip
pip install wheel
```

>If you have an NVIDIA GPU we suggest installing the latest version of Pytorch with CUDA support, otherwise the package with CPU-only support will be installed. 
>For more details please refer to the official [PyTorch](https://pytorch.org/) page.  
>In this demo, we used Pytorch 1.11.0, and TorchVision 0.12.0 compiled for CUDA 11.3. You can install them with:
>
>```ps
>pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
>```

Edit the file ***requirements.txt*** and uncomment the **onnx** package.

>If you have already installed PyTorch and TorchVision for CUDA, comment **torch** and **torchvision** packages.

Now install ***requirements.txt***

>```ps
pip install -r requirements.txt
```

Now we can export pre-trained models:

```ps
python export.py --weights yolov5n.pt --imgsz 640 640 --include onnx
python export.py --weights yolov5s.pt --imgsz 640 640 --include onnx
python export.py --weights yolov5m.pt --imgsz 640 640 --include onnx
python export.py --weights yolov5l.pt --imgsz 640 640 --include onnx
python export.py --weights yolov5x.pt --imgsz 640 640 --include onnx
python export.py --weights yolov5n6.pt --imgsz 1280 1280 --include onnx
python export.py --weights yolov5s6.pt --imgsz 1280 1280 --include onnx
python export.py --weights yolov5m6.pt --imgsz 1280 1280 --include onnx
python export.py --weights yolov5l6.pt --imgsz 1280 1280 --include onnx
python export.py --weights yolov5x6.pt --imgsz 1280 1280 --include onnx
```

>It is not required to export all the pre-trained models. Please, check on the [YOLOv5 pre-trained model release page](https://github.com/ultralytics/yolov5/releases) for the ones that fit the requirements.
>We can specify the ***GPU id*** to use with the ***--device*** option. For example, if we want to export the pre-trained model YOLOv5x6 in ONNX format using the first CUDA GPU capable on our machine, we can run the following command:
>
>```ps
>python export.py --weights yolov5x6.pt --imgsz 1280 1280 --include onnx --device 0
>```
>
>For visualizing exported models we can use [**Netron** viewer](https://github.com/lutzroeder/netron).   
>We can install it with:
>
>```ps
>pip install netron
>```
>And use ***netron [FILE]*** to visualize the model. For example:
>
>```ps
>netron yolov5x6.pt
>```

## References and other useful links

### Ball Detector ONNX Demo

- <https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx>
- <https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/transforms>
- <https://towardsdatascience.com/mask-detection-using-yolov5-ae40979227a6>
- <https://dev.to/azure/onnx-no-it-s-not-a-pokemon-deploy-your-onnx-model-with-c-and-azure-functions-28f>
- <https://stackoverflow.com/questions/57264865/cant-get-input-column-name-of-onnx-model-to-work>
- <https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/inspect-intermediate-data-ml-net>
- <https://github.com/dotnet/machinelearning/blob/main/docs/code/VBufferCareFeeding.md>
- <https://github.com/dotnet/machinelearning/blob/main/src/Microsoft.ML.OnnxTransformer/OnnxTransform.cs>
- <https://stackoverflow.com/questions/64357642/how-to-load-image-from-memory-with-bitmap-or-byte-array-for-image-processing-in>
- <https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.imageestimatorscatalog.extractpixels?view=ml-dotnet>
- <https://stackoverflow.com/questions/70880362/transform-densetensor-in-microsoft-ml-onnxruntime>

### TensorFlow fine-tuning Demo

- <https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ImageClassification_Training>
- <https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification>
- <https://levelup.gitconnected.com/training-an-ml-net-image-classification-model-on-gpus-using-google-colab-ee40b38af7e5>

## License

---

Copyright (C) 2022 Deltatre.  
Licensed under [MIT license](./LICENSE).
