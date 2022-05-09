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
You can find the training notebook in the `notebooks` folder, while training scripts and scoring demo app are in `src` folder. Slides can be found in the `docs` folder.

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

To run the demo code, you need:

- a *Microsoft Azure subscription*, and to setup Azure Machine Learning resources you can refer to the [Quickstart: Create workspace resources you need to get started with Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources) tutorial.
- Yolov5 pre-trained models and training scripts from [Ultralytics repository](https://github.com/ultralytics/yolov5), cloned in `src/yolov5` folder.
- A Python 3.8.x environment to run notebook and scripts

    ```ps
    python -m pip install -U pip
    pip install wheel
    pip install azureml-sdk[notebooks]
    pip install click matplotlib numpy opencv-python
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    pip install -r src/yolov5/requirements.txt
    ```

Before executing the training scripts, you need to prepare and place the set of images you want to use in a dataset folder, and upload them to Azure ML. For labeling images, we used the open-source tool [CVAT](https://github.com/openvinotoolkit/cvat).

The dataset folder must follow the Yolov5 structure: there you have the `images` and `labels` folder. Within each folder, `train`, `valid`, and `test` sub-folders. Within each folder, then, you have to put images and their corresponding labels. Labels must be in the Yolov5 format, as described in [Yolov5 documentation](https://docs.ultralytics.com/tutorials/train-custom-datasets/).

An example of the dataset folder structure is:

```ps
└── yolov5_dataset
    ├── images
    │   ├── train
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   ├── ...
    │   │   └── imageN.jpg
    │   ├── val
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   ├── ...
    │   │   └── imageN.jpg
    │   └── test
    │       ├── image1.jpg
    │       ├── image2.jpg
    │       ├── ...
    │       └── imageN.jpg
    └── labels
        ├── train
        │   ├── image1.txt
        │   ├── image2.txt
        │   ├── ...
        │   └── imageN.txt
        ├── val
        │   ├── image1.txt
        │   ├── image2.txt
        │   ├── ...
        │   └── imageN.txt
        └── test
            ├── image1.txt
            ├── image2.txt
            ├── ...
            └── imageN.txt
```

For the best results, each class should contain almost the same number of representative samples, and usually, the more samples you have, the more the quality of the fine-tuned model improves. But you need to test and verify on your own dataset the metrics and decide the proper actions to take to fulfill your requirements.

### Export YOLOv5 in ONNX model format

Taking into account that the full training times for YOLOv5n/s/m/l/x are 1/2/4/6/8 days on an [NVIDIA V100 GPU](https://www.nvidia.com/en-us/data-center/a100/), the fastest way is to download pre-trained PyTorch models and convert them into the [ONNX](https://onnx.ai/) model format. The same, can be one for your fine-tuned model, as done in the demo notebook.

To export pre-trained models:

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

>It is not required to export all the pre-trained models. Please, check on the [YOLOv5 pre-trained model release page](https://github.com/ultralytics/yolov5/releases) for the ones that fit your requirements.
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

- <https://github.com/deltatrelabs/deltatre-net-conf-2022-mlnet>
- <https://github.com/ultralytics/yolov5/releases>
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
- <https://azure.microsoft.com/en-us/services/machine-learning/>
- <https://cvat.org>
- <https://makesense.ai>
- <https://labelbox.com>
- <https://roboflow.com>

## License

---

Copyright (C) 2022 Deltatre.  
Licensed under [MIT license](./LICENSE).
