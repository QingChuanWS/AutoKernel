# Depthwise benchmark

具体调度过程在`src/depthwise/depthwise_gen.cc`中, 

#### X86平台(Intel® Core™ i5-4570 CPU @ 3.20GHz × 4, 8G DDR3)

关闭除depthwise之外的所有算子的注册, 调用benchmark测试程序之后的结果如下:

- 单线程(运行10次取平均值)

| model              | Autokernel   | Tengine      |
| ------------------ | ------------ | ------------ |
| squeezenet_v1.1    | 17.76 ms     | 17.56 ms     |
| **mobilenetv1**    | **32.32 ms** | **31.00 ms** |
| **mobilenetv2**    | **36.02 ms** | **36.13 ms** |
| **mobilenetv3**    | **35.25 ms** | **34.65 ms** |
| shufflenetv2       | 10.97 ms     | 9.84 ms      |
| resnet18           | 63.30 ms     | 63.37 ms     |
| resnet50           | 142.97 ms    | 143.06 ms    |
| googlenet          | 104.88 ms    | 105.19 ms    |
| inceptionv3        | 258.33 ms    | 264.63 ms    |
| vgg16              | 454.18 ms    | 457.19 ms    |
| **mssd**           | **62.69 ms** | **61.35 ms** |
| retinaface         | 12.94 ms     | 12.04 ms     |
| yolov3_tiny        | 118.01 ms    | 118.03 ms    |
| **mobilefacenets** | **16.06 ms** | **13.84 ms** |

- 4线程(运行10次取平均值)

| model              | Autokernel(Halide多线程) | AutoKernel(Tengine多线程) | Tengine      |
| ------------------ | ------------------------ | ------------------------- | ------------ |
| squeezenet_v1.1    | 18.04 ms                 | 10.62 ms                  | 10.66 ms     |
| **mobilenetv1**    | **31.48 ms**             | **19.40 ms**              | **18.09 ms** |
| **mobilenetv2**    | **36.71 ms**             | **29.05 ms**              | **27.56 ms** |
| **mobilenetv3**    | **35.80 ms**             | **31.93 ms**              | **31.80 ms** |
| shufflenetv2       | 11.18 ms                 | 9.64 ms                   | 9.07 ms      |
| resnet18           | 64.09 ms                 | 28.67 ms                  | 27.51 ms     |
| resnet50           | 144.37 ms                | 62.94 ms                  | 65.83 ms     |
| googlenet          | 104.79 ms                | 81.03 ms                  | 78.74 ms     |
| inceptionv3        | 260.25 ms                | 138.93 ms                 | 135.88 ms    |
| vgg16              | 450.64 ms                | 206.79 ms                 | 220.79 ms    |
| **mssd**           | **62.05 ms**             | **38.69 ms**              | **38.00 ms** |
| retinaface         | 14.43 ms                 | 11.55 ms                  | 11.53 ms     |
| yolov3_tiny        | 119.01 ms                | 59.60 ms                  | 59.95 ms     |
| **mobilefacenets** | **16.61 ms**             | **12.14 ms**              | **11.47 ms** |