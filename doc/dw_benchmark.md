# Depthwise benchmark

具体调度过程在`src/depthwise/depthwise_gen.cc`中, 

#### X86平台(Intel® Core™ i5-4570 CPU @ 3.20GHz × 4, 8G DDR3)

关闭除depthwise之外的所有算子的注册, 调用benchmark测试程序之后的结果如下:

- 单线程

AutoKernel(运行10次取均值)

```
mobilenetv1  min =   37.65 ms   max =   37.91 ms   avg =   37.72 ms
mobilenetv2  min =   46.59 ms   max =   46.73 ms   avg =   46.65 ms
mobilenetv3  min =   38.40 ms   max =   38.69 ms   avg =   38.49 ms
```

Tengine(运行10次取均值)

```
mobilenetv1  min =   28.63 ms   max =   28.94 ms   avg =   28.71 ms
mobilenetv2  min =   33.01 ms   max =   34.12 ms   avg =   33.18 ms
mobilenetv3  min =   34.21 ms   max =   34.72 ms   avg =   34.36 ms
```

- 4线程

AutoKerne(运行10次取均值)

```
mobilenetv1  min =   36.21 ms   max =   36.48 ms   avg =   36.36 ms
mobilenetv2  min =   44.71 ms   max =   45.37 ms   avg =   45.04 ms
mobilenetv3  min =   38.67 ms   max =   38.93 ms   avg =   38.79 ms
```

Tengine(运行10次取均值)

```
mobilenetv1  min =   16.21 ms   max =   24.18 ms   avg =   17.20 ms
mobilenetv2  min =   24.49 ms   max =   31.41 ms   avg =   25.40 ms
mobilenetv3  min =   30.09 ms   max =   45.67 ms   avg =   31.95 ms
```

