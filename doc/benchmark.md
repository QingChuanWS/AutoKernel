## Benchmark

Benchmark 是评估AutoKernel算子在目标硬件平台网络模型运行速度的简单途径，只依赖于网络结构（xxx_benchmark.tmfile）即可。

目前由于Tengine框架中的卷积操作可能会调用winograd算法对卷积进行加速, 在测试之前需要手动关闭winograd算子, 以保证数据的准确性.

具体措施:

将`path/to/your/Tengine/src/dev/cpu/op/conv/x86/conv_kernel_x86.c`中的

```c
static int winograd_support(struct conv_param* param, int in_h, int in_w)
{
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int input_chan = param->input_channel;
    int output_chan = param->output_channel;
    int group = param->group;

    if (in_h <= 10 && in_w <= 10)
        return 0;

    if (group != 1 || kernel_h != 3 || kernel_w != 3 || stride_h != 1 || stride_w != 1 || dilation_h != 1 ||
        dilation_w != 1 || input_chan < 16 || output_chan < 16 || output_chan % 16)
        return 0;

    return 0; //TEST AUTOKERNEL --> RAW RETURN 1
}
```

然后对tengine进行重新编译即可关闭winograd进行测试.

同时注意在测试AutoKernel的im2col算子时, 需要关闭其他Autokernel算子, 以保证测试结果.

具体措施:

修改`Autokernel\autokernel_plugin\src\plugin_init.cpp`如以下形式:

```cpp
  1 #include <stdio.h>
  2 #include "pool/pool.h"
  3 #include "direct_conv/direct_conv.h"
  4 #include "im2col_conv/im2col_conv.h"
  5 #include "fc/fc.h"
  6 #include "softmax/softmax.h"
  7 
  8 extern "C" int autokernel_plugin_init(void)
  9 {
 10     /* register halide operator */
 11 //    RegisterAutoKernelSoftmax();
 12 //    RegisterAutoKernelFc();
 13 //    RegisterAutoKernelPool();
 14     RegisterAutoKernelDirect_conv();
 15     RegisterAutoKernelIm2col_conv();
 16     printf("AutoKernel plugin inited\n");
 17     return 0;
 18 }
```

然后重新编译Autokernel即可.

#### X86平台(Intel® Core™ i5-4570 CPU @ 3.20GHz × 4, 8G DDR3)

采用AutoKernel im2col替换Tengine原生的im2col算子后的benchmark结果:

单线程情况下的测试数据:

Autokernel算子:

```
/workspace/AutoKernel/autokernel_plugin/build# ./tests/tm_benchmark -r 10
start to run register cpu allocator
loop_counts = 10
num_threads = 1
power       = 0
AutoKernel plugin inited
function:autokernel_plugin_init executed
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   13.35 ms   max =   21.02 ms   avg =   15.09 ms
     	 mobilenetv1  min =   22.20 ms   max =   23.41 ms   avg =   22.60 ms
         mobilenetv2  min =   34.80 ms   max =   35.93 ms   avg =   35.19 ms
         mobilenetv3  min =   49.37 ms   max =   51.87 ms   avg =   49.87 ms
        shufflenetv2  min =   10.74 ms   max =   12.58 ms   avg =   11.36 ms
            resnet18  min =   45.45 ms   max =   50.28 ms   avg =   46.24 ms
            resnet50  min =   87.82 ms   max =   92.12 ms   avg =   88.73 ms
           googlenet  min =   88.92 ms   max =   91.05 ms   avg =   89.70 ms
         inceptionv3  min =  175.67 ms   max =  180.32 ms   avg =  177.02 ms
               vgg16  min =  344.61 ms   max =  350.54 ms   avg =  347.90 ms
                mssd  min =   41.93 ms   max =   44.53 ms   avg =   42.54 ms
          retinaface  min =   27.76 ms   max =   29.26 ms   avg =   28.12 ms
         yolov3_tiny  min =   77.85 ms   max =   79.07 ms   avg =   78.38 ms
      mobilefacenets  min =   13.06 ms   max =   15.18 ms   avg =   13.54 ms
ALL TEST DONE
```

Tengine原生im2col算子:

```
/workspace/AutoKernel/autokernel_plugin/build# ./tests/tm_benchmark -r 10
start to run register cpu allocator
loop_counts = 10
num_threads = 1
power       = 0
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   19.13 ms   max =   19.29 ms   avg =   19.21 ms
         mobilenetv1  min =   28.44 ms   max =   28.68 ms   avg =   28.55 ms
         mobilenetv2  min =   32.58 ms   max =   33.45 ms   avg =   32.77 ms
         mobilenetv3  min =   33.99 ms   max =   34.38 ms   avg =   34.14 ms
        shufflenetv2  min =    9.60 ms   max =    9.83 ms   avg =    9.67 ms
            resnet18  min =   92.14 ms   max =   92.91 ms   avg =   92.43 ms
            resnet50  min =  175.89 ms   max =  177.07 ms   avg =  176.33 ms
           googlenet  min =  122.19 ms   max =  122.66 ms   avg =  122.37 ms
         inceptionv3  min =  307.86 ms   max =  309.10 ms   avg =  308.32 ms
               vgg16  min =  845.51 ms   max =  848.38 ms   avg =  846.59 ms
                mssd  min =   59.68 ms   max =   60.27 ms   avg =   59.90 ms
          retinaface  min =   16.72 ms   max =   17.08 ms   avg =   16.90 ms
         yolov3_tiny  min =  146.77 ms   max =  148.50 ms   avg =  147.31 ms
      mobilefacenets  min =   13.85 ms   max =   14.26 ms   avg =   13.97 ms
ALL TEST DONE
```

4线程情况下的测试数据:

Autokernel算子:

```shell
start to run register cpu allocator
loop_counts = 10
num_threads = 4
power       = 0
AutoKernel plugin inited
function:autokernel_plugin_init executed
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   13.80 ms   max =   18.38 ms   avg =   14.39 ms
         mobilenetv1  min =   22.56 ms   max =   27.01 ms   avg =   23.24 ms
         mobilenetv2  min =   34.14 ms   max =   37.50 ms   avg =   34.79 ms
         mobilenetv3  min =   48.29 ms   max =   50.40 ms   avg =   48.80 ms
        shufflenetv2  min =   10.38 ms   max =   13.30 ms   avg =   10.90 ms
            resnet18  min =   45.22 ms   max =   50.13 ms   avg =   46.19 ms
            resnet50  min =   87.31 ms   max =   91.22 ms   avg =   88.81 ms
           googlenet  min =   89.05 ms   max =   91.96 ms   avg =   89.73 ms
         inceptionv3  min =  173.55 ms   max =  178.29 ms   avg =  174.89 ms
               vgg16  min =  314.51 ms   max =  349.58 ms   avg =  322.53 ms
                mssd  min =   41.83 ms   max =   45.07 ms   avg =   42.70 ms
          retinaface  min =   28.64 ms   max =   29.65 ms   avg =   28.95 ms
         yolov3_tiny  min =   77.55 ms   max =   79.11 ms   avg =   78.22 ms
      mobilefacenets  min =   13.39 ms   max =   16.65 ms   avg =   14.11 ms
ALL TEST DONE
```

Tengine原生im2col算子:

```shell
/workspace/Tengine/benchmark# ./../build/benchmark/tm_benchmark -r 10 -t 4
start to run register cpu allocator
loop_counts = 10
num_threads = 4
power       = 0
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   10.61 ms   max =   23.76 ms   avg =   13.56 ms
         mobilenetv1  min =   16.32 ms   max =   19.24 ms   avg =   16.81 ms
         mobilenetv2  min =   24.19 ms   max =   31.02 ms   avg =   25.06 ms
         mobilenetv3  min =   29.92 ms   max =   37.77 ms   avg =   31.57 ms
        shufflenetv2  min =    6.75 ms   max =   17.37 ms   avg =    8.73 ms
            resnet18  min =   45.13 ms   max =   47.47 ms   avg =   45.74 ms
            resnet50  min =   81.08 ms   max =   95.63 ms   avg =   83.26 ms
           googlenet  min =   82.57 ms   max =   86.65 ms   avg =   83.84 ms
         inceptionv3  min =  157.95 ms   max =  173.54 ms   avg =  161.76 ms
               vgg16  min =  336.36 ms   max =  352.59 ms   avg =  341.88 ms
                mssd  min =   35.64 ms   max =   47.16 ms   avg =   37.20 ms
          retinaface  min =   12.80 ms   max =   26.71 ms   avg =   14.97 ms
         yolov3_tiny  min =   73.33 ms   max =   77.30 ms   avg =   74.52 ms
      mobilefacenets  min =    9.68 ms   max =   18.95 ms   avg =   11.03 ms
ALL TEST DONE
```

