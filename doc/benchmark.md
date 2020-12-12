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
 14 //    RegisterAutoKernelDirect_conv();
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
start to run register cpu allocator
loop_counts = 10
num_threads = 1
power       = 0
affinity    = 255
AutoKernel plugin inited
function:autokernel_plugin_init executed
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   24.02 ms   max =   24.47 ms   avg =   24.32 ms
         mobilenetv1  min =   37.60 ms   max =   37.86 ms   avg =   37.69 ms
         mobilenetv2  min =   45.91 ms   max =   46.47 ms   avg =   46.17 ms
         mobilenetv3  min =   54.35 ms   max =   55.00 ms   avg =   54.64 ms
        shufflenetv2  min =   14.25 ms   max =   15.62 ms   avg =   14.45 ms
            resnet18  min =   97.46 ms   max =   98.07 ms   avg =   97.78 ms
            resnet50  min =  194.63 ms   max =  196.84 ms   avg =  195.47 ms
           googlenet  min =  127.62 ms   max =  127.85 ms   avg =  127.77 ms
         inceptionv3  min =  293.39 ms   max =  294.46 ms   avg =  293.82 ms
               vgg16  min =  773.04 ms   max =  802.17 ms   avg =  776.85 ms
                mssd  min =   68.20 ms   max =   69.38 ms   avg =   68.68 ms
          retinaface  min =   38.03 ms   max =   38.40 ms   avg =   38.21 ms
         yolov3_tiny  min =  161.75 ms   max =  162.76 ms   avg =  162.22 ms
      mobilefacenets  min =   19.07 ms   max =   19.21 ms   avg =   19.12 ms
ALL TEST DONE
```

Tengine原生im2col算子:

```
start to run register cpu allocator
loop_counts = 10
num_threads = 1
power       = 0
affinity    = 255
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   17.69 ms   max =   17.84 ms   avg =   17.75 ms
         mobilenetv1  min =   29.02 ms   max =   29.25 ms   avg =   29.11 ms
         mobilenetv2  min =   33.40 ms   max =   34.62 ms   avg =   33.66 ms
         mobilenetv3  min =   34.50 ms   max =   35.01 ms   avg =   34.73 ms
        shufflenetv2  min =    9.64 ms   max =   11.22 ms   avg =    9.96 ms
            resnet18  min =   63.03 ms   max =   63.75 ms   avg =   63.20 ms
            resnet50  min =  141.81 ms   max =  142.28 ms   avg =  142.03 ms
           googlenet  min =  104.80 ms   max =  106.44 ms   avg =  105.05 ms
         inceptionv3  min =  255.97 ms   max =  262.22 ms   avg =  257.11 ms
               vgg16  min =  457.50 ms   max =  461.69 ms   avg =  460.52 ms
                mssd  min =   60.28 ms   max =   62.13 ms   avg =   60.81 ms
          retinaface  min =   11.91 ms   max =   12.05 ms   avg =   11.97 ms
         yolov3_tiny  min =  118.89 ms   max =  123.09 ms   avg =  119.85 ms
      mobilefacenets  min =   13.88 ms   max =   14.41 ms   avg =   14.10 ms
ALL TEST DONE

```

4线程情况下的测试数据:

Autokernel算子:

```shell
start to run register cpu allocator
loop_counts = 10
num_threads = 1
power       = 0
affinity    = 255
AutoKernel plugin inited
function:autokernel_plugin_init executed
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   13.19 ms   max =   15.05 ms   avg =   13.71 ms
         mobilenetv1  min =   22.31 ms   max =   22.96 ms   avg =   22.64 ms
         mobilenetv2  min =   34.22 ms   max =   35.38 ms   avg =   34.77 ms
         mobilenetv3  min =   48.54 ms   max =   49.19 ms   avg =   48.77 ms
        shufflenetv2  min =   10.36 ms   max =   11.38 ms   avg =   10.83 ms
            resnet18  min =   44.86 ms   max =   45.81 ms   avg =   45.19 ms
            resnet50  min =   87.30 ms   max =   89.58 ms   avg =   88.09 ms
           googlenet  min =   88.93 ms   max =   91.22 ms   avg =   89.67 ms
         inceptionv3  min =  176.15 ms   max =  179.82 ms   avg =  177.63 ms
               vgg16  min =  342.26 ms   max =  353.38 ms   avg =  345.57 ms
                mssd  min =   41.54 ms   max =   43.71 ms   avg =   42.24 ms
          retinaface  min =   19.05 ms   max =   20.87 ms   avg =   19.36 ms
         yolov3_tiny  min =   78.27 ms   max =   82.57 ms   avg =   79.46 ms
      mobilefacenets  min =   13.12 ms   max =   15.15 ms   avg =   13.60 ms
ALL TEST DONE
```

Tengine原生im2col算子:

```
start to run register cpu allocator
loop_counts = 10
num_threads = 4
power       = 0
affinity    = 255
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =    9.51 ms   max =   17.25 ms   avg =   10.63 ms
         mobilenetv1  min =   16.14 ms   max =   19.97 ms   avg =   16.97 ms
         mobilenetv2  min =   23.81 ms   max =   29.23 ms   avg =   25.30 ms
         mobilenetv3  min =   30.04 ms   max =   37.72 ms   avg =   31.63 ms
        shufflenetv2  min =    6.79 ms   max =   18.63 ms   avg =    9.44 ms
            resnet18  min =   26.75 ms   max =   28.29 ms   avg =   27.24 ms
            resnet50  min =   61.71 ms   max =   74.52 ms   avg =   64.59 ms
           googlenet  min =   75.53 ms   max =   78.10 ms   avg =   76.56 ms
         inceptionv3  min =  129.89 ms   max =  137.94 ms   avg =  132.92 ms
               vgg16  min =  204.31 ms   max =  215.71 ms   avg =  207.06 ms
                mssd  min =   34.49 ms   max =   49.94 ms   avg =   36.50 ms
          retinaface  min =    8.43 ms   max =   16.82 ms   avg =   10.36 ms
         yolov3_tiny  min =   56.91 ms   max =   60.76 ms   avg =   57.86 ms
      mobilefacenets  min =    9.36 ms   max =   20.57 ms   avg =   12.39 ms
ALL TEST DONE
```

