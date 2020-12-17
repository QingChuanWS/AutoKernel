# Depthwise benchmark

具体调度过程在`src/depthwise/depthwise_gen.cc`中, 

#### X86平台(Intel® Core™ i5-4570 CPU @ 3.20GHz × 4, 8G DDR3)

关闭除depthwise之外的所有算子的注册, 调用benchmark测试程序之后的结果如下:

- 单线程(运行10次取平均值)

| param                           | Autokernel dw_con (us)  | Tengine(us) |
| ------------------------------- | ----------------------- | ----------- |
| c = 32, input_size = 112, s = 1 | 2164.60 (im2col : 1560) | 2604.80     |
| c = 64, input_size = 112, s = 2 | 1224.00 (im2col : 998)  | 4082.20     |
| c = 128, input_size = 56, s = 1 | 2133.30 (im2col : 1693) | 1167.40     |
| c = 128, input_size = 56, s = 2 | 682.60 (im2col : 587)   | 639.90      |
| c = 256, input_size = 28, s = 1 | 1281.70 (im2col : 1120) | 386.90      |
| c = 256, input_size = 28, s = 2 | 553.80 (im2col : 486)   | 381.40      |
| c = 512, input_size = 14, s = 1 | 1056.50 (im2col : 931)  | 336.40      |
| c = 512, input_size = 14, s = 2 | 519.20 (im2col : 444)   | 190.10      |
| c = 1024, input_size = 7, s = 2 | 807.30 (im2col : 689)   | 156.90      |
