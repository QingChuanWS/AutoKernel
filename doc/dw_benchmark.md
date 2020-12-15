# Depthwise benchmark

具体调度过程在`src/depthwise/depthwise_gen.cc`中, 

#### X86平台(Intel® Core™ i5-4570 CPU @ 3.20GHz × 4, 8G DDR3)

关闭除depthwise之外的所有算子的注册, 调用benchmark测试程序之后的结果如下:

- 单线程(运行10次取平均值)

| param                           | Autokernel dw_con (us) | Tengine() |
| ------------------------------- | ---------------------- | --------- |
| c = 32, input_size = 112, s = 1 | 3193.10                | 2604.80   |
| c = 64, input_size = 112, s = 2 | 1646.00                | 4082.20   |
| c = 128, input_size = 56, s = 1 | 3172.40                | 1167.40   |
| c = 128, input_size = 56, s = 2 | 970.10                 | 639.90    |
| c = 256, input_size = 28, s = 1 | 1906.80                | 386.90    |
| c = 256, input_size = 28, s = 2 | 597.30                 | 381.40    |
| c = 512, input_size = 14, s = 1 | 1178.10                | 336.40    |
| c = 512, input_size = 14, s = 2 | 439.30                 | 190.10    |
| c = 1024, input_size = 7, s = 2 | 425.00                 | 156.90    |

- 4线程(运行10次取平均值)

| param                           | Autokernel dw_con (us) | Tengine(us) |
| ------------------------------- | ---------------------- | ----------- |
| c = 32, input_size = 112, s = 1 | 1092.30                |             |
| c = 64, input_size = 112, s = 2 | 721.20                 |             |
| c = 128, input_size = 56, s = 1 | 1077.70                |             |
| c = 128, input_size = 56, s = 2 | 464.40                 |             |
| c = 256, input_size = 28, s = 1 | 692.80                 |             |
| c = 256, input_size = 28, s = 2 | 294.00                 |             |
| c = 512, input_size = 14, s = 1 | 482.00                 |             |
| c = 512, input_size = 14, s = 2 | 188.90                 |             |
| c = 1024, input_size = 7, s = 2 | 287.10                 |             |