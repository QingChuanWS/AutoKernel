#include <iostream>
#include "depthwise.h"

static inline unsigned long get_cur_time(void)
{
    struct timespec tm;

    clock_gettime(CLOCK_MONOTONIC, &tm);

    return (tm.tv_sec * 1000000 + tm.tv_nsec / 1000);
}

// add helper data struct and functions here
struct conv_priv_info
{
    void* interleave_buffer;    // kernel transform buffer
    void* interleave_buffer_pack4;    // kernel pack4
    void* im2col_buffer;    // input data transform buffer
    void* im2col_buffer_pack4;    // input data transform buffer pack4
    void* input_pad;
    void* dot_block;
    void* transform_input;
    void* output_bordered;
    int im2col_buffer_size;    // kernel transform buffer size
    int im2col_buffer_pack4_size;    // kernel transform buffer size
    int interleave_buffer_size;    // input data transform buffer size
    int interleave_buffer_pack4_size;
    int external_im2col_mem;    // flag
    int external_im2col_pack4_mem;    // flag
    int external_interleave_mem;    // flag
    int external_interleave_pack4_mem;    // flag
    int cpu_type;
    int winograd;

    /* hybrid int8 params */
    void* p_input_max;
    void* p_kernel_max;
};

static int get_private_mem_size(struct ir_tensor* filter)
{
    return filter->elem_num * filter->elem_size;    // only support FP32
}

int conv_auto_set_shared_mem(struct conv_priv_info* priv_info, void* mem, int mem_size)
{
    priv_info->external_im2col_mem = 1;
    priv_info->im2col_buffer = mem;
    priv_info->im2col_buffer_size = mem_size;
    return 0;
}

int conv_auto_get_shared_mem_size(struct ir_tensor* input, struct ir_tensor* output, struct conv_param* param)
{
    int group = param->group;
    int input_chan = param->input_channel / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;
    int output_xy = output->dims[2] * output->dims[3];
    int elem_size = input->elem_size;

    return elem_size * output_xy * kernel_size;
}

int conv_auto_get_shared_pack4_mem_size(struct ir_tensor* filter, struct ir_tensor* output, struct conv_param* param)
{
    int K = filter->elem_num / filter->dims[0];
    int N = output->dims[2] * output->dims[3];
    int elem_size = filter->elem_size;

    // Only Support fp32
    return (8 * K * (N / 8 + N % 8)) * elem_size;
}

void auto_im2col(float* data_img, float* data_col, \
	    int inh, int inw, int inc, \
	    int outh, int outw, int outc, \
        int ksize_h, int ksize_w, \
        int sh, int sw, int ph, int pw, int dh, int dw)
{
    const int channels_col = ksize_h * ksize_w * inc;

    for(int c = 0; c < channels_col; ++c)
    {
        const int kw = c % ksize_w;
        int c_ = c / ksize_w;
        const int kh = c_ % ksize_h;
        c_ = c_ / ksize_h;
        const int im_col = kw * dw - pw;
        const int w_low = std::max(0, -im_col / sw + (-im_col % sw > 0));
        const int w_high = std::min(outw, (inw - im_col) / sw + ((inw - im_col) % sw > 0));
        for(int h = 0; h < outh; ++h)
	    {
	        const int im_row = kh * dh + h * sh - ph;
	        float* out = data_col + (c * outh + h) * outw;
	        const float* end = out + w_high;

	        if(im_row >= 0 && im_row < inh)
	        {
                float* in = data_img + inw * (im_row + inh * c_) + im_col + (w_low - 1) * sw;
	            memset(out, 0, w_low * sizeof(float));
		        out += w_low;
		        while(out < end)
		        {
		            in += sw;
	                *(out++) = *in;
		        }
		    memset(out, 0, (outw - w_high) * sizeof(float));
	        }
	        else
	        {
     		    memset(out, 0, outw * sizeof(float));
            }
	    }
    }
}

void auto_add_bias(float* output, float* bias, int c_out, int hw)
{
    for(int c = 0; c < c_out; ++c)
    {
        for(int i = 0; i < hw; ++i)
        {
            output[c * hw + i] += bias[c];
        }
    }
}

void auto_relu(float* data, int size, int activation)
{
    for(int i = 0; i < size; i++)
    {
        data[i] = std::max(data[i], ( float )0);
        if(activation > 0)
        {
            data[i] = std::min(data[i], ( float )activation);
        }
    }
}

static void interleave(struct ir_tensor* filter, struct conv_priv_info* priv_info)
{
    /* simply copy the data */
    memcpy(priv_info->interleave_buffer, filter->data, filter->elem_num * filter->elem_size);
}


static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;

    /* get cpu affinity */
    conv_priv_info->cpu_type = exec_graph->cpu_affinity;
    
    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        if (exec_node->shared_mem_size < exec_graph->shared_mem_size)
	    {   
            if (conv_auto_set_shared_mem(conv_priv_info, exec_graph->shared_mem, exec_graph->shared_mem_size) < 0)
	        {
                printf("halide dw_im2col: set shared memory failed\n");
		    set_tengine_errno(EFAULT);
		    return -1;
	        }
	    }
	    conv_priv_info->external_interleave_pack4_mem = 0;
        
	    /* do prerun interleave */
	    if (!conv_priv_info->external_im2col_mem)
	    {
	        int mem_size = conv_auto_get_shared_mem_size(input_tensor, output_tensor, conv_param);
	        void* mem = sys_malloc(mem_size);
	        conv_priv_info->im2col_buffer = mem;
	        conv_priv_info->im2col_buffer_size = mem_size;
	    }

	    if (!conv_priv_info->external_interleave_mem)
	    {
	        int mem_size = get_private_mem_size(filter_tensor);
	        void* mem = sys_malloc(mem_size);
	        conv_priv_info->interleave_buffer = mem;
	        conv_priv_info->interleave_buffer_size = mem_size;
	    }

	    // Only support FP32
	    interleave(filter_tensor, conv_priv_info);
        }
        else
        {
            printf("Tengine work node not support %d\n", exec_graph->mode);
	        return -1;
        }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    bool info_autokernel = false;
    const char* debug_env = std::getenv("DEBUG_INFO");
    if((debug_env) && (debug_env[0] == '1'))
    {
        info_autokernel = true;
    } 
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
      struct ir_tensor* input_tensor;
    struct ir_tensor* weight_tensor;
    struct ir_tensor* output_tensor;
    struct ir_tensor* bias_tensor = NULL;
    // int num_thread = exec_graph->num_thread;
    // int cpu_affinity = exec_graph->cpu_affinity;

    /* set the input data and shape again, in case of reshape or dynamic shape */
    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;
    struct conv_priv_info* conv_priv_info = (struct conv_priv_info* ) exec_node->ops_priv;

    /* fp32 run */
    int group = conv_param->group;
    int inc_g = (input_tensor->dims[1]) / group;
    int outc_g = (output_tensor->dims[1]) / group;
    int ksize_h = conv_param->kernel_h;
    int ksize_w = conv_param->kernel_w;
    int stride_h = conv_param->stride_h;
    int stride_w = conv_param->stride_w;
    int pad_h = conv_param->pad_h0;
    int pad_w = conv_param->pad_w0;
    int dilation_h = conv_param->dilation_h;
    int dilation_w = conv_param->dilation_w;
    int activation = conv_param->activation;

    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
	    int K = weight_tensor->elem_num / weight_tensor->dims[0];
	    int N = output_tensor->dims[2] * output_tensor->dims[3];
	    int M = output_tensor->dims[1] / group;                   
		unsigned long off_time_1 = 0;	
		unsigned long off_time_2 = 0;
		for(int j = 0; j < group; j++)
		{  
			unsigned long start_time = get_cur_time();
	    	{
			auto_im2col((float*)(input_tensor->data) + j * input_tensor->dims[2] * input_tensor->dims[3], //input data point
				(float*)(conv_priv_info->im2col_buffer),  // im2col buffer
		        input_tensor->dims[2], input_tensor->dims[3], inc_g,  // input size
		        output_tensor->dims[2], output_tensor->dims[3], outc_g,  // output size
                ksize_h, ksize_w,  // k size
                stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w); // conv param
			
			unsigned long end_time_1 = get_cur_time();
			off_time_1 += end_time_1 - start_time;			
			//std::printf("im2col used %lu us\n", off_time_1);
		    
			Halide::Runtime::Buffer<float> filter_data((float*)conv_priv_info->interleave_buffer + K * j, K);
		    Halide::Runtime::Buffer<float> output_data(((float*)output_tensor->data) + N * j, N);
		    Halide::Runtime::Buffer<float> input_data((float*)conv_priv_info->im2col_buffer, N, K);
		    halide_depthwise(filter_data, input_data, output_data);
           	}
			 
			//unsigned long end_time_2 = get_cur_time();
			//off_time_2 += end_time_2 - end_time_1;
			//std::printf("gemm used %lu\n", off_time_2);
	    }   
		std::printf("im2col used %lu us\n", off_time_1);
		//std::printf("gemm used %lu us\n", off_time_2);
	
	    if(ir_node->input_num > 2)
	    {
	        float* bias_data = (float*)bias_tensor->data;
	        for(int i = 0; i < output_tensor->dims[0]; i++)
    	    {
		        auto_add_bias((float*)output_tensor->data, bias_data, output_tensor->dims[1], output_tensor->dims[2]*output_tensor->dims[3]);
	        }   
	    }
	
        if(activation >= 0)
	    {
	        auto_relu((float*)output_tensor->data, output_tensor->elem_num, activation);
	    }

        if(info_autokernel)printf("[INFO]: runing AutoKernel depthwise ...\n");
    }
    else
    {
        printf("Tengine work node not support %d\n", exec_graph->mode);
	    return -1;
    }
    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
     struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;

    /* fp32 postrun */
    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        if (conv_priv_info->external_interleave_pack4_mem && !conv_priv_info->external_interleave_mem && conv_priv_info->interleave_buffer != NULL)
        {
            sys_free(conv_priv_info->interleave_buffer_pack4);
            conv_priv_info->interleave_buffer_pack4 = NULL;
        }

        if (!conv_priv_info->external_im2col_mem && conv_priv_info->im2col_buffer != NULL)
        {
            sys_free(conv_priv_info->im2col_buffer);
            conv_priv_info->im2col_buffer = NULL;
        }
        if (!conv_priv_info->external_im2col_pack4_mem && conv_priv_info->im2col_buffer_pack4 != NULL)
        {
            sys_free(conv_priv_info->im2col_buffer_pack4);
            conv_priv_info->im2col_buffer_pack4 = NULL;
        }
        if (conv_priv_info->external_interleave_pack4_mem && conv_priv_info->interleave_buffer_pack4 != NULL)
        {
            sys_free(conv_priv_info->interleave_buffer_pack4);
            conv_priv_info->interleave_buffer_pack4 = NULL;
        }
    }
    else
    {
        printf("Tengine work node not support %d\n", exec_graph->mode);
	return -1;
    }
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* filter_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;

    /* init the private info data of convolution op */
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )sys_malloc(sizeof(struct conv_priv_info));
    if (conv_priv_info == NULL)
    {
        set_tengine_errno(ENOMEM);
	return -1;
    }
    memset(conv_priv_info, 0, sizeof(struct conv_priv_info));
    exec_node->ops_priv = conv_priv_info;

    /* get shared memory size */
    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        exec_node->shared_mem_size = conv_auto_get_shared_mem_size(input_tensor, output_tensor, conv_param);
        exec_node->shared_pack4_mem_size = conv_auto_get_shared_pack4_mem_size(filter_tensor, output_tensor, conv_param);
    }
    else
    {
        printf("Tengine work node not support %d\n", exec_graph->mode);
	    return -1;
    }
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;
    sys_free(conv_priv_info);
    exec_node->ops_priv = NULL;
    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    struct conv_param* param = ( struct conv_param* )exec_node->op.param_mem;
    struct ir_node* ir_node = exec_node;
    struct ir_graph* ir_graph = ir_node->graph;

    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_h0;
    int pad_h1 = param->pad_h1;
    int pad_w1 = param->pad_w1;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
  
    int in_c = input_tensor->dims[1] / group;
    int out_c = output_tensor->dims[1] / group;

    if (input_tensor->data_type != TENGINE_DT_FP32)
	return 0;
    if (kernel_h != kernel_w || input_tensor->dims[0] > 1)
	return 0;
    
    if (param->group > 1 && in_c == 1 && out_c == 1 && pad_h0 == pad_h1 && pad_w0 == pad_w1 
       && dilation_h == 1 && dilation_w == 1 && kernel_h == 3 && kernel_w == 3
       && ((stride_h == 1 && stride_w == 1) || (stride_w == 2 && stride_h == 2)))
    {    
        return OPS_SCORE_STATIC;
    }
    else
    {
	return 0;
    }
}

static struct node_ops autokernel_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_autokernel_ops(void* arg)
{
    return register_builtin_node_ops(OP_CONV, &autokernel_node_ops);
}

//static int unreg_autokernel_ops(void* arg)
//{
//    unregister_builtin_node_ops(OP_DEPTHWISE, &autokernel_node_ops);
//    return 0;
//}

void RegisterAutoKernelDepthwise()
{
    register_norm_module_init(2, "reg_autokernel_ops", reg_autokernel_ops, NULL);
}

