//
// Created by slimakanzer on 21.07.18.
//

extern "C" float convolution(
        float *input,
        float *kernel,
        float *output,

        int width_input,
        int height_input,
        int deep_input,

        int width_kernel,
        int height_kernel,
        int deep_kernel,
        int long_kernel,

        int width_output,
        int height_output,
        int deep_output,


        int stride_x,
        int stride_y,
        int padding_x,
        int padding_y
);
