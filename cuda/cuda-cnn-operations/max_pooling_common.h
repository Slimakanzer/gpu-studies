//
// Created by slimakanzer on 25.07.18.
//

#ifndef FRAMEWORK_MAX_POOLING_COMMON_H
#define FRAMEWORK_MAX_POOLING_COMMON_H
extern "C" float max_pooling(
        float *input,
        float *output,

        int width_input,
        int height_input,
        int deep_input,

        int width_kernel,
        int height_kernel,

        int width_output,
        int height_output,
        int deep_output,


        int stride_x,
        int stride_y,
        int padding_x,
        int padding_y
);
#endif //FRAMEWORK_MAX_POOLING_COMMON_H
