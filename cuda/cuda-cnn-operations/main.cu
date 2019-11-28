//
// Created by slimakanzer on 21.07.18.
//
#include "ops.cu"
// Constant versions of gflags
#ifdef USE_GFLAGS
#include <gflags/gflags.h>

    #ifndef _WIN32
        #define gflags google
    #endif
#else
#define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
#define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
#define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
#define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
#define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag ((default_value))
#endif

///////////////////////////////////////////////////////////////////////////////////////////
// Command-line flags

// Only for synthetic data
DEFINE_bool(use_synthetic, true, "Synthetic data for learning");
DEFINE_int32(H, 32, "Height of input layer");
DEFINE_int32(W, 32, "Width of input layer");
DEFINE_int32(C, 1, "Number of channels of input layer");
DEFINE_int32(R, 2, "Height of kernel");
DEFINE_int32(S, 2, "Width of kernel");
DEFINE_int32(K, 1, "Number of kernels");
DEFINE_int32(str_x, 1, "Stride along the axis X");
DEFINE_int32(str_y, 1, "Stride along the axis Y");
DEFINE_int32(pad_x, 0, "Padding along the axis X");
DEFINE_int32(pad_y, 0, "Padding along the axis Y");


DEFINE_uint64(batch_size, 8, "Batch size for training");

DEFINE_string(test_images, "none", "Path to file with test dataset");
DEFINE_string(test_labels, "none", "Path to file with test labels");

DEFINE_bool(convolution, false, "Use convolution operation");
DEFINE_bool(max_pooling, false, "Use max-pooling operation");
DEFINE_bool(softmax, true, "Use softmax operation");


int main(int argc, char **argv){

#ifdef USE_GFLAGS
        gflags::ParseCommandLineFlags(&argc, &argv, true);
#endif

        if(FLAGS_convolution){

            dim3 synthetic_inner(FLAGS_W, FLAGS_H, FLAGS_C);
            dim3 synthetic_kernel(FLAGS_S, FLAGS_R, FLAGS_K);

            convolution_prepare(
                    FLAGS_use_synthetic,
                    FLAGS_test_images,
                    FLAGS_test_labels,
                    synthetic_inner,
                    synthetic_kernel,

                    FLAGS_str_x,
                    FLAGS_str_y,
                    FLAGS_pad_x,
                    FLAGS_pad_y
            );
        } else if(FLAGS_max_pooling){

            dim3 synthetic_inner(FLAGS_W, FLAGS_H, FLAGS_C);
            dim3 synthetic_kernel(FLAGS_S, FLAGS_R, FLAGS_K);

            max_pooling_prepare(
                    FLAGS_use_synthetic,
                    FLAGS_test_images,
                    FLAGS_test_labels,


                    synthetic_inner,
                    synthetic_kernel,

                    FLAGS_str_x,
                    FLAGS_str_y,
                    FLAGS_pad_x,
                    FLAGS_pad_y
            );
        } else if (FLAGS_softmax){
            dim3 sythetic_neurons(1000, 1);

            softmax_prepare(
                    FLAGS_use_synthetic,
                    FLAGS_test_images,
                    FLAGS_test_labels,
                    sythetic_neurons

                    );

        }

    return 0;
}


