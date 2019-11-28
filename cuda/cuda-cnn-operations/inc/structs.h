//
// Created by slimakanzer on 25.07.18.
//

#ifndef FRAMEWORK_STRUCTS_H
#define FRAMEWORK_STRUCTS_H

struct tensor_3D{
    std::string name;
    int H;
    int W;
    int D;

    std::vector<float> value;
};

struct tensor_4D{
    std::string name;
    int H;
    int W;
    int D;
    int L;

    std::vector<float> value;
};

#endif