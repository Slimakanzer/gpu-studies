#!/usr/bin/env bash
mkdir ./shaders/bin 2>/dev/null
mkdir ./shaders/bin/square 2>/dev/null

./external/vulkan/1.1.101.0/x86_64/bin/glslangValidator ./shaders/src/shader.frag -V -o ./shaders/bin/frag.spv
./external/vulkan/1.1.101.0/x86_64/bin/glslangValidator ./shaders/src/shader.vert -V -o ./shaders/bin/vert.spv

./external/vulkan/1.1.101.0/x86_64/bin/glslangValidator ./shaders/src/square/square.frag -V -o ./shaders/bin/square/frag.spv
./external/vulkan/1.1.101.0/x86_64/bin/glslangValidator ./shaders/src/square/square.vert -V -o ./shaders/bin/square/vert.spv
./external/vulkan/1.1.101.0/x86_64/bin/glslangValidator ./shaders/src/square/square.tesc -V -o ./shaders/bin/square/tesc.spv
./external/vulkan/1.1.101.0/x86_64/bin/glslangValidator ./shaders/src/square/square.tese -V -o ./shaders/bin/square/tese.spv