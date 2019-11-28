# CMake entry point
cmake_minimum_required (VERSION 3.0)
project (Tutorials)

find_package(OpenGL REQUIRED)


if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
    message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever name, like bin_Visual2012_64bits/)" )
endif()
if( CMAKE_SOURCE_DIR MATCHES " " )
	message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()
if( CMAKE_BINARY_DIR MATCHES " " )
	message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()



# Compile external dependencies 
add_subdirectory (external)

# On Visual 2005 and above, this module can set the debug working directory
cmake_policy(SET CMP0026 OLD)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/rpavlik-cmake-modules-fe2273")
include(CreateLaunchers)
include(MSVCMultipleProcessCompile) # /MP

if(INCLUDE_DISTRIB)
	add_subdirectory(distrib)
endif(INCLUDE_DISTRIB)



include_directories(
	external/AntTweakBar-1.16/include/
	external/glfw-3.1.2/include/
	external/glm-0.9.7.1/
	external/glew-1.13.0/include/
	external/assimp-3.0.1270/include/
	external/bullet-2.81-rev2613/src/
	.
)

set(ALL_LIBS
	${OPENGL_LIBRARY}
	glfw
	GLEW_1130
)

add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNINGS
)

#copy shaders in build for task 1
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_1_triangle/SimpleFragmentShader.fragmentshader
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_1_triangle/SimpleVertexShader.vertexshader
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

#copy shaders in build for task 2
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_2_move/FigureVertexShader.vertexshader
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_2_move/FigureFragmentShader.fragmentshader
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_2_move/input.bmp
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

#copy shaders in build for task 3
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_3_plane/PlaneFragmentShader.fragmentshader
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_3_plane/PlaneVertexShader.vertexshader
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

#copy shaders in build for task 4
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_4_normalmap/NormalFragmentShader.fragmentshader
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_4_normalmap/NormalVertexShader.vertexshader
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_4_normalmap/test.obj
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_4_normalmap/specular.DDS
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_4_normalmap/diffuse.DDS
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/task_4_normalmap/normal.bmp
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
		)


# execute task 1
add_executable(task_1_triangle
		task_1_triangle/main.cpp
		common/shader.cpp
		common/shader.hpp
		task_1_triangle/SimpleFragmentShader.fragmentshader
		task_1_triangle/SimpleVertexShader.vertexshader
)

# execute task 2
add_executable(task_2_move
		task_2_move/figure.cpp
		common/shader.cpp
		common/shader.hpp
		common/texture.cpp
		common/texture.hpp
		task_2_move/move.cpp
		task_2_move/move.h
		task_2_move/FigureVertexShader.vertexshader
		task_2_move/FigureFragmentShader.fragmentshader
		)

# execute task 3
add_executable(task_3_plane
		task_3_plane/plane.cpp
		task_3_plane/move.cpp
		task_3_plane/move.h
		common/shader.cpp
		common/shader.hpp
		task_3_plane/PlaneFragmentShader.fragmentshader
		task_3_plane/PlaneVertexShader.vertexshader
)

# execute task 4
add_executable(task_4_normalmap
		task_4_normalmap/normal.cpp
		task_4_normalmap/move.cpp
		task_4_normalmap/move.h
		common/shader.cpp
		common/shader.hpp
		common/objloader.cpp
		common/objloader.hpp
		common/texture.cpp
		common/texture.hpp
		common/vboindexer.cpp
		common/vboindexer.hpp
		common/tangentspace.cpp
		common/tangentspace.hpp
		task_4_normalmap/NormalFragmentShader.fragmentshader
		task_4_normalmap/NormalVertexShader.vertexshader
		)

target_link_libraries(task_1_triangle
		${ALL_LIBS}
		)

target_link_libraries(task_2_move
		${ALL_LIBS}
		)

target_link_libraries(task_3_plane
		${ALL_LIBS}
		)

target_link_libraries(task_4_normalmap
		${ALL_LIBS}
		)


