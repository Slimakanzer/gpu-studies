//
// Created by slimakanzer on 08.02.19.
//
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "move.h"
#include "common/shader.hpp"
using namespace glm;

GLFWwindow* window;

void generate_buffer(GLfloat* buffer);

int main(void){
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        getchar();
        return -1;
    }

    glfwDefaultWindowHints();

    window = glfwCreateWindow( 1024, 768, "TASK 3", NULL, NULL);
    if( window == NULL ){
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    static GLfloat g_vertex_buffer_data[1800];
    generate_buffer(g_vertex_buffer_data);

    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    GLuint programID = LoadShaders( "PlaneVertexShader.vertexshader", "PlaneFragmentShader.fragmentshader" );
    GLuint MatrixID = glGetUniformLocation(programID, "MVP");

    do{
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(programID);

        computeMatrices();
        glm::mat4 ProjectionMatrix = getProjectionMatrix();
        glm::mat4 ViewMatrix = getViewMatrix();
        glm::mat4 ModelMatrix = glm::mat4(1.0);
        glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
                0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
                3,                  // size
                GL_FLOAT,           // type
                GL_FALSE,           // normalized?
                0,                  // stride
                (void*)0            // array buffer offset
        );

        glDrawArrays(GL_TRIANGLES, 0, 100*2*3);
        glDisableVertexAttribArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();

    }
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0 );

    glDeleteBuffers(1, &vertexbuffer);
    glDeleteVertexArrays(1, &VertexArrayID);
    glDeleteProgram(programID);

    glfwTerminate();

    return 0;
}

void generate_buffer(GLfloat* buffer){
    GLfloat step = 0.1f;

    for (int i = 0; i < 100; ++i) {
        int index = i * 6 * 3;

        buffer[index] = 0.0f;
        buffer[index+1] = 0.0f;
        buffer[index+2] = step * i;

        buffer[index+3] = 0.0f;
        buffer[index+4] = 0.0f;
        buffer[index+5] = step * (i+1);

        buffer[index+6] = 1.0f;
        buffer[index+7] = 0.0f;
        buffer[index+8] = step * i;

        //---------------------------------------------//

        buffer[index+9] = 0.0f;
        buffer[index+10] = 0.0f;
        buffer[index+11] = step * (i+1);

        buffer[index+12] = 1.0f;
        buffer[index+13] = 0.0f;
        buffer[index+14] = step * i;


        buffer[index+15] = 1.0f;
        buffer[index+16] = 0.0f;
        buffer[index+17] = step * (i+1);
    }
}
