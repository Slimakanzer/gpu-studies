//
// Created by slimakanzer on 07.02.19.
//
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include "move.h"
extern GLFWwindow* window;

glm::mat4 ViewMatrix;
glm::mat4 ProjectionMatrix;

glm::mat4 getViewMatrix(){
    return ViewMatrix;
}
glm::mat4 getProjectionMatrix(){
    return ProjectionMatrix;
}

float speed = 3.0f;
float speedRange = 5.0f;
float angle = 0.0f; //initial angle
float range = 4.0f;

void computeMatrices(){
    static double lastTime = glfwGetTime();

    double currentTime = glfwGetTime();
    float deltaTime = float(currentTime - lastTime);

    if (glfwGetKey( window, GLFW_KEY_UP ) == GLFW_PRESS){
        range -= deltaTime * speedRange;
    }
    if (glfwGetKey( window, GLFW_KEY_DOWN ) == GLFW_PRESS){
        range +=  deltaTime * speedRange;
    }

    if (glfwGetKey( window, GLFW_KEY_RIGHT ) == GLFW_PRESS){
        angle += deltaTime * speed;
    }

    if (glfwGetKey( window, GLFW_KEY_LEFT ) == GLFW_PRESS){
        angle -= deltaTime * speed;
    }

    glm::vec3 position = glm::vec3( cos(angle) * range, range, sin(angle) * range);

    ProjectionMatrix = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 100.0f);
    ViewMatrix       = glm::lookAt(
            position,           // Camera is here
            glm::vec3(0,0,0), // and looks at the origin
            glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
    );
    lastTime = currentTime;
}