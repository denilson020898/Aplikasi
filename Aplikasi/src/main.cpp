#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_inverse.hpp"

#include <iostream>
#include <stb_image.h>

#include "Shader.h"
#include "bvh2.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

unsigned int SCR_WIDTH = 1330;
unsigned int SCR_HEIGHT = 768;

glm::vec3 cameraPos = glm::vec3(0.0f, 70.0f, 200.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float cameraSpeed = 2.5f;

bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
float fov = 45.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

// settings
static GLuint vboVert, vboIndices;
static GLuint vao;

glm::mat4 model = glm::mat4(1.0f);

short bvh_elements = 0;
Bvh2* bvh;
int frame = 0;
int frameChange = 1;

void tmpProcess(Joint* joint,
                std::vector<glm::vec4>& vertices,
                std::vector<GLshort>& indices,
                GLshort parentIndex = 0)
{
  glm::vec4 translatedVertex = joint->matrix[3];

  vertices.push_back(translatedVertex);

  GLshort myindex = (GLshort)(vertices.size() - 1);

  if (parentIndex != myindex)
  {
    indices.push_back(parentIndex);
    indices.push_back(myindex);
  }

  for (auto& child : joint->children)
  {
    tmpProcess(child, vertices, indices, myindex);
  }
}

void update()
{

  if (frameChange)
  {
    frame = frame + frameChange;
  }
  else
  {
    return;
  }

  int frameto = frame % bvh->getNumFrames();
  //std::cout << "move to " << frameto << std::endl;
  bvh->moveTo(frameto);

  std::vector<glm::vec4> vertices;
  std::vector<GLshort> bvhindices;

  tmpProcess((Joint*)bvh->getRootJoint(), vertices, bvhindices);

  glBindBuffer(GL_ARRAY_BUFFER, vboVert);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), &vertices[0], GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

int main()
{
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Penulisan Ilmiah", NULL, NULL);
  if (window == NULL)
  {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);
  //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  glfwSwapInterval(1);
  //glEnable(GL_DEPTH_TEST);
  glLineWidth(2.5);
  glPointSize(5.0);

  Shader bvhShader("shader.vs", "shader.fs");

  bvh = new Bvh2;
  bvh->load("data/example.bvh");
  bvh->testOutput();
  //bvh->printJoint(bvh->getRootJoint());
  bvh->moveTo(frame);

  std::vector<glm::vec4> vertices;
  std::vector<GLshort> bvhindices;

  tmpProcess((Joint*)bvh->getRootJoint(), vertices, bvhindices);
  bvh_elements = (short)bvhindices.size();

  glGenBuffers(1, &vboVert);
  glBindBuffer(GL_ARRAY_BUFFER, vboVert);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), &vertices[0], GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenBuffers(1, &vboIndices);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboIndices);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(bvhindices[0]) * bvhindices.size(), &bvhindices[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  bvhShader.use();


  glBindBuffer(GL_ARRAY_BUFFER, vboVert);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboIndices);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);

  // GRID

  float gridVertices[] = {
    // positions          // colors           // texture coords
     100.0f, 0.0f,  100.0f, 
     100.0f, 0.0f, -100.0f, 
    -100.0f, 0.0f, -100.0f, 
    -100.0f, 0.0f,  100.0f, 
  };
  unsigned int gridIndices[] = {
    0, 1, 3, // first triangle
    1, 2, 3  // second triangle
  };

  unsigned int VBO, VAO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(gridVertices), gridVertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(gridIndices), gridIndices, GL_STATIC_DRAW);

  // position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  // color attribute

  Shader gridShader("floor.vs", "floor.fs");

  //model = glm::scale(model, glm::vec3(0.5f, 0.5f, 0.5f));
  while (!glfwWindowShouldClose(window))
  {

    float currentTime = (float)glfwGetTime();
    deltaTime = currentTime - lastFrame;
    lastFrame = currentTime;

    processInput(window);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);


    // skeleton
    glm::mat4 view = glm::mat4(1.0f);
    view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::mat4(1.0f);
    projection = glm::perspective(glm::radians(fov), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 1000.0f);
    glm::mat4 mvp = projection * view * model;

    //grid
    gridShader.use();
    gridShader.setMat4("mvp", mvp);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    bvhShader.use();
    bvhShader.setMat4("mvp", mvp);
    update();
    glBindVertexArray(vao);
    glDrawElements(GL_LINES, bvh_elements, GL_UNSIGNED_SHORT, (GLvoid*)0);
    glDrawElements(GL_POINTS, bvh_elements, GL_UNSIGNED_SHORT, (GLvoid*)0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
  glViewport(0, 0, width, height);
}

void processInput(GLFWwindow *window)
{
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    cameraSpeed = 100.0f;
  else
    cameraSpeed = 50.0f;

  float appliedSpeed = cameraSpeed * deltaTime;
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    cameraPos += appliedSpeed * cameraFront;
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    cameraPos -= appliedSpeed * cameraFront;
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * appliedSpeed;
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * appliedSpeed;

  if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    cameraPos += appliedSpeed * cameraUp;
  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    cameraPos -= appliedSpeed * cameraUp;

}

void mouse_callback(GLFWwindow * window, double xpos, double ypos)
{
  float xposf = (float)xpos;
  float yposf = (float)ypos;

  int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
  if (state == GLFW_PRESS)
  {

  if (firstMouse)
  {
    lastX = xposf;
    lastY = yposf;
    firstMouse = false;
  }

  float xoffset = xposf - lastX;
  float yoffset = lastY - yposf;
  lastX = xposf;
  lastY = yposf;

  float sensitivity = 0.1f;
  xoffset *= sensitivity;
  yoffset *= sensitivity;

  yaw += xoffset;
  pitch += yoffset;

  if (pitch > 89.0f)
    pitch = 89.0f;
  if (pitch < -89.0f)
    pitch = -89.0f;

  glm::vec3 front;
  front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
  front.y = sin(glm::radians(pitch));
  front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
  cameraFront = glm::normalize(front);
  }
}

void scroll_callback(GLFWwindow * window, double xoffset, double yoffset)
{
  if (fov >= 1.0f && fov <= 45.0f)
    fov -= (float)yoffset;
  if (fov <= 1.0f)
    fov = 1.0f;
  if (fov >= 45.0f)
    fov = 45.0f;
}
