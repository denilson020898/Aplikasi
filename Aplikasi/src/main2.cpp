#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <iostream>

#include "Shader.h"
#include "bvh2.h"
#include "FPSLimiter.h"

// GLFW callbacks declarations
void frameBufferSizeCallback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

// renderer settings
unsigned int screenWidth = 1600;
unsigned int screenHeight = 900;
int FPS = 100;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// camera settings
glm::vec3 cameraPos = glm::vec3(0.0f, 70.0f, 300.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float cameraSpeed = 2.5f;
bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = screenWidth / 2.0f;
float lastY = screenHeight / 2.0f;
float fov = 45.0f;

// bvh settings
float boneWidth = 1.5;
float jointPointSize = 8.0;
float comPointSize = 8.0;

Bvh2* bvh;
unsigned int bvhVBO, bvhEBO, bvhVAO;
short bvhElements = 0;
int bvhFrame = 0;
bool frameChange = false;

void processBvh(Joint * joint, std::vector<glm::vec4>& vertices, 
                std::vector<short>& indices, short parentIndex = 0)
{
  glm::vec4 translatedVertex = joint->matrix[3];
  vertices.push_back(translatedVertex);
  short myindex = (short)(vertices.size() - 1);
  if (parentIndex != myindex)
  {
    indices.push_back(parentIndex);
    indices.push_back(myindex);
  }
  for (auto& child : joint->children)
  {
    processBvh(child, vertices, indices, myindex);
  }
}

void updateBvh()
{
  if (frameChange)
  {
    bvhFrame++;
  }
  else
  {
    return;
  }

  bvhFrame = bvhFrame % bvh->getNumFrames();
  //std::cout << "move to " << frameto << std::endl;
  bvh->moveTo(bvhFrame);

  std::vector<glm::vec4> vertices;
  std::vector<short> bvhindices;

  processBvh((Joint*)bvh->getRootJoint(), vertices, bvhindices);

  glBindBuffer(GL_ARRAY_BUFFER, bvhVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), &vertices[0], GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

int main()
{
  FPSLimiter fpslimiter;

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "PENULISAN ILMIAH", nullptr, nullptr);
  if (window == nullptr)
  {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, frameBufferSizeCallback);
  glfwSetCursorPosCallback(window, mouseCallback);
  glfwSetScrollCallback(window, scrollCallback);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize Glad" << std::endl;
    return -1;
  }
  glEnable(GL_DEPTH_TEST);

  // floor
  float floorVertices[] = {
     100.0f, 0.0f,  100.0f,
     100.0f, 0.0f, -100.0f,
    -100.0f, 0.0f, -100.0f,
    -100.0f, 0.0f,  100.0f
  };
  unsigned int floorIndices[] = {
    0, 1, 3,
    1, 2, 3
  };
  unsigned int floorVBO, floorEBO, floorVAO;

  glGenVertexArrays(1, &floorVAO);
  glGenBuffers(1, &floorVBO);
  glGenBuffers(1, &floorEBO);

  glBindVertexArray(floorVAO);

  glBindBuffer(GL_ARRAY_BUFFER, floorVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(floorVertices), floorVertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, floorEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(floorIndices), floorIndices, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  Shader floorShader("floor.vs", "floor.fs");

  // bvh
  bvh = new Bvh2;
  bvh->load("data/example2.bvh");
  bvh->testOutput();
  bvh->printJoint(bvh->getRootJoint());
  bvh->moveTo(bvhFrame);
  std::vector<glm::vec4> vertices;
  std::vector<short> bvhindices;
  processBvh((Joint*)bvh->getRootJoint(), vertices, bvhindices);
  bvhElements = (short)bvhindices.size();

  glGenVertexArrays(1, &bvhVAO);
  glGenBuffers(1, &bvhVBO);
  glGenBuffers(1, &bvhEBO);

  glBindVertexArray(bvhVAO);

  glBindBuffer(GL_ARRAY_BUFFER, bvhVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), &vertices[0], GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, bvhVBO);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bvhEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(bvhindices[0]) * bvhindices.size(), &bvhindices[0], GL_DYNAMIC_DRAW);

  Shader bvhShader("shader.vs", "shader.fs");

  // ImGui Context
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");
  bool showDemoWindow = true;

  // mvp
  glm::mat4 model = glm::mat4(1.0f);
  glm::mat4 view = glm::mat4(1.0f);
  glm::mat4 projection = glm::mat4(1.0f);
  glm::mat4 mvp = glm::mat4(1.0f);

  // scale the model if it's too big
  //model = glm::scale(model, glm::vec3(0.25f, 0.25f, 0.25f));
  while (!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    processInput(window);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float currentTime = (float)glfwGetTime();
    deltaTime = currentTime - lastFrame;
    lastFrame = currentTime;

    glLineWidth(boneWidth);
    glPointSize(jointPointSize);

    // pre render calculation
    view = glm::lookAt(cameraPos, cameraPos+cameraFront, cameraUp);
    projection = glm::perspective(glm::radians(fov), (float)screenWidth/(float)screenHeight, 
                                  0.1f, 1000.0f);
    mvp = projection * view * model;

    // draw floor
    floorShader.use();
    floorShader.setMat4("mvp", mvp);
    glBindVertexArray(floorVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    // skeleton
    bvhShader.use();
    bvhShader.setMat4("mvp", mvp);
    updateBvh();
    glBindVertexArray(bvhVAO);
    glDrawElements(GL_LINES, bvhElements, GL_UNSIGNED_SHORT, (void*)0);
    glDrawElements(GL_POINTS, bvhElements, GL_UNSIGNED_SHORT, (void*)0);

    // BVH Player Settings;
    {
      ImGui::Begin("BVH Player Settings");
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS) || FPS in settings (%i)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate, FPS);

      if (ImGui::Button("PLAY / PAUSE"))
        frameChange = !frameChange;
      ImGui::SameLine();
      ImGui::SliderInt("Frame", &bvhFrame, 0, bvh->getNumFrames());
      ImGui::SameLine();
      if (ImGui::Button("GO TO"))
        updateBvh();



      ImGui::SliderFloat("Bone Width", &boneWidth, 0.01, 10.0f);
      ImGui::SliderFloat("Joint Size", &jointPointSize, 0.01, 10.0f);
      ImGui::End();
    }

    // BVH Stats
    {
      ImGui::Begin("BVH Stats");
      ImGui::End();
    }

    if (showDemoWindow)
      ImGui::ShowDemoWindow(&showDemoWindow);

    ImGui::Render();


    //std::cout << 1.0f / deltaTime <<  " [" << FPS << "]" << std::endl;
    fpslimiter.Pulse(FPS);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  glfwTerminate();
  return 0;
}

// GLFW callbacks definitions
void frameBufferSizeCallback(GLFWwindow * window, int width, int height)
{
  glViewport(0, 0, width, height);
}

void processInput(GLFWwindow * window)
{
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    cameraSpeed = 300.0f;
  else
    cameraSpeed = 150.0f;

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

  if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
    boneWidth -= 0.5;
  if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
    boneWidth += 0.5;
  if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
    jointPointSize -= 0.5;
  if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
    jointPointSize += 0.5;

  if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
    FPS -= 1;
  if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
    FPS += 1;
}

void mouseCallback(GLFWwindow * window, double xpos, double ypos)
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

void scrollCallback(GLFWwindow * window, double xoffset, double yoffset)
{
  if (fov >= 1.0f && fov <= 45.0f)
    fov -= (float)yoffset;
  if (fov <= 1.0f)
    fov = 1.0f;
  if (fov >= 45.0f)
    fov = 45.0f;
}
