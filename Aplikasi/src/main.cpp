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

// GLFW callbacks declarations
void frameBufferSizeCallback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
//void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

// renderer settings
unsigned int screenWidth = 1800;
unsigned int screenHeight = 1000;
int FPS = 100;
float deltaTime = 0.0f;
float lastFrame = 0.0f;
bool loop = false;

// camera settings
glm::vec3 cameraPos = glm::vec3(100.0f, 70.0f, 300.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float cameraSpeed = 2.5f;
bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = screenWidth / 2.0f;
float lastY = screenHeight / 2.0f;
float fov = 45.0f;
float backgroundColor[3] = {0.2f, 0.2f, 0.2f};
float floorColor[3] = {0.05f, 0.05f, 0.05f};
float boneColor[3] = {0.5f, 0.5f, 0.5f};
float jointColor[3] = {0.5f, 1.0f, 1.0f};
float comColor[3] = {1.0f, 1.0f, 0.5f};

// COG properties
int selectedGender = 0;
float totalBodyWeight = 60.0f;

float headNeckMassPercent[2] = { 6.94f  , 6.68f  };
float trunkMassPercent[2]    = { 43.46f , 42.58f };
float upperArmMassPercent[2] = { 2.71f  , 2.55f  };
float foreArmMassPercent[2]  = { 1.62f  , 1.38f  };
float handMassPercent[2]     = { 0.61f  , 0.56f  };
float thighMassPercent[2]    = { 14.16f , 14.78f };
float shankMassPercent[2]    = { 4.33f  , 4.81f  };
float footMassPercent[2]     = { 1.37f  , 1.29f  };

float headNeckLengthPercent[2] = { 50.02f , 48.41f };
float trunkLengthPercent[2]    = { 43.10f , 37.82f };
float upperArmLengthPercent[2] = { 57.72f , 57.54f };
float foreArmLengthPercent[2]  = { 45.74f , 45.59f };
float handLengthPercent[2]     = { 79.00f , 74.74f };
float thighLengthPercent[2]    = { 40.95f , 36.12f };
float shankLengthPercent[2]    = { 43.95f , 43.52f };
float footLengthPercent[2]     = { 44.15f , 40.14f };

// bvh settings
float boneWidth = 2.0;
float jointPointSize = 4.0;
float comPointSize = 8.0;

Bvh2* bvh;
unsigned int bvhVBO, bvhEBO, bvhVAO;
std::vector<glm::vec4> bvhVertices;
std::vector<short> bvhIndices;
short bvhElements = 0;
int bvhFrame = 0;
bool frameChange = false;

unsigned int cogVBO, cogVAO;
std::vector<glm::vec4> cogVertices;

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

glm::vec4 myLerp(glm::vec4 x, glm::vec4 y, float t) {
  return x * (1.f - t / 100.0f) + y * (t / 100.0f);
}

void processCOG(const std::vector<glm::vec4>& bvhVertices, std::vector<glm::vec4>& cogVertices)
{
  cogVertices.clear();

  // head
  cogVertices.push_back(myLerp(bvhVertices[3], bvhVertices[5], headNeckLengthPercent[selectedGender]));
  // trunk
  cogVertices.push_back(myLerp(bvhVertices[0], bvhVertices[2], headNeckLengthPercent[selectedGender]));
  // left upper arm
  cogVertices.push_back(myLerp(bvhVertices[6], bvhVertices[8], headNeckLengthPercent[selectedGender]));
  // right upper arm
  cogVertices.push_back(myLerp(bvhVertices[11], bvhVertices[13], headNeckLengthPercent[selectedGender]));
  // left fore arm
  cogVertices.push_back(myLerp(bvhVertices[8], bvhVertices[9], headNeckLengthPercent[selectedGender]));
  // right fore arm
  cogVertices.push_back(myLerp(bvhVertices[13], bvhVertices[14], headNeckLengthPercent[selectedGender]));
  // left hand
  cogVertices.push_back(myLerp(bvhVertices[9], bvhVertices[10], headNeckLengthPercent[selectedGender]));
  // right hand
  cogVertices.push_back(myLerp(bvhVertices[14], bvhVertices[15], headNeckLengthPercent[selectedGender]));
  // left thigh
  cogVertices.push_back(myLerp(bvhVertices[16], bvhVertices[17], headNeckLengthPercent[selectedGender]));
  // right thigh
  cogVertices.push_back(myLerp(bvhVertices[21], bvhVertices[22], headNeckLengthPercent[selectedGender]));
  // left shank
  cogVertices.push_back(myLerp(bvhVertices[17], bvhVertices[18], headNeckLengthPercent[selectedGender]));
  // right shank
  cogVertices.push_back(myLerp(bvhVertices[22], bvhVertices[23], headNeckLengthPercent[selectedGender]));
  // left foot
  cogVertices.push_back(myLerp(bvhVertices[18], bvhVertices[20], headNeckLengthPercent[selectedGender]));
  // right foot
  cogVertices.push_back(myLerp(bvhVertices[23], bvhVertices[25], headNeckLengthPercent[selectedGender]));

  // BODY COG
  

  glGenVertexArrays(1, &cogVAO);
  glGenBuffers(1, &cogVBO);

  glBindVertexArray(cogVAO);

  glBindBuffer(GL_ARRAY_BUFFER, cogVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(cogVertices[0]) * cogVertices.size(), &cogVertices[0], GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, cogVBO);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
}

void updateBvh()
{
  if (frameChange)
  {
    bvhFrame++;
    if (loop)
    {
      bvhFrame = bvhFrame % bvh->getNumFrames();
    }
    else if (!loop && (unsigned int)bvhFrame < bvh->getNumFrames())
    {
    }
    else 
    {
      frameChange = false;
      bvhFrame = 0;
    }
  }

  //std::cout << "move to " << frameto << std::endl;
  bvh->moveTo(bvhFrame);

  bvhVertices.clear();
  bvhIndices.clear();
  processBvh((Joint*)bvh->getRootJoint(), bvhVertices, bvhIndices);

  glBindBuffer(GL_ARRAY_BUFFER, bvhVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(bvhVertices[0]) * bvhVertices.size(), &bvhVertices[0], GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

int main()
{
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
  //glfwSetScrollCallback(window, scrollCallback);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize Glad" << std::endl;
    return -1;
  }
  glEnable(GL_DEPTH_TEST);
  glfwSwapInterval(1); 

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
  bvh->load("data/example4.bvh");
  bvh->moveTo(bvhFrame);
  bvhVertices.clear();
  bvhIndices.clear();
  processBvh((Joint*)bvh->getRootJoint(), bvhVertices, bvhIndices);
  bvhElements = (short)bvhIndices.size();

  glGenVertexArrays(1, &bvhVAO);
  glGenBuffers(1, &bvhVBO);
  glGenBuffers(1, &bvhEBO);

  glBindVertexArray(bvhVAO);

  glBindBuffer(GL_ARRAY_BUFFER, bvhVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(bvhVertices[0]) * bvhVertices.size(), &bvhVertices[0], GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, bvhVBO);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bvhEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(bvhIndices[0]) * bvhIndices.size(), &bvhIndices[0], GL_DYNAMIC_DRAW);

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

  //double lastTimeFrame = glfwGetTime();
  while (!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    processInput(window);
    glClearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], 1.0f);
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
    floorShader.setVec3("ourColor", floorColor[0], floorColor[1], floorColor[2]);
    glBindVertexArray(floorVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    // skeleton
    bvhShader.use();
    bvhShader.setMat4("mvp", mvp);
    updateBvh();
    glBindVertexArray(bvhVAO);
    bvhShader.setVec3("ourColor", boneColor[0], boneColor[1], boneColor[2]);
    glDrawElements(GL_LINES, bvhElements, GL_UNSIGNED_SHORT, (void*)0);
    bvhShader.setVec3("ourColor", jointColor[0], jointColor[1], jointColor[2]);
    glDrawElements(GL_POINTS, bvhElements, GL_UNSIGNED_SHORT, (void*)0);

    // cog
    processCOG(bvhVertices, cogVertices);
    floorShader.setVec3("ourColor", comColor[0], comColor[1], comColor[2]);
    glBindVertexArray(cogVAO);
    glDrawArrays(GL_POINTS, 0, cogVertices.size());

    // BVH Player Settings;
    {
      ImGui::Begin("BVH Player Settings");

      ImGui::SliderInt("Frame", &bvhFrame, 0, bvh->getNumFrames());
      ImGui::SameLine();
      ImGui::Checkbox("Loop", &loop);
      ImGui::SameLine();
      if (ImGui::Button("Play / Pause"))
        frameChange = !frameChange;
      ImGui::SameLine();
      if (ImGui::Button("<") && bvhFrame != 0)
        bvhFrame--;
      ImGui::SameLine();
      if (ImGui::Button(">") && bvhFrame != bvh->getNumFrames())
        bvhFrame++;

      //ImGui::InputInt("Desired FPS", &FPS);
      ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
      ImGui::Text("Application average %.3f ms/frame", 1000.0f / ImGui::GetIO().Framerate);
      ImGui::Text("Number of frames: %i", bvh->getNumFrames());

      if (ImGui::CollapsingHeader("Display Settings"))
      {
        ImGui::PushItemWidth(200);
        ImGui::ColorEdit3("Floor Color", floorColor);
        ImGui::SameLine();
        ImGui::ColorEdit3("Background Color", backgroundColor);
        ImGui::SameLine();

        if (ImGui::Button("Reset"))
        {
          backgroundColor[0] = 0.2f;
          backgroundColor[1] = 0.2f;
          backgroundColor[2] = 0.2f;
          floorColor[0] = 0.05f;
          floorColor[1] = 0.05f;
          floorColor[2] = 0.05f;
          boneColor[0] = 0.5f;
          boneColor[1] = 0.5f;
          boneColor[2] = 0.5f;
          jointColor[0] = 0.5f;
          jointColor[1] = 1.0f;
          jointColor[2] = 1.0f;
          comColor[0] = 1.0f;
          comColor[1] = 1.0f;
          comColor[2] = 0.5f;
          boneWidth = 2.0;
          jointPointSize = 4.0;
          comPointSize = 8.0;
        }

        ImGui::ColorEdit3("Bone Color ", boneColor);
        ImGui::SameLine();
        ImGui::ColorEdit3("Joint Color", jointColor);
        ImGui::SameLine();
        ImGui::ColorEdit3("COM Color", comColor);

        ImGui::SliderFloat("Bone Width ", &boneWidth, 0.001f, 10.0f);
        ImGui::SameLine();
        ImGui::SliderFloat("Joint Size ", &jointPointSize, 0.001f, 10.0f);
        ImGui::SameLine();
        ImGui::SliderFloat("COM Size", &comPointSize, 0.001f, 10.0f);
        ImGui::PopItemWidth();
      }

      ImGui::End();
    }

    // BVH Stats
    {
      ImGui::Begin("BVH Status");

      auto nameVector = bvh->getJointNames();
      for (size_t i = 0; i < nameVector.size(); i++)
      {
        if (nameVector[i] == "EndSite")
          nameVector[i] = nameVector[i - 1] + nameVector[i];
      }

      if (ImGui::CollapsingHeader("PlotLines Example")) {
        int index = 13;
        float plotHeight = 200.0f;
        static constexpr int length = 176;
        static float linesx[length] = { 0 };
        static float linesy[length] = { 0 };
        static float linesz[length] = { 0 };
        if (frameChange)
        {
          linesx[bvhFrame] = bvhVertices[index].x;
          linesy[bvhFrame] = bvhVertices[index].y;
          linesz[bvhFrame] = bvhVertices[index].z;
        }
        //ImGui::PlotLines("Test X", linesx, length, 0, "x", -150.0f, 150.0f, ImVec2(0,plotHeight));
        //ImGui::PlotLines("Test Y", linesy, length, 0, "y", -150.0f, 150.0f, ImVec2(0,plotHeight));
        //ImGui::PlotLines("Test Z", linesz, length, 0, "z", -150.0f, 150.0f, ImVec2(0,plotHeight));
        ImGui::PlotHistogram("Text X", linesx, length, 0, "X", -150.0f, 150.0f, ImVec2(0, plotHeight));
        ImGui::PlotHistogram("Text Y", linesy, length, 0, "Y", -150.0f, 150.0f, ImVec2(0, plotHeight));
        ImGui::PlotHistogram("Text Z", linesz, length, 0, "Z", -150.0f, 150.0f, ImVec2(0, plotHeight));
      }

      if (ImGui::CollapsingHeader("Joints' World X Y Z Positions"))
      {
        for (size_t i = 0; i < nameVector.size(); i++)
        {
          glm::vec3 channels = glm::vec3(bvhVertices[i].x, bvhVertices[i].y, bvhVertices[i].z);
          nameVector[i].append(" [");
          nameVector[i].append(std::to_string(i));
          nameVector[i].append("]");
          ImGui::InputFloat3(nameVector[i].c_str(), &channels[0], "%.6f", ImGuiInputTextFlags_ReadOnly);
        }
      }

      if (ImGui::CollapsingHeader("COG Properties"))
      {
        ImGui::Text(" ");

        ImGui::Separator();
        ImGui::Columns(1);
        ImGui::Text("Gender");
        ImGui::Separator();
        ImGui::Columns(1);
        ImGui::Columns(2);
        ImGui::RadioButton("Male", &selectedGender, 0);
        ImGui::NextColumn();
        ImGui::RadioButton("Female", &selectedGender, 1);
        ImGui::Columns(1);
        ImGui::InputFloat("Total Body Weight", &totalBodyWeight);
        ImGui::Separator();

        ImGui::Text(" ");

        ImGui::Columns(1);
        ImGui::Separator();
        ImGui::Text("Segment Mass Percent");

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Head & Neck Mass Male", &headNeckMassPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Head & Neck Mass Female", &headNeckMassPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Trunk Mass Male", &trunkMassPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Trunk Mass Female", &trunkMassPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Upper Arm Mass Male", &upperArmMassPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Upper Arm Mass Female", &upperArmMassPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Fore Arm Mass Male", &foreArmMassPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Fore Arm Mass Female", &foreArmMassPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Hand Mass Male", &handMassPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Hand Mass Female", &handMassPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Thigh Mass Male", &thighMassPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Thigh Mass Female", &thighMassPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Shank Mass Male", &shankMassPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Shank Mass Female", &shankMassPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Foot Mass Male", &footMassPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Foot Mass Female", &footMassPercent[1]); 
        ImGui::Columns(1);
        ImGui::Separator();

        ImGui::Text(" ");

        ImGui::Columns(1);
        ImGui::Separator();
        ImGui::Text("Segment Length Percent");

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Head & Neck Length Male", &headNeckLengthPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Head & Neck Length Female", &headNeckLengthPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Trunk Length Male", &trunkLengthPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Trunk Length Female", &trunkLengthPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Upper Arm Length Male", &upperArmLengthPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Upper Arm Length Female", &upperArmLengthPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Fore Arm Length Male", &foreArmLengthPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Fore Arm Length Female", &foreArmLengthPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Hand Length Male", &handLengthPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Hand Length Female", &handLengthPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Thigh Length Male", &thighLengthPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Thigh Length Female", &thighLengthPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Shank Length Male", &shankLengthPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Shank Length Female", &shankLengthPercent[1]); 
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Separator();
        ImGui::InputFloat("Foot Length Male", &footLengthPercent[0]); 
        ImGui::NextColumn();
        ImGui::InputFloat("Foot Length Female", &footLengthPercent[1]); 
        ImGui::Columns(1);
        ImGui::Separator();

        ImGui::Text(" ");
      }

      if (ImGui::CollapsingHeader("Body COM"))
      {
      }

      if (ImGui::CollapsingHeader("Trunk"))
      {
      }
      if (ImGui::CollapsingHeader("Head"))
      {
      }
      if (ImGui::CollapsingHeader("Left Arm"))
      {
      }
      if (ImGui::CollapsingHeader("Right Arm"))
      {
      }
      if (ImGui::CollapsingHeader("Left Leg"))
      {
      }
      if (ImGui::CollapsingHeader("Right Leg"))
      {
      }

      ImGui::End();
    }

    if (showDemoWindow)
      ImGui::ShowDemoWindow(&showDemoWindow);

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);

    //while (glfwGetTime() < lastTimeFrame + 1.0 / FPS) 
    //{
    //}
    //lastTimeFrame += 1.0 / FPS; 
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

//void scrollCallback(GLFWwindow * window, double xoffset, double yoffset)
//{
//  if (fov >= 1.0f && fov <= 45.0f)
//    fov -= (float)yoffset;
//  if (fov <= 1.0f)
//    fov = 1.0f;
//  if (fov >= 45.0f)
//    fov = 45.0f;
//}

