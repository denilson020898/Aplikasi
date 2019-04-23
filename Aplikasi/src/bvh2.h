#pragma once

#include <string>
#include <vector>
#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define Xposition 0x01
#define Yposition 0x02
#define Zposition 0x04
#define Zrotation 0x10
#define Xrotation 0x20
#define Yrotation 0x40

struct Offset
{
  float x, y, z;
};

struct Joint
{
  const char* name = nullptr;
  Joint* parent = nullptr;
  Offset offset;
  unsigned int numChannels = 0;
  short* channelsOrder = nullptr;
  std::vector<Joint*> children;
  glm::mat4 matrix;
  unsigned int channelStart = 0;
};

struct Hierarchy
{
  Joint* rootJoint;
  int numChanneles;
};

struct Motion
{
  unsigned int numFrames;
  unsigned int numMotionChannels = 0;
  float* data = nullptr;
  unsigned int* jointChannelsOffsets;
};

class Bvh2
{
public:
  Bvh2();
  ~Bvh2();

  void printJoint(const Joint* const joint) const;
  void load(const std::string& filename);
  void testOutput() const;
  void moveTo(unsigned int frame);

  const Joint* getRootJoint() const { return rootJoint; }
  unsigned int getNumFrames() const { return motionData.numFrames - 1; }
  std::vector<std::string> getJointNames() { return jointNames; };

private:
  Joint* loadJoint(std::istream& stream, Joint* parent = nullptr);
  void loadHierarchy(std::istream& stream);
  void loadMotion(std::istream& stream);
  void setJointNames(const Joint* const joint);

private:
  Joint* rootJoint;
  Motion motionData;

  std::vector<std::string> jointNames;
};
