#include "bvh2.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <fstream>
#include <iostream>

// trim from start
static inline std::string &ltrim(std::string &s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// trim from end
static inline std::string &rtrim(std::string &s)
{
  s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

// trim from both ends
static inline std::string &trim(std::string &s)
{
  return ltrim(rtrim(s));
}

void deleteJoint(Joint* joint)
{
  if (joint == nullptr)
  {
    return;
  }

  for (Joint* child : joint->children)
  {
    deleteJoint(child);
  }

  if (joint->channelsOrder != nullptr)
  {
    delete joint->channelsOrder;
  }

  delete joint;
}

void moveJoint(Joint* joint, Motion* motionData, int frameStartsIndex)
{
  int startIndex = frameStartsIndex + joint->channelStart;

  joint->matrix = glm::translate(glm::mat4(1.0f),
                                 glm::vec3(joint->offset.x,
                                 joint->offset.y,
                                 joint->offset.z));

  for (unsigned int i = 0; i < joint->numChannels; i++)
  {
    const short& channel = joint->channelsOrder[i];
    float value = motionData->data[startIndex + i];

    if (channel & Xposition)
      joint->matrix = glm::translate(joint->matrix, glm::vec3(value, 0, 0));
    if (channel & Yposition)
      joint->matrix = glm::translate(joint->matrix, glm::vec3(0, value, 0));
    if (channel & Zposition)
      joint->matrix = glm::translate(joint->matrix, glm::vec3(0, 0, value));
    if (channel & Xrotation)
      joint->matrix = glm::rotate(joint->matrix, glm::radians(value), glm::vec3(1, 0, 0));
    if (channel & Yrotation)
      joint->matrix = glm::rotate(joint->matrix, glm::radians(value), glm::vec3(0, 1, 0));
    if (channel & Zrotation)
      joint->matrix = glm::rotate(joint->matrix, glm::radians(value), glm::vec3(0, 0, 1));
  }

  if (joint->parent != nullptr)
    joint->matrix = joint->parent->matrix * joint->matrix;

  for (auto& child : joint->children)
    moveJoint(child, motionData, frameStartsIndex);
}

Bvh2::Bvh2()
  :
  rootJoint(nullptr)
{
  motionData.data = 0;
}

Bvh2::~Bvh2()
{
  deleteJoint(rootJoint);
  if (motionData.data != nullptr)
  {
    delete[] motionData.data;
  }
}

void Bvh2::load(const std::string & filename)
{
  std::fstream file;
  file.open(filename.c_str(), std::ios_base::in);

  if (file.is_open())
  {
    std::string line;

    while (file.good())
    {
      file >> line;
      if (trim(line) == "HIERARCHY")
      {
        loadHierarchy(file);
      }
      break;
    }
    file.close();
  }
}

void Bvh2::testOutput() const
{
  if (rootJoint == nullptr)
    return;

  std::cout << "num frames: " << motionData.numFrames << std::endl;
  std::cout << "num motion channels: " << motionData.numMotionChannels << std::endl;
}

void Bvh2::moveTo(unsigned int frame)
{
  unsigned int startIndex = frame * motionData.numMotionChannels;
  moveJoint(rootJoint, &motionData, startIndex);
}

Joint * Bvh2::loadJoint(std::istream & stream, Joint * parent)
{
  Joint* joint = new Joint;
  joint->parent = parent;
  joint->matrix = glm::mat4(1.0f);

  std::string* name = new std::string;
  stream >> *name;
  joint->name = name->c_str();

  std::string tmp;
  joint->matrix = glm::mat4(1.0f);

  static int _channelStart = 0;
  unsigned channelOrderIndex = 0;
  while (stream.good())
  {
    stream >> tmp;
    tmp = trim(tmp);

    char c = tmp.at(0);
    if (c == 'X' || c == 'Y' || c == 'Z')
    {
      if (tmp == "Xposition")
        joint->channelsOrder[channelOrderIndex++] = Xposition;
      if (tmp == "Yposition")
        joint->channelsOrder[channelOrderIndex++] = Yposition;
      if (tmp == "Zposition")
        joint->channelsOrder[channelOrderIndex++] = Zposition;
      if (tmp == "Xrotation")
        joint->channelsOrder[channelOrderIndex++] = Xrotation;
      if (tmp == "Yrotation")
        joint->channelsOrder[channelOrderIndex++] = Yrotation;
      if (tmp == "Zrotation")
        joint->channelsOrder[channelOrderIndex++] = Zrotation;
    }
    if (tmp == "OFFSET")
    {
      stream >> joint->offset.x >> joint->offset.y >> joint->offset.z;
      glm::mat4 mat = joint->parent == nullptr ? glm::mat4(1.0f) : joint->parent->matrix;
    }
    else if (tmp == "CHANNELS")
    {
      stream >> joint->numChannels;

      motionData.numMotionChannels += joint->numChannels;
      joint->channelStart = _channelStart;
      _channelStart += joint->numChannels;
      joint->channelsOrder = new short[joint->numChannels];
    }
    else if (tmp == "JOINT")
    {
      Joint* tmpJoint = loadJoint(stream, joint);
      tmpJoint->parent = joint;
      joint->children.push_back(tmpJoint);
    }
    else if (tmp == "End")
    {
      stream >> tmp >> tmp;
      Joint* tmpJoint = new Joint;

      tmpJoint->parent = joint;
      tmpJoint->numChannels = 0;
      tmpJoint->name = "EndSite";
      joint->children.push_back(tmpJoint);

      stream >> tmp;
      if (tmp == "OFFSET")
        stream >> tmpJoint->offset.x >> tmpJoint->offset.y >> tmpJoint->offset.z;

      stream >> tmp;
    }
    else if (tmp == "}")
    {
      return joint;
    }
  }
  return joint;
}

void Bvh2::loadHierarchy(std::istream & stream)
{
  std::string tmp;
  while (stream.good())
  {
    stream >> tmp;
    if (trim(tmp) == "ROOT")
      rootJoint = loadJoint(stream);
    else if (trim(tmp) == "MOTION")
      loadMotion(stream);
  }
}

void Bvh2::loadMotion(std::istream & stream)
{
  std::string tmp;

  while (stream.good())
  {
    stream >> tmp;
    if (trim(tmp) == "Frames:")
    {
      stream >> motionData.numFrames;
    }
    else if (trim(tmp) == "Frame")
    {
      float frameTime;
      stream >> tmp >> frameTime;

      int numFrames = motionData.numFrames;
      int numChannels = motionData.numMotionChannels;

      motionData.data = new float[numFrames * numChannels];

      for (int frame = 0; frame < numFrames; frame++)
      {
        for (int channel = 0; channel < numChannels; channel++)
        {
          float x;
          std::stringstream ss;
          stream >> tmp;
          ss << tmp;
          ss >> x;

          int index = frame * numChannels + channel;
          motionData.data[index] = x;
        }
      }
    }
  }
}


