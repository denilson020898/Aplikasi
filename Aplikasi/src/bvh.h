#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <set>
#include <sstream>
#include <algorithm>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#define Xposition 0x01
#define Yposition 0x02
#define Zposition 0x04
#define Zrotation 0x10
#define Xrotation 0x20
#define Yrotation 0x40


#include <algorithm>
#include <string>
#include <cctype>
#include <functional>

namespace k
{

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

}

namespace k
{

  typedef struct
  {
    float x, y, z;
  } OFFSET;

  typedef struct JOINT JOINT;

  struct JOINT
  {
    // joint name
    const char* name = NULL;

    // joint parent
    JOINT* parent = NULL;

    // offset data
    OFFSET offset;

    // num of channels joint has
    unsigned int num_channels = 0;

    // binary or-ed channels - obsolete
//    short channels;

    // ordered list of channels
    short* channels_order = NULL;

    // joint's children
    std::vector<JOINT*> children;

    // local transofrmation matrix (premultiplied with parents'
    glm::mat4 matrix;

    // index of joint's channel data in motion array
    unsigned int channel_start = 0;
  };

  typedef struct
  {
    JOINT* rootJoint;
    int num_channels;
  } HIERARCHY;

  typedef struct
  {
    unsigned int num_frames;            // koliko frameova motion ima
    unsigned int num_motion_channels = 0;   // koliko channela je upisan u jedan frame
    float* data = NULL;                        // pocetak motion float-ova
    unsigned* joint_channel_offsets;    // broj kanala od pocetka hijerarhije za i-ti joint
  } MOTION;

  typedef struct
  {
    unsigned int num_hierarchies;
    unsigned int num_total_joints;
    unsigned int num_motion_channels = 0;
    unsigned int num_frames;

  } bvh_statistics;

  /**
   * @brief The Bvh class
   */

  class Bvh
  {
    JOINT* loadJoint(std::istream& stream, JOINT* parent = NULL);
    void loadHierarchy(std::istream& stream);
    void loadMotion(std::istream& stream);

  public:
    Bvh();
    ~Bvh();

    void load(const std::string& filename);
    void testOutput() const;
    void printJoint(const JOINT* const joint) const;

    void moveTo(unsigned frame);

    const JOINT* getRootJoint() const { return rootJoint; }
    unsigned getNumFrames() const { return motionData.num_frames; }
  private:
    JOINT* rootJoint;

    // parsiranje je napravljeno za set ali set ne spasava u redu u kojem su upisani
    // tj. postojani u MOTION dijelu
//    std::set<JOINT*> allJoints;
    MOTION motionData;
    //    bvh_statistics stats;
  };

}
