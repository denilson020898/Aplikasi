workspace "Aplikasi"
  architecture "x64"
  startproject "Aplikasi"
  configurations
  {
    "Debug",
    "Release"
  }

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

IncludeDir = {}
IncludeDir["GLFW"] = "Aplikasi/vendor/GLFW/include"
IncludeDir["Glad"] = "Aplikasi/vendor/Glad/include"
IncludeDir["ImGui"] = "Aplikasi/vendor/ImGui"
IncludeDir["glm"] = "Aplikasi/vendor/glm"

LibDir = {}
LibDir["GLFW"] = "Aplikasi/vendor/GLFW/lib/%{cfg.buildcfg}"

project "Aplikasi"
  location "Aplikasi"
  kind "ConsoleApp"
  language "C++"
  staticruntime "off"

  targetdir ("bin/" .. outputdir .. "/%{prj.name}")
  objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

  files
  {
    "%{prj.name}/src/**.h",
    "%{prj.name}/src/**.cpp",
    "%{prj.name}/vendor/glm/glm/**.hpp",
    "%{prj.name}/vendor/glm/glm/**.inl",
    "%{prj.name}/vendor/Glad/src/**.c"
  }

  includedirs
  {
    "%{prj.name}/src",
    "%{IncludeDir.GLFW}",
    "%{IncludeDir.Glad}",
    "%{IncludeDir.glm}",
    "%{IncludeDir.ImGui}"
  }

  libdirs
  {
    "%{LibDir.GLFW}"
  }

  links
  {
    "opengl32.lib",
    "glfw3.lib"
  }

  filter "system:windows"
    --cppdialect "C++17"
    systemversion "latest"

    defines
    {
    }

  filter "configurations:Debug"
    runtime "Debug"
    symbols "on"

  filter "configurations:Release"
    runtime "Release"
    optimize "on"