HQEngine Game Framework
========
Description and Goals
----------------------------
- This project goal is a cross platform C++ 3D game framework, allowing as much code re-use as possible across many platforms (Windows, Windows Phone, Android, IOS, Mac OSX, ...).
- To achieve this, many platform-specific functionalities have been hidden by abstract layers, including rendering API, audio API, file system, window API, threading, etc.

Current status
----------------------------
- Implemented parts:
	* Maths library. (supports SSE and Neon SIMD optimizations)
	* Images loader. (supports various file formats including compressed ones used by OpenGL and Direct3D like DXT and ETC)
	* Rendering API layer. (supports Direct3D (9/11), OpenGL (ES) back-ends)
	* Audio layer. (supports OpenAL and XAudio 2 back-ends)
	* Single code-base for Window creation and Game loop. (hides the setting up of window in various platforms)
	* Threading system. (supports win32, C++11, pthread back-ends)
	* Template hash table, memory alignment, linked list, stack.  (this removes the dependence of C++ STL in some platforms not supporting it, or not supporting C++11)
	* Skeleton animation. 
	* Scene graph.
- The current main maintained platform are Windows and its variances (Phone/Metro). Android has also been maintained occasionally. IOS and Mac OSX currently have fewer maintenances due to limited resources. Linux was supported initially, but dropped because of limited documents about low level APIs.
	
Sample status
----------------------------
- Included Reflective Shadow Map sample. 
	* This sample implements a global illumination approximating algorithm. Currently it runs very inefficiently, since a lot of operations are done at every pixels. 

Prerequisites
----------------------------
- For Windows Desktop version:
	* OpenGL headers, libraries. 
	* NVIDIA Cg 3.1 headers, libraries. Best way is installing Cg Toolkit 3.1 from NVIDIA. The pre-built libraries and headers are already included for Win32 platform.
	* OpenAL headers, libraries. Best way is installing OpenAL SDK from Creative Lab. 
	* DirectX SDK June 2010. 
	* Visual C++ 2008. 
- For Android:
	* Android SDK, NDK. 
	* Cygwin. (If using Windows)
- For Windows Metro/Phone version:
	* Visual Studio 2012 for Windows 8/Windows Phone 8. 
 
Folder structure
----------------------------
- HQEngine. (main framework folder)
	* Source. (Source code)
	* Android. (Android version)
	* Android-OpenAL-soft. (OpenAL library for Android)
	* VS. (Windows Desktop version, contains Visual Studio 2008 solution)
	* VS2012. (Windows Metro/Phone versions)
		* WindowsStoreApp. (Windows Store App version)
		* WindowsPhone. (Windows Phone 8 version)
- Cg. (Cg headers and pre-built libraries for Win32)
- glsl_optimizer. (glsl optimizer headers and pre-built libraries for Win32)
- hlsl2glsl.	(Cg to glsl translator's headers and pre-built libraries for Win32)
- java2cpp. (Android framework's JNI wrapper headers)
- libogg. (lib ogg)
- libvorbis. (lib vorbis)
- HQShaderCompiler. (some utility codes for compiling shader code in HQEngine format)
- samples. (contains various sample codes)

How to build
----------------------------
- Windows Desktop:
	* Open 3D Math library's solution "/HQEngine/VS/HQEngineUtilMath.sln".
	* Select Build->Batch Build->Select All->Build to build various versions (Debug/Release) of Math library.
	* Open game framework's solution "/HQEngine/VS/HQEngine.sln". 
	Note: it's normal if there's a error popup saying that a C sharp project cannot be opened because you are using Visual C++ Express.
	* Config Prerequisites' include and libraries paths. 
	* Build solution.
	* Output dll and exe files are in "/HQEngine/VS/Output/Debug" folder. 
	* Run "test" project inside Visual Studio. Don't run "test.exe" directly in "/HQEngine/VS/Output/Debug" folder.
- Windows Store App:
	* Open Windows Store App's solution "/HQEngine/VS2012/WindowsStoreApp/HQEngine/HQEngine.sln". 
	Note: Be sure to use Windows 8 version if you are using VS Express.
	* Build and run "test" project.
- Windows Phone 8:
	* Open Windows Phone's solution "/HQEngine/VS2012/WindowsPhone/HQEngine/HQEngine.sln". 
	Note: Be sure to use Windows Phone version if you are using VS Express.
	* Build and run "testWP" project.
- Android:
	* Open Cygwin. (If using Windows)
	* Build .so file for the "test" project:
		* Set "ndk_build_script" environment variable to make it point to the path of your ndk-build script. 
		For example: "export ndk_build_script=/cygdrive/e/android-ndk-r7b/ndk-build".
		* Go to "/HQEngine/Android/test" folder.
		* Run "build_so.sh" script to build the .so file for the test project.
	* Import and build Android Eclipse project in "HQEngine/Android/HQEngine" folder.
	* Run script "HQEngine/Android/test/push_res_sd.sh" to push resources to Android device.
	* Import, build and run Android Eclipse project in "HQEngine/Android/test" folder.
	
License
---------------------------
- This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.
	
	