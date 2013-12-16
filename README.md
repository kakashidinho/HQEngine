HQEngine Game Framework
========
1. Prerequisites
----------------------------
- OpenGL headers, libraries. (For Windows Desktop version)
- NVIDIA Cg headers, libraries. (For Windows Desktop version)
- OpenAL headers, libraries. (For Windows Desktop version)
- DirectX SDK. (For Windows Desktop version)
- Visual C++ 2008. (For Windows Desktop version)
- Android SDK, NDK. (For Android)
- Cygwin. (For Android)
- Visual Studio 2012. (For Windows Metro/Phone version)
 
2. Folder structure
----------------------------
- HQEngine. (main framework folder)
	- Source. (Source code)
	- Android. (Android version)
	- Android-OpenAL-soft. (OpenAL library for Android)
	- VS. (Windows Desktop version, contains Visual Studio 2008 solution)
	- VS2012. (Windows Metro/Phone version)
		- WindowsStore. (Windows Store App version)
		- WindowsPhone. (Windows Phone 8 version)
- java2cpp. (Android framework's JNI wrapper headers)
- libogg. (lib ogg)
- libvorbis. (lib vorbis)

3. How to build
----------------------------
- Windows Desktop:
	- Open solution "/HQEngine/VS/HQEngine.sln".
	- Config Prerequisites' include and libraries paths.
	- Build and run "test" project.
- Android:
	- *TO DO*
- Windows Store App:
	- *TO DO*
- Windows Phone 8:
	- *TO DO*
	