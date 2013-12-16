#ifndef NATIVE_WINDOW_H
#define NATIVE_WINDOW_H

#ifdef __APPLE__
#include <JavaVM/jni.h>
#else
#include <jni.h>
#include <jawt_md.h>
#endif

#include "../HQRenderer.h"

#ifdef WIN32
#	define EXPORT_GAME_CLASS 1
#	if EXPORT_GAME_CLASS
#		if HQSPRITESEDITOR_EXPORTS
#			define GAME_API __declspec(dllexport)
#		else
#			define GAME_API __declspec(dllimport)
#		endif
#	else
#		define GAME_API
#	endif
#else
#	define GAME_API
#endif

class GAME_API Game: public HQA16ByteObject {
public:
	Game(JNIEnv *env, jobject canvas);
#ifdef WIN32
	Game(HWND hwnd);
#endif
	~Game();

	void Paint();
	bool LoadTexture(const char *file);
	void OnWindowResized(hquint32 width, hquint32 height);

	HQRenderer & GetRenderer() {return renderer;}
	const HQPoint<hqint32> &GetWorldOrigin() const {return worldOrigin;}
private:

	void Init();

	void CreateViewProjMatrix();

	void DrawRect(HQRectf rect, hquint32 texture);
#ifdef WIN32
#	pragma warning(push)
#	pragma warning(disable:4251)
#endif
	HQMatrix4 viewprojMatrix;
	HQMatrix4 worldMatrix;

	HQPoint<hqint32> worldOrigin;

	HQRenderer renderer;
	
	hquint32 texture;
	hquint32 program;
	hquint32 vinputlayout;
	hquint32 quadVbuffer;

#ifdef WIN32
#	pragma warning(pop)
#endif
};

#endif