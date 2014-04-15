#include "Game.h"

#ifdef WIN32
#if defined _DEBUG || defined DEBUG
#	ifdef _STATIC_CRT
#		pragma comment(lib , "../../VS/Output/Debug static CRT/HQEngineRendererD.lib")
#		pragma comment(lib,"../../VS/Output/Debug static CRT/HQUtilD.lib")
#		pragma comment(lib,"../../VS/Output/StaticDebug static CRT/HQUtilMathD.lib")
#	else
#		pragma comment(lib , "../../VS/Output/Debug/HQEngineRendererD.lib")
#		pragma comment(lib,"../../VS/Output/StaticDebug/HQUtilMathD.lib")
#		pragma comment(lib,"../../VS/Output/Debug/HQUtilD.lib")
#	endif
#elif defined _STATIC_CRT
#	pragma comment(lib , "../../VS/Output/Release static CRT/HQEngineRenderer.lib")
#	pragma comment(lib,"../../VS/Output/StaticRelease static CRT/HQUtilMath.lib")
#	pragma comment(lib,"../../VS/Output/Release static CRT/HQUtil.lib")
#else
#	pragma comment(lib , "../../VS/Output/Release/HQEngineRenderer.lib")
#	pragma comment(lib,"../../VS/Output/StaticRelease/HQUtilMath.lib")
#	pragma comment(lib,"../../VS/Output/Release/HQUtil.lib")
#endif
#endif

Game *game = NULL;

#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT void JNICALL Java_hqenginespriteseditor_MyCanvas_nativeInitRenderer
  (JNIEnv *env, jobject canvas)
{
	game = new Game(env, canvas);
}

//for testing
#ifdef WIN32
JNIEXPORT void JNICALL Java_hqenginespriteseditor_MyCanvas_nativeInitRendererWin32
	(HWND hwnd)
{
	game = new Game(hwnd);
}
#endif

JNIEXPORT void JNICALL Java_hqenginespriteseditor_MyCanvas_nativePaint
  (JNIEnv *env, jobject canvas)
{
	game->Paint();
}

JNIEXPORT void JNICALL Java_hqenginespriteseditor_MyCanvas_nativeResize
  (JNIEnv *env, jobject object, jint width, jint height)
{
	if (game)
		game->OnWindowResized(width, height);
}

JNIEXPORT jint JNICALL Java_hqenginespriteseditor_MyCanvas_nativeGetWorldX
  (JNIEnv *env, jobject object, jint x)
{
	if (game == NULL)
		return 0;
	const HQPointi worldOrigin = game->GetWorldOrigin();
	return x - worldOrigin.x;
}

JNIEXPORT jint JNICALL Java_hqenginespriteseditor_MyCanvas_nativeGetWorldY
  (JNIEnv *env, jobject object, jint y)
{
	if (game == NULL)
		return 0;
	const HQPointi worldOrigin = game->GetWorldOrigin();
	return y - worldOrigin.y;
}

JNIEXPORT jboolean JNICALL Java_hqenginespriteseditor_MyCanvas_nativeLoadTexture
  (JNIEnv *env, jobject object, jstring jtextureFile)
{
	jboolean isCopy;
	const char *textureFile = env->GetStringUTFChars(jtextureFile, &isCopy);
	
	bool re = game->LoadTexture(textureFile);


	env->ReleaseStringUTFChars(jtextureFile, textureFile);
	return re;
}

JNIEXPORT void JNICALL Java_hqenginespriteseditor_MyCanvas_nativeReleaseRenderer
  (JNIEnv *env, jobject object)
{
	delete game ;
	game = NULL;

}




#ifdef __cplusplus
}
#endif