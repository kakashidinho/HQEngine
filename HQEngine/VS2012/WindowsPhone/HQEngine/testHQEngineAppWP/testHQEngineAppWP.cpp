#include "pch.h"
#define HQ_CUSTOM_MALLOC
#define HQ_CUSTOM_NEW_OPERATOR
#include "HQEngineCustomHeap.h"
#include "HQEngineApp.h"
#include "HQEngine\winstore\HQWinStoreFileSystem.h"

HQEngineApp *app;

class KeyListener: public HQEngineKeyListener
{
public:
	 void KeyReleased(HQKeyCodeType keyCode)
	 {
		 switch (keyCode)
		 {
		 case HQKeyCode::ESCAPE:
			 app->Stop();
			 break;
		 }
	 }
};

class MouseListener: public HQEngineMouseListener
{
public:
	void MouseMove(const HQPointi &point)
	{
		char buffer[256];
		sprintf_s(buffer, 256, "mouse moved: %d, %d\n", point.x, point.y);
		OutputDebugStringA(buffer);
	}
};

class Renderer: public HQEngineRenderDelegate
{
public:
	 void Render(HQTime dt)
	 {
		 HQColor red = {1, 0, 0, 1};
		 app->GetRenderDevice()->SetClearColor(red);
		 app->GetRenderDevice()->Clear(HQ_TRUE, HQ_FALSE, HQ_FALSE);
	 }
};

#ifdef _M_ARM
extern"C" void oamcpy(void *dst, const void *src, size_t nn);
extern "C" int hello(int a, int b);
extern "C" float* HQNEONVECTOR4NORMALIZE2(const float* v , float *normalizedVec);
extern "C" void HQNeonVector4Normalize3(const float* v , float *normalizedVec);
extern "C" void HQNeonVector4Normalize(const float* v , float *normalizedVec);
#endif

int HQEngineMain (int argc, char **argv)
{
	int *p = HQ_NEW int;

	int a = 0;
#ifdef _M_ARM
	int c = hello(a, a);

	oamcpy(p, &a, sizeof(int));
#endif
	HQVector4 vec(1, 2, 3);

	float x = vec.x;

	HQVector4 *re;

	HQVector4 vec2(1, 2, 3);

#ifdef _M_ARM
	re = (HQVector4*)HQNEONVECTOR4NORMALIZE2(vec, vec);

	HQNeonVector4Normalize(vec2, vec2);
#endif

	re = &vec2;

	HQA16ByteMatrix4Ptr mat(1, 2, 3, 4,
							5, 6, 7, 8,
							9, 10, 11, 12,
							13, 14, 15, 16);

	HQA16ByteMatrix3x4Ptr mat2(1, 2, 3, 4,
							5, 6, 7, 8,
							9, 10, 11, 12);

	mat->Transpose();

	HQA16ByteVector4Ptr vec3(1, 2, 3);

	vec3->Normalize();

	re = vec3;

	auto sizeVec = sizeof(HQVector4);

	HQA16ByteQuaternionPtr quat;
	quat->QuatFromRotAxisOx(HQPiFamily::_PIOVER3);

	quat->QuatToMatrix4r(mat);

	quat->QuatToMatrix3x4c(mat2);

	HQA16ByteQuaternionPtr quat2;
	quat2->QuatFromMatrix3x4c(*mat2);

	HQA16ByteQuaternionPtr quat3;
	quat2->QuatFromMatrix4r(*mat);

	float det;
	HQMatrix4Inverse(mat, &det, mat);

	HQA16ByteMatrix4Ptr mat2Inv;
	HQMatrix3x4Inverse(mat2, mat2Inv);

	auto identity = *mat2Inv * *mat2  ;

	float length = quat->GetMagnitude();

	//test file system
	auto file = HQWinStoreFileSystem::OpenFileForRead("Assets/input.txt");
	char buffer[256];
	if (file)
	{
		HQWinStoreFileSystem::fread(buffer, 256, 1, file);
		fclose(file);
	}


	app = HQEngineApp::CreateInstance();
	app->InitWindow();

	app->SetRenderDelegate(Renderer());

	app->SetKeyListener(KeyListener());

	app->SetMouseListener(MouseListener());

	app->EnableMouseCursor(false);

	app->Run();

	app->Release();

	return 0;
}
