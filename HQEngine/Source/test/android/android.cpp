#include <jni.h>
#include <unistd.h>

extern "C"
{
	JNIEXPORT void JNICALL Java_hqengineTest_test_TestActivity_onCreateNative(JNIEnv *env, jobject thiz, jstring resourcePath)
	{
		/*-----------change to resource directory---------------*/
		//copy resource path
		const char *nativeString = env->GetStringUTFChars(resourcePath, 0);

		chdir(nativeString);
		chdir("test");
		
		env->ReleaseStringUTFChars(resourcePath, nativeString);
	}
}