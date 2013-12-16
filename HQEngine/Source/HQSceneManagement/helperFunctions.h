#ifndef HELPER_FUNC
#define HELPER_FUNC

#include "../HQPlatformDef.h"

namespace HQSceneManagementHelper
{
char * GetContainingDir(const char *file);//returned pointer should to be deleted
void SetCurrentDir(const char *dir);

#if defined WIN32 || defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
wchar_t * GetCurrentDir();//returned pointer should be deleted
void SetCurrentDir(const wchar_t *dir);
#else
char * GetCurrentDir();//returned pointer should be deleted
#endif
};//namespace HQSceneManagementHelper

#endif