APP_ABI := x86 armeabi armeabi-v7a
APP_STL := gnustl_static

ifeq ($(NDK_DEBUG),1)
	APP_OPTIM := debug
else
	APP_OPTIM := release
endif

APP_PLATFORM := android-7