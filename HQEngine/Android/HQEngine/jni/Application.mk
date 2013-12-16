APP_ABI := x86 armeabi-v7a armeabi
APP_STL := gnustl_static
#APP_CPPFLAGS += -fexceptions #ndk-r6 style
APP_PLATFORM := android-7

ifeq ($(NDK_DEBUG),1)
	APP_OPTIM := debug
else
	APP_OPTIM := release
endif
