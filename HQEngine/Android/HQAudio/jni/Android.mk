#######################prebuild libs############################

############  HQEngine ##############
ROOT_PATH := $(call my-dir)
LOCAL_PATH := $(ROOT_PATH)

include $(CLEAR_VARS)
LOCAL_MODULE := HQEngine-prebuilt
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_SRC_FILES := ../../HQEngine/libs/armeabi-v7a/libHQEngine.so
else
	ifeq ($(TARGET_ARCH_ABI),armeabi)
		LOCAL_SRC_FILES := ../../HQEngine/libs/armeabi/libHQEngine.so
	else
		LOCAL_SRC_FILES := ../../HQEngine/libs/x86/libHQEngine.so
	endif
endif

include $(PREBUILT_SHARED_LIBRARY)

############  open AL ##############
LOCAL_PATH := $(ROOT_PATH)

include $(CLEAR_VARS)
LOCAL_MODULE := openal-static-prebuilt
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_SRC_FILES := ../prebuild-libs/armeabi-v7a/libopenal-static.a
else
    ifeq ($(TARGET_ARCH_ABI),armeabi)
		LOCAL_SRC_FILES := ../prebuild-libs/armeabi/libopenal-static.a
	else
		LOCAL_SRC_FILES := ../prebuild-libs/x86/libopenal-static.a
	endif
endif

include $(PREBUILT_STATIC_LIBRARY)

############# lib ogg ##############
LOCAL_PATH := $(ROOT_PATH)

include $(CLEAR_VARS)
LOCAL_MODULE := ogg-static-prebuilt
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_SRC_FILES := ../prebuild-libs/armeabi-v7a/libogg-static.a
else
    ifeq ($(TARGET_ARCH_ABI),armeabi)
		LOCAL_SRC_FILES := ../prebuild-libs/armeabi/libogg-static.a
	else
		LOCAL_SRC_FILES := ../prebuild-libs/x86/libogg-static.a
	endif
endif

include $(PREBUILT_STATIC_LIBRARY)

############# lib vorbis ##############
LOCAL_PATH := $(ROOT_PATH)

include $(CLEAR_VARS)
LOCAL_MODULE := vorbis-static-prebuilt
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_SRC_FILES := ../prebuild-libs/armeabi-v7a/libvorbis-static.a
else
    ifeq ($(TARGET_ARCH_ABI),armeabi)
		LOCAL_SRC_FILES := ../prebuild-libs/armeabi/libvorbis-static.a
	else
		LOCAL_SRC_FILES := ../prebuild-libs/x86/libvorbis-static.a
	endif
endif

include $(PREBUILT_STATIC_LIBRARY)

#########################################################

LOCAL_PATH := $(ROOT_PATH)/../../../Source/HQAudio

include $(CLEAR_VARS)

LOCAL_MODULE := HQAudio

LOCAL_C_INCLUDES := $(ROOT_PATH)/../../../../ThirdParty-mod/Android-OpenAL-soft/android/include/AL $(ROOT_PATH)/../../../../ThirdParty-mod/libogg/include $(ROOT_PATH)/../../../../ThirdParty-mod/libvorbis/include

LOCAL_SRC_FILES := HQAudioBase.cpp openAL/HQAudioAL.cpp openAL/HQAudioStreamBufferAL.cpp HQAudioInternal.cpp

LOCAL_FILTER_ASM := cp

LOCAL_CPP_FEATURES := exceptions

LOCAL_ARM_MODE := arm

LOCAL_CFLAGS := -fvisibility=hidden

ifeq ($(NDK_DEBUG),1)
	LOCAL_CFLAGS += -g -ggdb  -D_DEBUG=1
endif

LOCAL_STATIC_LIBRARIES := openal-static-prebuilt vorbis-static-prebuilt

LOCAL_SHARED_LIBRARIES := HQEngine-prebuilt

LOCAL_WHOLE_STATIC_LIBRARIES := ogg-static-prebuilt

LOCAL_LDLIBS := -llog

include $(BUILD_SHARED_LIBRARY)
