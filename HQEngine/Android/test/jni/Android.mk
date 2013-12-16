ROOT_PATH := $(call my-dir)
####################### prebuild libs ###############################
LOCAL_PATH := $(ROOT_PATH)

include $(CLEAR_VARS)
LOCAL_MODULE := HQEngine_prebuilt

ifeq ($(NDK_DEBUG),1)
	LIB_PATH := ../../HQEngine/obj/local/
else
	LIB_PATH := ../../HQEngine/libs
endif

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_SRC_FILES := $(LIB_PATH)/armeabi-v7a/libHQEngine.so
else
	ifeq ($(TARGET_ARCH_ABI),armeabi)
		LOCAL_SRC_FILES := $(LIB_PATH)/armeabi/libHQEngine.so
	else
		LOCAL_SRC_FILES := $(LIB_PATH)/x86/libHQEngine.so
	endif
endif

include $(PREBUILT_SHARED_LIBRARY)

#####################################
LOCAL_PATH := $(ROOT_PATH)

include $(CLEAR_VARS)
LOCAL_MODULE := HQAudio_prebuilt

ifeq ($(NDK_DEBUG),1)
	LIB_PATH := ../../HQAudio/obj/local/
else
	LIB_PATH := ../../HQAudio/libs
endif

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_SRC_FILES := $(LIB_PATH)/armeabi-v7a/libHQAudio.so
else
    ifeq ($(TARGET_ARCH_ABI),armeabi)
		LOCAL_SRC_FILES := $(LIB_PATH)/armeabi/libHQAudio.so
	else
		LOCAL_SRC_FILES := $(LIB_PATH)/x86/libHQAudio.so
	endif
endif

include $(PREBUILT_SHARED_LIBRARY)

#####################################
LOCAL_PATH := $(ROOT_PATH)

include $(CLEAR_VARS)
LOCAL_MODULE := HQSceneManagement_prebuilt

ifeq ($(NDK_DEBUG),1)
	LIB_PATH := ../../HQSceneManagement/obj/local/
else
	LIB_PATH := ../../HQSceneManagement/libs
endif

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_SRC_FILES := $(LIB_PATH)/armeabi-v7a/libHQSceneManagement.so
else
    ifeq ($(TARGET_ARCH_ABI),armeabi)
		LOCAL_SRC_FILES := $(LIB_PATH)/armeabi/libHQSceneManagement.so
	else
		LOCAL_SRC_FILES := $(LIB_PATH)/x86/libHQSceneManagement.so
	endif
endif

include $(PREBUILT_SHARED_LIBRARY)

################# shared library #############################
LOCAL_PATH := $(ROOT_PATH)/../../../Source/test

include $(CLEAR_VARS)

LOCAL_MODULE := test

LOCAL_SRC_FILES := Game.cpp winmain.cpp android/android.cpp

LOCAL_C_INCLUDES := $(ROOT_PATH)/../../../Source

LOCAL_CPP_FEATURES := exceptions

LOCAL_ARM_MODE := arm

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_ARM_NEON := true
	LOCAL_CFLAGS := -DGLES2=1
else
    ifeq ($(TARGET_ARCH_ABI),armeabi)
		LOCAL_CFLAGS += -DCMATH=1
	else
		LOCAL_CFLAGS := -DGLES2=1
	endif
endif

LOCAL_CFLAGS += -g -ggdb

ifeq ($(NDK_DEBUG),1)
	LOCAL_CFLAGS += -D_DEBUG=1
endif

LOCAL_LDLIBS := -llog

LOCAL_SHARED_LIBRARIES := HQEngine_prebuilt HQAudio_prebuilt HQSceneManagement_prebuilt

include $(BUILD_SHARED_LIBRARY)
