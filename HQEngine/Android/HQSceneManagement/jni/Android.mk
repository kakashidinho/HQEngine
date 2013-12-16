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

#########################################################

LOCAL_PATH := $(ROOT_PATH)/../../../Source/HQSceneManagement

include $(CLEAR_VARS)

LOCAL_MODULE := HQSceneManagement

LOCAL_C_INCLUDES := 

LOCAL_SRC_FILES := helperFunctions.cpp HQMeshNode.cpp HQMeshNode_Animation.cpp HQSceneNode.cpp

LOCAL_CPP_FEATURES := exceptions

LOCAL_ARM_MODE := arm

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_ARM_NEON := true
else
	ifeq ($(TARGET_ARCH_ABI),armeabi)
		LOCAL_CFLAGS += -DCMATH=1
	endif
endif

LOCAL_CFLAGS := -fvisibility=hidden

ifeq ($(NDK_DEBUG),1)
	LOCAL_CFLAGS += -g -ggdb  -D_DEBUG=1
endif

LOCAL_SHARED_LIBRARIES := HQEngine-prebuilt

LOCAL_LDLIBS := -llog

include $(BUILD_SHARED_LIBRARY)
