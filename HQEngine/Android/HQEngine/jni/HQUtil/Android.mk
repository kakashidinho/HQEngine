LOCAL_PATH := $(call my-dir)/../../../../Source/HQUtil

include $(CLEAR_VARS)

LOCAL_MODULE := HQUtil

LOCAL_SRC_FILES := pthread/HQMutexPthread.cpp pthread/HQSemaphorePthread.cpp pthread/HQThreadPthread.cpp linux/HQTimerLinux.cpp HQLogStream.cpp\
				   pthread/HQConditionVariablePThread.cpp

LOCAL_CPP_FEATURES := exceptions rtti

LOCAL_ARM_MODE := arm

LOCAL_EXPORT_CFLAGS += -fvisibility=hidden

ifeq ($(NDK_DEBUG),1)
	LOCAL_EXPORT_CFLAGS += -g -ggdb  -D_DEBUG=1
endif

LOCAL_CFLAGS := $(LOCAL_EXPORT_CFLAGS)

LOCAL_EXPORT_LDLIBS := -llog

include $(BUILD_STATIC_LIBRARY)
