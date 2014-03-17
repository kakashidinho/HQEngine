LOCAL_PATH := $(call my-dir)/../../../../Source/HQUtilMath

include $(CLEAR_VARS)

LOCAL_MODULE := HQUtilMath

LOCAL_SRC_FILES := HQUtilMathPCH.cpp HQAABB.cpp HQBSP.cpp HQMatrix3x4.cpp HQMatrix4.cpp HQOBB.cpp HQPlane.cpp HQPolygon.cpp HQPrimeNumber.cpp HQQuaternion.cpp HQRay.cpp HQSphere.cpp HQVector.cpp 
				   

LOCAL_FILTER_ASM := cp

LOCAL_CPP_FEATURES := exceptions

LOCAL_ARM_MODE := arm

LOCAL_EXPORT_CFLAGS += -fvisibility=hidden

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_ARM_NEON := true
else
	ifeq ($(TARGET_ARCH_ABI),armeabi)
		LOCAL_EXPORT_CFLAGS += -DHQ_CMATH=1
	endif
endif

ifeq ($(NDK_DEBUG),1)
	LOCAL_EXPORT_CFLAGS += -g -ggdb  -D_DEBUG=1
else
endif

LOCAL_CFLAGS := $(LOCAL_EXPORT_CFLAGS)

include $(BUILD_STATIC_LIBRARY)

