LOCAL_PATH := $(call my-dir)/../../../Source/HQUtilMath

include $(CLEAR_VARS)

LOCAL_MODULE := HQUtilMath

LOCAL_SRC_FILES := HQAABB.cpp HQBSP.cpp HQMatrix3x4.cpp HQMatrix4.cpp HQOBB.cpp HQPlane.cpp HQPolygon.cpp HQPrimeNumber.cpp HQQuaternion.cpp HQRay.cpp HQSphere.cpp HQVector.cpp

LOCAL_CPP_FEATURES := exceptions

LOCAL_ARM_MODE := arm

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_ARM_NEON := true
else
    LOCAL_CFLAGS := -DCMATH=1
endif

include $(BUILD_SHARED_LIBRARY)

