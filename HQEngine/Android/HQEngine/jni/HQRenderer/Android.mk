LOCAL_PATH := $(call my-dir)/../../../../Source/HQRendererGL

include $(CLEAR_VARS)

LOCAL_MODULE := HQRenderer

LOCAL_SRC_FILES := Android.cpp AndroidGLES.cpp HQDeviceEnumGL.cpp HQDeviceGL.cpp HQDeviceGL_caps.cpp HQFixedFunctionShaderManagerGL.cpp HQRenderTargetFBO.cpp HQShaderGL.cpp  HQShaderGL_Common.cpp HQShaderGL_GLSLController.cpp HQShaderGL_GLSL_VarParser.cpp HQStateManagerGL.cpp HQTextureManagerGL.cpp HQVertexStreamManagerGL.cpp HQVertexStreamManagerPreShaderGL.cpp ../BaseImpl/HQTextureManagerBaseImpl.cpp ../HQEngineRenderer/HQRenderer.cpp ../HQEngineRenderer/HQRenderDeviceDebugLayer.cpp ../HQEngineRenderer/HQReturnValDebugString.cpp\
				   HQFFShaderControllerGL.cpp

LOCAL_CPP_FEATURES := exceptions rtti

LOCAL_C_INCLUDES := $(LOCAL_PATH)/..

LOCAL_ARM_MODE := arm

LOCAL_EXPORT_CFLAGS := -fvisibility=hidden

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	LOCAL_ARM_NEON := true
else
	ifeq ($(TARGET_ARCH_ABI),armeabi)
		LOCAL_EXPORT_CFLAGS += -DCMATH=1
	endif
endif

ifeq ($(NDK_DEBUG),1)
	LOCAL_EXPORT_CFLAGS += -g -ggdb -D_DEBUG=1
endif

LOCAL_CFLAGS := $(LOCAL_EXPORT_CFLAGS)

LOCAL_EXPORT_LDLIBS := -lGLESv1_CM -lGLESv2 -lstdc++

include $(BUILD_STATIC_LIBRARY)

