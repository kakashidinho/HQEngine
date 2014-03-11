LOCAL_PATH := $(call my-dir)/../../../Source/HQEngine

include $(CLEAR_VARS)

LOCAL_MODULE := HQEngine

TINY_XML_SRC_PATH := $(LOCAL_PATH)/../../../ThirdParty-mod/tinyxml

LOCAL_SRC_FILES :=  HQEngineApp.cpp HQEventSeparateThread.cpp HQDefaultFileManager.cpp HQEngineCommonInternal.cpp HQEngineEffectManagerImpl.cpp HQEngineResManagerImpl.cpp android/HQAndroidGameThread.cpp android/HQEngineAppAndroid.cpp android/HQEngineJNI.cpp android/HQEngineWindowAndroid.cpp \
$(TINY_XML_SRC_PATH)/tinystr.cpp \
$(TINY_XML_SRC_PATH)/tinyxml.cpp \
$(TINY_XML_SRC_PATH)/tinyxmlerror.cpp \
$(TINY_XML_SRC_PATH)/tinyxmlparser.cpp

LOCAL_C_INCLUDES := $(LOCAL_PATH)/.. $(LOCAL_PATH)/../../../ThirdParty-mod/java2cpp

LOCAL_CPP_FEATURES := exceptions rtti

LOCAL_ARM_MODE := arm

LOCAL_STATIC_LIBRARIES := ImagesLoader

LOCAL_WHOLE_STATIC_LIBRARIES := HQUtilMath HQUtil HQRenderer

include $(BUILD_SHARED_LIBRARY)

$(call import-module,ImagesLoader)
$(call import-module,HQUtilMath)
$(call import-module,HQUtil)
$(call import-module,HQRenderer)
