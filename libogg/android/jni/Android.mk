TARGET_PLATFORM := android-3

ROOT_PATH := $(call my-dir)

########################################################################################################

include $(CLEAR_VARS)

LOCAL_MODULE     := ogg-static
LOCAL_ARM_MODE   := arm
LOCAL_PATH       := $(ROOT_PATH)/../../src
LOCAL_C_INCLUDES := $(ROOT_PATH)/../../include

LOCAL_SRC_FILES  := bitwise.c framing.c

include $(BUILD_STATIC_LIBRARY)

########################################################################################################

include $(CLEAR_VARS)

LOCAL_MODULE     := ogg
LOCAL_ARM_MODE   := arm
LOCAL_PATH       := $(ROOT_PATH)

LOCAL_WHOLE_STATIC_LIBRARIES := libogg-static

include $(BUILD_SHARED_LIBRARY)

########################################################################################################
