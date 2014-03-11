TARGET_PLATFORM := android-3

ROOT_PATH := $(call my-dir)

########################################################################################################
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

########################################################################################################

include $(CLEAR_VARS)

LOCAL_MODULE     := vorbis-static
LOCAL_ARM_MODE   := arm
LOCAL_PATH       := $(ROOT_PATH)/../../lib
LOCAL_C_INCLUDES := $(ROOT_PATH)/../../include $(ROOT_PATH)/../../../libogg/include

LOCAL_SRC_FILES  := analysis.c bitrate.c block.c codebook.c envelope.c floor0.c floor1.c info.c lookup.c lpc.c lsp.c mapping0.c mdct.c psy.c registry.c res0.c sharedbook.c smallft.c synthesis.c vorbisenc.c window.c vorbisfile.c

include $(BUILD_STATIC_LIBRARY)

########################################################################################################

include $(CLEAR_VARS)

LOCAL_MODULE     := vorbis
LOCAL_ARM_MODE   := arm
LOCAL_PATH       := $(ROOT_PATH)

LOCAL_STATIC_LIBRARIES := ogg-static-prebuilt

LOCAL_WHOLE_STATIC_LIBRARIES := vorbis-static

#LOCAL_ALLOW_UNDEFINED_SYMBOLS := true

include $(BUILD_SHARED_LIBRARY)

########################################################################################################
