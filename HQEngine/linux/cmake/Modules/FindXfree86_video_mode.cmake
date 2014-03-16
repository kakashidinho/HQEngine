# - Try to find Xf86vmode extension for X11
# Once done, this will define
#
#  Xf86vmode_FOUND - system has Magick++
#  Xf86vmode_INCLUDE_DIRS - the Magick++ include directories
#  Xf86vmode_LIBRARIES - link these to use Magick++

include(LibFindMacros)

# Include dir
find_path(Xfree86_video_mode_INCLUDE_DIR
  NAMES X11/extensions/xf86vmode.h
  PATHS ${Xfree86_video_mode_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
find_library(Xfree86_video_mode_LIBRARY
  NAMES Xxf86vm
  PATHS ${Xfree86_video_mode_PKGCONF_LIBRARY_DIRS}
)


# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(Xfree86_video_mode_PROCESS_INCLUDES Xfree86_video_mode_INCLUDE_DIR Xfree86_video_mode_INCLUDE_DIRS)
set(Xfree86_video_mode_PROCESS_LIBS Xfree86_video_mode_LIBRARY Xfree86_video_mode_LIBRARIES)
libfind_process(Xfree86_video_mode)
