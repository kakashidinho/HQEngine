# - Try to find X11_xinput2 extension for X11
# Once done, this will define
#
#  X11_xinput2_FOUND - system has X11_xinput2
#  X11_xinput2_INCLUDE_DIRS - the X11_xinput2 include directories
#  X11_xinput2_LIBRARIES - link these to use X11_xinput2

include(LibFindMacros)

# Include dir
find_path(X11_xinput2_INCLUDE_DIR
  NAMES X11/extensions/XInput2.h
  PATHS ${X11_xinput2_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
find_library(X11_xinput2_LIBRARY
  NAMES Xi
  PATHS ${X11_xinput2_PKGCONF_LIBRARY_DIRS}
)


# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(X11_xinput2_PROCESS_INCLUDES X11_xinput2_INCLUDE_DIR X11_xinput2_INCLUDE_DIRS)
set(X11_xinput2_PROCESS_LIBS X11_xinput2_LIBRARY X11_xinput2_LIBRARIES)
libfind_process(X11_xinput2)
