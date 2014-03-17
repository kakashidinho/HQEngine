# - Try to find X11_randr extension for X11
# Once done, this will define
#
#  X11_randr_FOUND - system has X11_randr
#  X11_randr_INCLUDE_DIRS - the X11_randr include directories
#  X11_randr_LIBRARIES - link these to use X11_randr

include(LibFindMacros)

# Include dir
find_path(X11_randr_INCLUDE_DIR
  NAMES X11/extensions/Xrandr.h
  PATHS ${X11_randr_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
find_library(X11_randr_LIBRARY
  NAMES Xrandr
  PATHS ${X11_randr_PKGCONF_LIBRARY_DIRS}
)


# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(X11_randr_PROCESS_INCLUDES X11_randr_INCLUDE_DIR X11_randr_INCLUDE_DIRS)
set(X11_randr_PROCESS_LIBS X11_randr_LIBRARY X11_randr_LIBRARIES)
libfind_process(X11_randr)
