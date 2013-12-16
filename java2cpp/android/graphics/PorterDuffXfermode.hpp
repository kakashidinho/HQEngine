/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.graphics.PorterDuffXfermode
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_GRAPHICS_PORTERDUFFXFERMODE_HPP_DECL
#define J2CPP_ANDROID_GRAPHICS_PORTERDUFFXFERMODE_HPP_DECL


namespace j2cpp { namespace android { namespace graphics { class Xfermode; } } }
namespace j2cpp { namespace android { namespace graphics { namespace PorterDuff_ { class Mode; } } } }


#include <android/graphics/PorterDuff.hpp>
#include <android/graphics/Xfermode.hpp>


namespace j2cpp {

namespace android { namespace graphics {

	class PorterDuffXfermode;
	class PorterDuffXfermode
		: public object<PorterDuffXfermode>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)

		explicit PorterDuffXfermode(jobject jobj)
		: object<PorterDuffXfermode>(jobj)
		{
		}

		operator local_ref<android::graphics::Xfermode>() const;


		PorterDuffXfermode(local_ref< android::graphics::PorterDuff_::Mode > const&);
	}; //class PorterDuffXfermode

} //namespace graphics
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_GRAPHICS_PORTERDUFFXFERMODE_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_GRAPHICS_PORTERDUFFXFERMODE_HPP_IMPL
#define J2CPP_ANDROID_GRAPHICS_PORTERDUFFXFERMODE_HPP_IMPL

namespace j2cpp {



android::graphics::PorterDuffXfermode::operator local_ref<android::graphics::Xfermode>() const
{
	return local_ref<android::graphics::Xfermode>(get_jobject());
}


android::graphics::PorterDuffXfermode::PorterDuffXfermode(local_ref< android::graphics::PorterDuff_::Mode > const &a0)
: object<android::graphics::PorterDuffXfermode>(
	call_new_object<
		android::graphics::PorterDuffXfermode::J2CPP_CLASS_NAME,
		android::graphics::PorterDuffXfermode::J2CPP_METHOD_NAME(0),
		android::graphics::PorterDuffXfermode::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



J2CPP_DEFINE_CLASS(android::graphics::PorterDuffXfermode,"android/graphics/PorterDuffXfermode")
J2CPP_DEFINE_METHOD(android::graphics::PorterDuffXfermode,0,"<init>","(Landroid/graphics/PorterDuff$Mode;)V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_GRAPHICS_PORTERDUFFXFERMODE_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
