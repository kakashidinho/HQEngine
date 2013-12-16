/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.text.style.AbsoluteSizeSpan
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_TEXT_STYLE_ABSOLUTESIZESPAN_HPP_DECL
#define J2CPP_ANDROID_TEXT_STYLE_ABSOLUTESIZESPAN_HPP_DECL


namespace j2cpp { namespace android { namespace text { class TextPaint; } } }
namespace j2cpp { namespace android { namespace text { namespace style { class MetricAffectingSpan; } } } }
namespace j2cpp { namespace android { namespace text { class ParcelableSpan; } } }
namespace j2cpp { namespace android { namespace os { class Parcel; } } }


#include <android/os/Parcel.hpp>
#include <android/text/ParcelableSpan.hpp>
#include <android/text/TextPaint.hpp>
#include <android/text/style/MetricAffectingSpan.hpp>


namespace j2cpp {

namespace android { namespace text { namespace style {

	class AbsoluteSizeSpan;
	class AbsoluteSizeSpan
		: public object<AbsoluteSizeSpan>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)
		J2CPP_DECLARE_METHOD(5)
		J2CPP_DECLARE_METHOD(6)
		J2CPP_DECLARE_METHOD(7)
		J2CPP_DECLARE_METHOD(8)
		J2CPP_DECLARE_METHOD(9)

		explicit AbsoluteSizeSpan(jobject jobj)
		: object<AbsoluteSizeSpan>(jobj)
		{
		}

		operator local_ref<android::text::style::MetricAffectingSpan>() const;
		operator local_ref<android::text::ParcelableSpan>() const;


		AbsoluteSizeSpan(jint);
		AbsoluteSizeSpan(jint, jboolean);
		AbsoluteSizeSpan(local_ref< android::os::Parcel > const&);
		jint getSpanTypeId();
		jint describeContents();
		void writeToParcel(local_ref< android::os::Parcel >  const&, jint);
		jint getSize();
		jboolean getDip();
		void updateDrawState(local_ref< android::text::TextPaint >  const&);
		void updateMeasureState(local_ref< android::text::TextPaint >  const&);
	}; //class AbsoluteSizeSpan

} //namespace style
} //namespace text
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_TEXT_STYLE_ABSOLUTESIZESPAN_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_TEXT_STYLE_ABSOLUTESIZESPAN_HPP_IMPL
#define J2CPP_ANDROID_TEXT_STYLE_ABSOLUTESIZESPAN_HPP_IMPL

namespace j2cpp {



android::text::style::AbsoluteSizeSpan::operator local_ref<android::text::style::MetricAffectingSpan>() const
{
	return local_ref<android::text::style::MetricAffectingSpan>(get_jobject());
}

android::text::style::AbsoluteSizeSpan::operator local_ref<android::text::ParcelableSpan>() const
{
	return local_ref<android::text::ParcelableSpan>(get_jobject());
}


android::text::style::AbsoluteSizeSpan::AbsoluteSizeSpan(jint a0)
: object<android::text::style::AbsoluteSizeSpan>(
	call_new_object<
		android::text::style::AbsoluteSizeSpan::J2CPP_CLASS_NAME,
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_NAME(0),
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



android::text::style::AbsoluteSizeSpan::AbsoluteSizeSpan(jint a0, jboolean a1)
: object<android::text::style::AbsoluteSizeSpan>(
	call_new_object<
		android::text::style::AbsoluteSizeSpan::J2CPP_CLASS_NAME,
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_NAME(1),
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}



android::text::style::AbsoluteSizeSpan::AbsoluteSizeSpan(local_ref< android::os::Parcel > const &a0)
: object<android::text::style::AbsoluteSizeSpan>(
	call_new_object<
		android::text::style::AbsoluteSizeSpan::J2CPP_CLASS_NAME,
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_NAME(2),
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_SIGNATURE(2)
	>(a0)
)
{
}


jint android::text::style::AbsoluteSizeSpan::getSpanTypeId()
{
	return call_method<
		android::text::style::AbsoluteSizeSpan::J2CPP_CLASS_NAME,
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_NAME(3),
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_SIGNATURE(3), 
		jint
	>(get_jobject());
}

jint android::text::style::AbsoluteSizeSpan::describeContents()
{
	return call_method<
		android::text::style::AbsoluteSizeSpan::J2CPP_CLASS_NAME,
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_NAME(4),
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_SIGNATURE(4), 
		jint
	>(get_jobject());
}

void android::text::style::AbsoluteSizeSpan::writeToParcel(local_ref< android::os::Parcel > const &a0, jint a1)
{
	return call_method<
		android::text::style::AbsoluteSizeSpan::J2CPP_CLASS_NAME,
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_NAME(5),
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_SIGNATURE(5), 
		void
	>(get_jobject(), a0, a1);
}

jint android::text::style::AbsoluteSizeSpan::getSize()
{
	return call_method<
		android::text::style::AbsoluteSizeSpan::J2CPP_CLASS_NAME,
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_NAME(6),
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_SIGNATURE(6), 
		jint
	>(get_jobject());
}

jboolean android::text::style::AbsoluteSizeSpan::getDip()
{
	return call_method<
		android::text::style::AbsoluteSizeSpan::J2CPP_CLASS_NAME,
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_NAME(7),
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_SIGNATURE(7), 
		jboolean
	>(get_jobject());
}

void android::text::style::AbsoluteSizeSpan::updateDrawState(local_ref< android::text::TextPaint > const &a0)
{
	return call_method<
		android::text::style::AbsoluteSizeSpan::J2CPP_CLASS_NAME,
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_NAME(8),
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_SIGNATURE(8), 
		void
	>(get_jobject(), a0);
}

void android::text::style::AbsoluteSizeSpan::updateMeasureState(local_ref< android::text::TextPaint > const &a0)
{
	return call_method<
		android::text::style::AbsoluteSizeSpan::J2CPP_CLASS_NAME,
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_NAME(9),
		android::text::style::AbsoluteSizeSpan::J2CPP_METHOD_SIGNATURE(9), 
		void
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(android::text::style::AbsoluteSizeSpan,"android/text/style/AbsoluteSizeSpan")
J2CPP_DEFINE_METHOD(android::text::style::AbsoluteSizeSpan,0,"<init>","(I)V")
J2CPP_DEFINE_METHOD(android::text::style::AbsoluteSizeSpan,1,"<init>","(IZ)V")
J2CPP_DEFINE_METHOD(android::text::style::AbsoluteSizeSpan,2,"<init>","(Landroid/os/Parcel;)V")
J2CPP_DEFINE_METHOD(android::text::style::AbsoluteSizeSpan,3,"getSpanTypeId","()I")
J2CPP_DEFINE_METHOD(android::text::style::AbsoluteSizeSpan,4,"describeContents","()I")
J2CPP_DEFINE_METHOD(android::text::style::AbsoluteSizeSpan,5,"writeToParcel","(Landroid/os/Parcel;I)V")
J2CPP_DEFINE_METHOD(android::text::style::AbsoluteSizeSpan,6,"getSize","()I")
J2CPP_DEFINE_METHOD(android::text::style::AbsoluteSizeSpan,7,"getDip","()Z")
J2CPP_DEFINE_METHOD(android::text::style::AbsoluteSizeSpan,8,"updateDrawState","(Landroid/text/TextPaint;)V")
J2CPP_DEFINE_METHOD(android::text::style::AbsoluteSizeSpan,9,"updateMeasureState","(Landroid/text/TextPaint;)V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_TEXT_STYLE_ABSOLUTESIZESPAN_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
