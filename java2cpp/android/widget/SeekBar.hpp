/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.widget.SeekBar
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_WIDGET_SEEKBAR_HPP_DECL
#define J2CPP_ANDROID_WIDGET_SEEKBAR_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace android { namespace content { class Context; } } }
namespace j2cpp { namespace android { namespace widget { class AbsSeekBar; } } }
namespace j2cpp { namespace android { namespace widget { namespace SeekBar_ { class OnSeekBarChangeListener; } } } }
namespace j2cpp { namespace android { namespace util { class AttributeSet; } } }


#include <android/content/Context.hpp>
#include <android/util/AttributeSet.hpp>
#include <android/widget/AbsSeekBar.hpp>
#include <android/widget/SeekBar.hpp>
#include <java/lang/Object.hpp>


namespace j2cpp {

namespace android { namespace widget {

	class SeekBar;
	namespace SeekBar_ {

		class OnSeekBarChangeListener;
		class OnSeekBarChangeListener
			: public object<OnSeekBarChangeListener>
		{
		public:

			J2CPP_DECLARE_CLASS

			J2CPP_DECLARE_METHOD(0)
			J2CPP_DECLARE_METHOD(1)
			J2CPP_DECLARE_METHOD(2)

			explicit OnSeekBarChangeListener(jobject jobj)
			: object<OnSeekBarChangeListener>(jobj)
			{
			}

			operator local_ref<java::lang::Object>() const;


			void onProgressChanged(local_ref< android::widget::SeekBar >  const&, jint, jboolean);
			void onStartTrackingTouch(local_ref< android::widget::SeekBar >  const&);
			void onStopTrackingTouch(local_ref< android::widget::SeekBar >  const&);
		}; //class OnSeekBarChangeListener

	} //namespace SeekBar_

	class SeekBar
		: public object<SeekBar>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)

		typedef SeekBar_::OnSeekBarChangeListener OnSeekBarChangeListener;

		explicit SeekBar(jobject jobj)
		: object<SeekBar>(jobj)
		{
		}

		operator local_ref<android::widget::AbsSeekBar>() const;


		SeekBar(local_ref< android::content::Context > const&);
		SeekBar(local_ref< android::content::Context > const&, local_ref< android::util::AttributeSet > const&);
		SeekBar(local_ref< android::content::Context > const&, local_ref< android::util::AttributeSet > const&, jint);
		void setOnSeekBarChangeListener(local_ref< android::widget::SeekBar_::OnSeekBarChangeListener >  const&);
	}; //class SeekBar

} //namespace widget
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_WIDGET_SEEKBAR_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_WIDGET_SEEKBAR_HPP_IMPL
#define J2CPP_ANDROID_WIDGET_SEEKBAR_HPP_IMPL

namespace j2cpp {




android::widget::SeekBar_::OnSeekBarChangeListener::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

void android::widget::SeekBar_::OnSeekBarChangeListener::onProgressChanged(local_ref< android::widget::SeekBar > const &a0, jint a1, jboolean a2)
{
	return call_method<
		android::widget::SeekBar_::OnSeekBarChangeListener::J2CPP_CLASS_NAME,
		android::widget::SeekBar_::OnSeekBarChangeListener::J2CPP_METHOD_NAME(0),
		android::widget::SeekBar_::OnSeekBarChangeListener::J2CPP_METHOD_SIGNATURE(0), 
		void
	>(get_jobject(), a0, a1, a2);
}

void android::widget::SeekBar_::OnSeekBarChangeListener::onStartTrackingTouch(local_ref< android::widget::SeekBar > const &a0)
{
	return call_method<
		android::widget::SeekBar_::OnSeekBarChangeListener::J2CPP_CLASS_NAME,
		android::widget::SeekBar_::OnSeekBarChangeListener::J2CPP_METHOD_NAME(1),
		android::widget::SeekBar_::OnSeekBarChangeListener::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject(), a0);
}

void android::widget::SeekBar_::OnSeekBarChangeListener::onStopTrackingTouch(local_ref< android::widget::SeekBar > const &a0)
{
	return call_method<
		android::widget::SeekBar_::OnSeekBarChangeListener::J2CPP_CLASS_NAME,
		android::widget::SeekBar_::OnSeekBarChangeListener::J2CPP_METHOD_NAME(2),
		android::widget::SeekBar_::OnSeekBarChangeListener::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(android::widget::SeekBar_::OnSeekBarChangeListener,"android/widget/SeekBar$OnSeekBarChangeListener")
J2CPP_DEFINE_METHOD(android::widget::SeekBar_::OnSeekBarChangeListener,0,"onProgressChanged","(Landroid/widget/SeekBar;IZ)V")
J2CPP_DEFINE_METHOD(android::widget::SeekBar_::OnSeekBarChangeListener,1,"onStartTrackingTouch","(Landroid/widget/SeekBar;)V")
J2CPP_DEFINE_METHOD(android::widget::SeekBar_::OnSeekBarChangeListener,2,"onStopTrackingTouch","(Landroid/widget/SeekBar;)V")



android::widget::SeekBar::operator local_ref<android::widget::AbsSeekBar>() const
{
	return local_ref<android::widget::AbsSeekBar>(get_jobject());
}


android::widget::SeekBar::SeekBar(local_ref< android::content::Context > const &a0)
: object<android::widget::SeekBar>(
	call_new_object<
		android::widget::SeekBar::J2CPP_CLASS_NAME,
		android::widget::SeekBar::J2CPP_METHOD_NAME(0),
		android::widget::SeekBar::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



android::widget::SeekBar::SeekBar(local_ref< android::content::Context > const &a0, local_ref< android::util::AttributeSet > const &a1)
: object<android::widget::SeekBar>(
	call_new_object<
		android::widget::SeekBar::J2CPP_CLASS_NAME,
		android::widget::SeekBar::J2CPP_METHOD_NAME(1),
		android::widget::SeekBar::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}



android::widget::SeekBar::SeekBar(local_ref< android::content::Context > const &a0, local_ref< android::util::AttributeSet > const &a1, jint a2)
: object<android::widget::SeekBar>(
	call_new_object<
		android::widget::SeekBar::J2CPP_CLASS_NAME,
		android::widget::SeekBar::J2CPP_METHOD_NAME(2),
		android::widget::SeekBar::J2CPP_METHOD_SIGNATURE(2)
	>(a0, a1, a2)
)
{
}


void android::widget::SeekBar::setOnSeekBarChangeListener(local_ref< android::widget::SeekBar_::OnSeekBarChangeListener > const &a0)
{
	return call_method<
		android::widget::SeekBar::J2CPP_CLASS_NAME,
		android::widget::SeekBar::J2CPP_METHOD_NAME(3),
		android::widget::SeekBar::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(android::widget::SeekBar,"android/widget/SeekBar")
J2CPP_DEFINE_METHOD(android::widget::SeekBar,0,"<init>","(Landroid/content/Context;)V")
J2CPP_DEFINE_METHOD(android::widget::SeekBar,1,"<init>","(Landroid/content/Context;Landroid/util/AttributeSet;)V")
J2CPP_DEFINE_METHOD(android::widget::SeekBar,2,"<init>","(Landroid/content/Context;Landroid/util/AttributeSet;I)V")
J2CPP_DEFINE_METHOD(android::widget::SeekBar,3,"setOnSeekBarChangeListener","(Landroid/widget/SeekBar$OnSeekBarChangeListener;)V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_WIDGET_SEEKBAR_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
