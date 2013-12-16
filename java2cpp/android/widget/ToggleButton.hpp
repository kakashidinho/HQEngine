/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.widget.ToggleButton
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_WIDGET_TOGGLEBUTTON_HPP_DECL
#define J2CPP_ANDROID_WIDGET_TOGGLEBUTTON_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class CharSequence; } } }
namespace j2cpp { namespace android { namespace graphics { namespace drawable { class Drawable; } } } }
namespace j2cpp { namespace android { namespace content { class Context; } } }
namespace j2cpp { namespace android { namespace widget { class CompoundButton; } } }
namespace j2cpp { namespace android { namespace util { class AttributeSet; } } }


#include <android/content/Context.hpp>
#include <android/graphics/drawable/Drawable.hpp>
#include <android/util/AttributeSet.hpp>
#include <android/widget/CompoundButton.hpp>
#include <java/lang/CharSequence.hpp>


namespace j2cpp {

namespace android { namespace widget {

	class ToggleButton;
	class ToggleButton
		: public object<ToggleButton>
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
		J2CPP_DECLARE_METHOD(10)

		explicit ToggleButton(jobject jobj)
		: object<ToggleButton>(jobj)
		{
		}

		operator local_ref<android::widget::CompoundButton>() const;


		ToggleButton(local_ref< android::content::Context > const&, local_ref< android::util::AttributeSet > const&, jint);
		ToggleButton(local_ref< android::content::Context > const&, local_ref< android::util::AttributeSet > const&);
		ToggleButton(local_ref< android::content::Context > const&);
		void setChecked(jboolean);
		local_ref< java::lang::CharSequence > getTextOn();
		void setTextOn(local_ref< java::lang::CharSequence >  const&);
		local_ref< java::lang::CharSequence > getTextOff();
		void setTextOff(local_ref< java::lang::CharSequence >  const&);
		void setBackgroundDrawable(local_ref< android::graphics::drawable::Drawable >  const&);
	}; //class ToggleButton

} //namespace widget
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_WIDGET_TOGGLEBUTTON_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_WIDGET_TOGGLEBUTTON_HPP_IMPL
#define J2CPP_ANDROID_WIDGET_TOGGLEBUTTON_HPP_IMPL

namespace j2cpp {



android::widget::ToggleButton::operator local_ref<android::widget::CompoundButton>() const
{
	return local_ref<android::widget::CompoundButton>(get_jobject());
}


android::widget::ToggleButton::ToggleButton(local_ref< android::content::Context > const &a0, local_ref< android::util::AttributeSet > const &a1, jint a2)
: object<android::widget::ToggleButton>(
	call_new_object<
		android::widget::ToggleButton::J2CPP_CLASS_NAME,
		android::widget::ToggleButton::J2CPP_METHOD_NAME(0),
		android::widget::ToggleButton::J2CPP_METHOD_SIGNATURE(0)
	>(a0, a1, a2)
)
{
}



android::widget::ToggleButton::ToggleButton(local_ref< android::content::Context > const &a0, local_ref< android::util::AttributeSet > const &a1)
: object<android::widget::ToggleButton>(
	call_new_object<
		android::widget::ToggleButton::J2CPP_CLASS_NAME,
		android::widget::ToggleButton::J2CPP_METHOD_NAME(1),
		android::widget::ToggleButton::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}



android::widget::ToggleButton::ToggleButton(local_ref< android::content::Context > const &a0)
: object<android::widget::ToggleButton>(
	call_new_object<
		android::widget::ToggleButton::J2CPP_CLASS_NAME,
		android::widget::ToggleButton::J2CPP_METHOD_NAME(2),
		android::widget::ToggleButton::J2CPP_METHOD_SIGNATURE(2)
	>(a0)
)
{
}


void android::widget::ToggleButton::setChecked(jboolean a0)
{
	return call_method<
		android::widget::ToggleButton::J2CPP_CLASS_NAME,
		android::widget::ToggleButton::J2CPP_METHOD_NAME(3),
		android::widget::ToggleButton::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0);
}

local_ref< java::lang::CharSequence > android::widget::ToggleButton::getTextOn()
{
	return call_method<
		android::widget::ToggleButton::J2CPP_CLASS_NAME,
		android::widget::ToggleButton::J2CPP_METHOD_NAME(4),
		android::widget::ToggleButton::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< java::lang::CharSequence >
	>(get_jobject());
}

void android::widget::ToggleButton::setTextOn(local_ref< java::lang::CharSequence > const &a0)
{
	return call_method<
		android::widget::ToggleButton::J2CPP_CLASS_NAME,
		android::widget::ToggleButton::J2CPP_METHOD_NAME(5),
		android::widget::ToggleButton::J2CPP_METHOD_SIGNATURE(5), 
		void
	>(get_jobject(), a0);
}

local_ref< java::lang::CharSequence > android::widget::ToggleButton::getTextOff()
{
	return call_method<
		android::widget::ToggleButton::J2CPP_CLASS_NAME,
		android::widget::ToggleButton::J2CPP_METHOD_NAME(6),
		android::widget::ToggleButton::J2CPP_METHOD_SIGNATURE(6), 
		local_ref< java::lang::CharSequence >
	>(get_jobject());
}

void android::widget::ToggleButton::setTextOff(local_ref< java::lang::CharSequence > const &a0)
{
	return call_method<
		android::widget::ToggleButton::J2CPP_CLASS_NAME,
		android::widget::ToggleButton::J2CPP_METHOD_NAME(7),
		android::widget::ToggleButton::J2CPP_METHOD_SIGNATURE(7), 
		void
	>(get_jobject(), a0);
}


void android::widget::ToggleButton::setBackgroundDrawable(local_ref< android::graphics::drawable::Drawable > const &a0)
{
	return call_method<
		android::widget::ToggleButton::J2CPP_CLASS_NAME,
		android::widget::ToggleButton::J2CPP_METHOD_NAME(9),
		android::widget::ToggleButton::J2CPP_METHOD_SIGNATURE(9), 
		void
	>(get_jobject(), a0);
}



J2CPP_DEFINE_CLASS(android::widget::ToggleButton,"android/widget/ToggleButton")
J2CPP_DEFINE_METHOD(android::widget::ToggleButton,0,"<init>","(Landroid/content/Context;Landroid/util/AttributeSet;I)V")
J2CPP_DEFINE_METHOD(android::widget::ToggleButton,1,"<init>","(Landroid/content/Context;Landroid/util/AttributeSet;)V")
J2CPP_DEFINE_METHOD(android::widget::ToggleButton,2,"<init>","(Landroid/content/Context;)V")
J2CPP_DEFINE_METHOD(android::widget::ToggleButton,3,"setChecked","(Z)V")
J2CPP_DEFINE_METHOD(android::widget::ToggleButton,4,"getTextOn","()Ljava/lang/CharSequence;")
J2CPP_DEFINE_METHOD(android::widget::ToggleButton,5,"setTextOn","(Ljava/lang/CharSequence;)V")
J2CPP_DEFINE_METHOD(android::widget::ToggleButton,6,"getTextOff","()Ljava/lang/CharSequence;")
J2CPP_DEFINE_METHOD(android::widget::ToggleButton,7,"setTextOff","(Ljava/lang/CharSequence;)V")
J2CPP_DEFINE_METHOD(android::widget::ToggleButton,8,"onFinishInflate","()V")
J2CPP_DEFINE_METHOD(android::widget::ToggleButton,9,"setBackgroundDrawable","(Landroid/graphics/drawable/Drawable;)V")
J2CPP_DEFINE_METHOD(android::widget::ToggleButton,10,"drawableStateChanged","()V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_WIDGET_TOGGLEBUTTON_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
