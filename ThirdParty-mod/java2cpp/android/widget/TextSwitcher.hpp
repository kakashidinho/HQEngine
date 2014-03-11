/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.widget.TextSwitcher
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_WIDGET_TEXTSWITCHER_HPP_DECL
#define J2CPP_ANDROID_WIDGET_TEXTSWITCHER_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class CharSequence; } } }
namespace j2cpp { namespace android { namespace content { class Context; } } }
namespace j2cpp { namespace android { namespace view { class View; } } }
namespace j2cpp { namespace android { namespace view { namespace ViewGroup_ { class LayoutParams; } } } }
namespace j2cpp { namespace android { namespace widget { class ViewSwitcher; } } }
namespace j2cpp { namespace android { namespace util { class AttributeSet; } } }


#include <android/content/Context.hpp>
#include <android/util/AttributeSet.hpp>
#include <android/view/View.hpp>
#include <android/view/ViewGroup.hpp>
#include <android/widget/ViewSwitcher.hpp>
#include <java/lang/CharSequence.hpp>


namespace j2cpp {

namespace android { namespace widget {

	class TextSwitcher;
	class TextSwitcher
		: public object<TextSwitcher>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)

		explicit TextSwitcher(jobject jobj)
		: object<TextSwitcher>(jobj)
		{
		}

		operator local_ref<android::widget::ViewSwitcher>() const;


		TextSwitcher(local_ref< android::content::Context > const&);
		TextSwitcher(local_ref< android::content::Context > const&, local_ref< android::util::AttributeSet > const&);
		void addView(local_ref< android::view::View >  const&, jint, local_ref< android::view::ViewGroup_::LayoutParams >  const&);
		void setText(local_ref< java::lang::CharSequence >  const&);
		void setCurrentText(local_ref< java::lang::CharSequence >  const&);
	}; //class TextSwitcher

} //namespace widget
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_WIDGET_TEXTSWITCHER_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_WIDGET_TEXTSWITCHER_HPP_IMPL
#define J2CPP_ANDROID_WIDGET_TEXTSWITCHER_HPP_IMPL

namespace j2cpp {



android::widget::TextSwitcher::operator local_ref<android::widget::ViewSwitcher>() const
{
	return local_ref<android::widget::ViewSwitcher>(get_jobject());
}


android::widget::TextSwitcher::TextSwitcher(local_ref< android::content::Context > const &a0)
: object<android::widget::TextSwitcher>(
	call_new_object<
		android::widget::TextSwitcher::J2CPP_CLASS_NAME,
		android::widget::TextSwitcher::J2CPP_METHOD_NAME(0),
		android::widget::TextSwitcher::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



android::widget::TextSwitcher::TextSwitcher(local_ref< android::content::Context > const &a0, local_ref< android::util::AttributeSet > const &a1)
: object<android::widget::TextSwitcher>(
	call_new_object<
		android::widget::TextSwitcher::J2CPP_CLASS_NAME,
		android::widget::TextSwitcher::J2CPP_METHOD_NAME(1),
		android::widget::TextSwitcher::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}


void android::widget::TextSwitcher::addView(local_ref< android::view::View > const &a0, jint a1, local_ref< android::view::ViewGroup_::LayoutParams > const &a2)
{
	return call_method<
		android::widget::TextSwitcher::J2CPP_CLASS_NAME,
		android::widget::TextSwitcher::J2CPP_METHOD_NAME(2),
		android::widget::TextSwitcher::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject(), a0, a1, a2);
}

void android::widget::TextSwitcher::setText(local_ref< java::lang::CharSequence > const &a0)
{
	return call_method<
		android::widget::TextSwitcher::J2CPP_CLASS_NAME,
		android::widget::TextSwitcher::J2CPP_METHOD_NAME(3),
		android::widget::TextSwitcher::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0);
}

void android::widget::TextSwitcher::setCurrentText(local_ref< java::lang::CharSequence > const &a0)
{
	return call_method<
		android::widget::TextSwitcher::J2CPP_CLASS_NAME,
		android::widget::TextSwitcher::J2CPP_METHOD_NAME(4),
		android::widget::TextSwitcher::J2CPP_METHOD_SIGNATURE(4), 
		void
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(android::widget::TextSwitcher,"android/widget/TextSwitcher")
J2CPP_DEFINE_METHOD(android::widget::TextSwitcher,0,"<init>","(Landroid/content/Context;)V")
J2CPP_DEFINE_METHOD(android::widget::TextSwitcher,1,"<init>","(Landroid/content/Context;Landroid/util/AttributeSet;)V")
J2CPP_DEFINE_METHOD(android::widget::TextSwitcher,2,"addView","(Landroid/view/View;ILandroid/view/ViewGroup$LayoutParams;)V")
J2CPP_DEFINE_METHOD(android::widget::TextSwitcher,3,"setText","(Ljava/lang/CharSequence;)V")
J2CPP_DEFINE_METHOD(android::widget::TextSwitcher,4,"setCurrentText","(Ljava/lang/CharSequence;)V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_WIDGET_TEXTSWITCHER_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION