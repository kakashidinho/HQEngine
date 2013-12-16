/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.view.ViewStub
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_VIEW_VIEWSTUB_HPP_DECL
#define J2CPP_ANDROID_VIEW_VIEWSTUB_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace android { namespace graphics { class Canvas; } } }
namespace j2cpp { namespace android { namespace content { class Context; } } }
namespace j2cpp { namespace android { namespace view { class View; } } }
namespace j2cpp { namespace android { namespace view { namespace ViewStub_ { class OnInflateListener; } } } }
namespace j2cpp { namespace android { namespace util { class AttributeSet; } } }


#include <android/content/Context.hpp>
#include <android/graphics/Canvas.hpp>
#include <android/util/AttributeSet.hpp>
#include <android/view/View.hpp>
#include <android/view/ViewStub.hpp>
#include <java/lang/Object.hpp>


namespace j2cpp {

namespace android { namespace view {

	class ViewStub;
	namespace ViewStub_ {

		class OnInflateListener;
		class OnInflateListener
			: public object<OnInflateListener>
		{
		public:

			J2CPP_DECLARE_CLASS

			J2CPP_DECLARE_METHOD(0)

			explicit OnInflateListener(jobject jobj)
			: object<OnInflateListener>(jobj)
			{
			}

			operator local_ref<java::lang::Object>() const;


			void onInflate(local_ref< android::view::ViewStub >  const&, local_ref< android::view::View >  const&);
		}; //class OnInflateListener

	} //namespace ViewStub_

	class ViewStub
		: public object<ViewStub>
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
		J2CPP_DECLARE_METHOD(11)
		J2CPP_DECLARE_METHOD(12)
		J2CPP_DECLARE_METHOD(13)

		typedef ViewStub_::OnInflateListener OnInflateListener;

		explicit ViewStub(jobject jobj)
		: object<ViewStub>(jobj)
		{
		}

		operator local_ref<android::view::View>() const;


		ViewStub(local_ref< android::content::Context > const&);
		ViewStub(local_ref< android::content::Context > const&, jint);
		ViewStub(local_ref< android::content::Context > const&, local_ref< android::util::AttributeSet > const&);
		ViewStub(local_ref< android::content::Context > const&, local_ref< android::util::AttributeSet > const&, jint);
		jint getInflatedId();
		void setInflatedId(jint);
		jint getLayoutResource();
		void setLayoutResource(jint);
		void draw(local_ref< android::graphics::Canvas >  const&);
		void setVisibility(jint);
		local_ref< android::view::View > inflate();
		void setOnInflateListener(local_ref< android::view::ViewStub_::OnInflateListener >  const&);
	}; //class ViewStub

} //namespace view
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_VIEW_VIEWSTUB_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_VIEW_VIEWSTUB_HPP_IMPL
#define J2CPP_ANDROID_VIEW_VIEWSTUB_HPP_IMPL

namespace j2cpp {




android::view::ViewStub_::OnInflateListener::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

void android::view::ViewStub_::OnInflateListener::onInflate(local_ref< android::view::ViewStub > const &a0, local_ref< android::view::View > const &a1)
{
	return call_method<
		android::view::ViewStub_::OnInflateListener::J2CPP_CLASS_NAME,
		android::view::ViewStub_::OnInflateListener::J2CPP_METHOD_NAME(0),
		android::view::ViewStub_::OnInflateListener::J2CPP_METHOD_SIGNATURE(0), 
		void
	>(get_jobject(), a0, a1);
}


J2CPP_DEFINE_CLASS(android::view::ViewStub_::OnInflateListener,"android/view/ViewStub$OnInflateListener")
J2CPP_DEFINE_METHOD(android::view::ViewStub_::OnInflateListener,0,"onInflate","(Landroid/view/ViewStub;Landroid/view/View;)V")



android::view::ViewStub::operator local_ref<android::view::View>() const
{
	return local_ref<android::view::View>(get_jobject());
}


android::view::ViewStub::ViewStub(local_ref< android::content::Context > const &a0)
: object<android::view::ViewStub>(
	call_new_object<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(0),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



android::view::ViewStub::ViewStub(local_ref< android::content::Context > const &a0, jint a1)
: object<android::view::ViewStub>(
	call_new_object<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(1),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}



android::view::ViewStub::ViewStub(local_ref< android::content::Context > const &a0, local_ref< android::util::AttributeSet > const &a1)
: object<android::view::ViewStub>(
	call_new_object<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(2),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(2)
	>(a0, a1)
)
{
}



android::view::ViewStub::ViewStub(local_ref< android::content::Context > const &a0, local_ref< android::util::AttributeSet > const &a1, jint a2)
: object<android::view::ViewStub>(
	call_new_object<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(3),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(3)
	>(a0, a1, a2)
)
{
}


jint android::view::ViewStub::getInflatedId()
{
	return call_method<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(4),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(4), 
		jint
	>(get_jobject());
}

void android::view::ViewStub::setInflatedId(jint a0)
{
	return call_method<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(5),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(5), 
		void
	>(get_jobject(), a0);
}

jint android::view::ViewStub::getLayoutResource()
{
	return call_method<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(6),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(6), 
		jint
	>(get_jobject());
}

void android::view::ViewStub::setLayoutResource(jint a0)
{
	return call_method<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(7),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(7), 
		void
	>(get_jobject(), a0);
}


void android::view::ViewStub::draw(local_ref< android::graphics::Canvas > const &a0)
{
	return call_method<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(9),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(9), 
		void
	>(get_jobject(), a0);
}


void android::view::ViewStub::setVisibility(jint a0)
{
	return call_method<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(11),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(11), 
		void
	>(get_jobject(), a0);
}

local_ref< android::view::View > android::view::ViewStub::inflate()
{
	return call_method<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(12),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(12), 
		local_ref< android::view::View >
	>(get_jobject());
}

void android::view::ViewStub::setOnInflateListener(local_ref< android::view::ViewStub_::OnInflateListener > const &a0)
{
	return call_method<
		android::view::ViewStub::J2CPP_CLASS_NAME,
		android::view::ViewStub::J2CPP_METHOD_NAME(13),
		android::view::ViewStub::J2CPP_METHOD_SIGNATURE(13), 
		void
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(android::view::ViewStub,"android/view/ViewStub")
J2CPP_DEFINE_METHOD(android::view::ViewStub,0,"<init>","(Landroid/content/Context;)V")
J2CPP_DEFINE_METHOD(android::view::ViewStub,1,"<init>","(Landroid/content/Context;I)V")
J2CPP_DEFINE_METHOD(android::view::ViewStub,2,"<init>","(Landroid/content/Context;Landroid/util/AttributeSet;)V")
J2CPP_DEFINE_METHOD(android::view::ViewStub,3,"<init>","(Landroid/content/Context;Landroid/util/AttributeSet;I)V")
J2CPP_DEFINE_METHOD(android::view::ViewStub,4,"getInflatedId","()I")
J2CPP_DEFINE_METHOD(android::view::ViewStub,5,"setInflatedId","(I)V")
J2CPP_DEFINE_METHOD(android::view::ViewStub,6,"getLayoutResource","()I")
J2CPP_DEFINE_METHOD(android::view::ViewStub,7,"setLayoutResource","(I)V")
J2CPP_DEFINE_METHOD(android::view::ViewStub,8,"onMeasure","(II)V")
J2CPP_DEFINE_METHOD(android::view::ViewStub,9,"draw","(Landroid/graphics/Canvas;)V")
J2CPP_DEFINE_METHOD(android::view::ViewStub,10,"dispatchDraw","(Landroid/graphics/Canvas;)V")
J2CPP_DEFINE_METHOD(android::view::ViewStub,11,"setVisibility","(I)V")
J2CPP_DEFINE_METHOD(android::view::ViewStub,12,"inflate","()Landroid/view/View;")
J2CPP_DEFINE_METHOD(android::view::ViewStub,13,"setOnInflateListener","(Landroid/view/ViewStub$OnInflateListener;)V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_VIEW_VIEWSTUB_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
