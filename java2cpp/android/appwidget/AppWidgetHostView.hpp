/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.appwidget.AppWidgetHostView
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_APPWIDGET_APPWIDGETHOSTVIEW_HPP_DECL
#define J2CPP_ANDROID_APPWIDGET_APPWIDGETHOSTVIEW_HPP_DECL


namespace j2cpp { namespace android { namespace appwidget { class AppWidgetProviderInfo; } } }
namespace j2cpp { namespace android { namespace content { class Context; } } }
namespace j2cpp { namespace android { namespace view { namespace ViewGroup_ { class LayoutParams; } } } }
namespace j2cpp { namespace android { namespace widget { class RemoteViews; } } }
namespace j2cpp { namespace android { namespace widget { class FrameLayout; } } }
namespace j2cpp { namespace android { namespace widget { namespace FrameLayout_ { class LayoutParams; } } } }
namespace j2cpp { namespace android { namespace util { class AttributeSet; } } }


#include <android/appwidget/AppWidgetProviderInfo.hpp>
#include <android/content/Context.hpp>
#include <android/util/AttributeSet.hpp>
#include <android/view/ViewGroup.hpp>
#include <android/widget/FrameLayout.hpp>
#include <android/widget/RemoteViews.hpp>


namespace j2cpp {

namespace android { namespace appwidget {

	class AppWidgetHostView;
	class AppWidgetHostView
		: public object<AppWidgetHostView>
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

		explicit AppWidgetHostView(jobject jobj)
		: object<AppWidgetHostView>(jobj)
		{
		}

		operator local_ref<android::widget::FrameLayout>() const;


		AppWidgetHostView(local_ref< android::content::Context > const&);
		AppWidgetHostView(local_ref< android::content::Context > const&, jint, jint);
		void setAppWidget(jint, local_ref< android::appwidget::AppWidgetProviderInfo >  const&);
		jint getAppWidgetId();
		local_ref< android::appwidget::AppWidgetProviderInfo > getAppWidgetInfo();
		local_ref< android::widget::FrameLayout_::LayoutParams > generateLayoutParams(local_ref< android::util::AttributeSet >  const&);
		void updateAppWidget(local_ref< android::widget::RemoteViews >  const&);
		local_ref< android::view::ViewGroup_::LayoutParams > generateLayoutParams_1(local_ref< android::util::AttributeSet >  const&);
	}; //class AppWidgetHostView

} //namespace appwidget
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_APPWIDGET_APPWIDGETHOSTVIEW_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_APPWIDGET_APPWIDGETHOSTVIEW_HPP_IMPL
#define J2CPP_ANDROID_APPWIDGET_APPWIDGETHOSTVIEW_HPP_IMPL

namespace j2cpp {



android::appwidget::AppWidgetHostView::operator local_ref<android::widget::FrameLayout>() const
{
	return local_ref<android::widget::FrameLayout>(get_jobject());
}


android::appwidget::AppWidgetHostView::AppWidgetHostView(local_ref< android::content::Context > const &a0)
: object<android::appwidget::AppWidgetHostView>(
	call_new_object<
		android::appwidget::AppWidgetHostView::J2CPP_CLASS_NAME,
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_NAME(0),
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



android::appwidget::AppWidgetHostView::AppWidgetHostView(local_ref< android::content::Context > const &a0, jint a1, jint a2)
: object<android::appwidget::AppWidgetHostView>(
	call_new_object<
		android::appwidget::AppWidgetHostView::J2CPP_CLASS_NAME,
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_NAME(1),
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1, a2)
)
{
}


void android::appwidget::AppWidgetHostView::setAppWidget(jint a0, local_ref< android::appwidget::AppWidgetProviderInfo > const &a1)
{
	return call_method<
		android::appwidget::AppWidgetHostView::J2CPP_CLASS_NAME,
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_NAME(2),
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject(), a0, a1);
}

jint android::appwidget::AppWidgetHostView::getAppWidgetId()
{
	return call_method<
		android::appwidget::AppWidgetHostView::J2CPP_CLASS_NAME,
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_NAME(3),
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_SIGNATURE(3), 
		jint
	>(get_jobject());
}

local_ref< android::appwidget::AppWidgetProviderInfo > android::appwidget::AppWidgetHostView::getAppWidgetInfo()
{
	return call_method<
		android::appwidget::AppWidgetHostView::J2CPP_CLASS_NAME,
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_NAME(4),
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< android::appwidget::AppWidgetProviderInfo >
	>(get_jobject());
}



local_ref< android::widget::FrameLayout_::LayoutParams > android::appwidget::AppWidgetHostView::generateLayoutParams(local_ref< android::util::AttributeSet > const &a0)
{
	return call_method<
		android::appwidget::AppWidgetHostView::J2CPP_CLASS_NAME,
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_NAME(7),
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< android::widget::FrameLayout_::LayoutParams >
	>(get_jobject(), a0);
}

void android::appwidget::AppWidgetHostView::updateAppWidget(local_ref< android::widget::RemoteViews > const &a0)
{
	return call_method<
		android::appwidget::AppWidgetHostView::J2CPP_CLASS_NAME,
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_NAME(8),
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_SIGNATURE(8), 
		void
	>(get_jobject(), a0);
}





local_ref< android::view::ViewGroup_::LayoutParams > android::appwidget::AppWidgetHostView::generateLayoutParams_1(local_ref< android::util::AttributeSet > const &a0)
{
	return call_method<
		android::appwidget::AppWidgetHostView::J2CPP_CLASS_NAME,
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_NAME(13),
		android::appwidget::AppWidgetHostView::J2CPP_METHOD_SIGNATURE(13), 
		local_ref< android::view::ViewGroup_::LayoutParams >
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(android::appwidget::AppWidgetHostView,"android/appwidget/AppWidgetHostView")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,0,"<init>","(Landroid/content/Context;)V")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,1,"<init>","(Landroid/content/Context;II)V")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,2,"setAppWidget","(ILandroid/appwidget/AppWidgetProviderInfo;)V")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,3,"getAppWidgetId","()I")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,4,"getAppWidgetInfo","()Landroid/appwidget/AppWidgetProviderInfo;")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,5,"dispatchSaveInstanceState","(Landroid/util/SparseArray;)V")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,6,"dispatchRestoreInstanceState","(Landroid/util/SparseArray;)V")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,7,"generateLayoutParams","(Landroid/util/AttributeSet;)Landroid/widget/FrameLayout$LayoutParams;")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,8,"updateAppWidget","(Landroid/widget/RemoteViews;)V")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,9,"drawChild","(Landroid/graphics/Canvas;Landroid/view/View;J)Z")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,10,"prepareView","(Landroid/view/View;)V")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,11,"getDefaultView","()Landroid/view/View;")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,12,"getErrorView","()Landroid/view/View;")
J2CPP_DEFINE_METHOD(android::appwidget::AppWidgetHostView,13,"generateLayoutParams","(Landroid/util/AttributeSet;)Landroid/view/ViewGroup$LayoutParams;")

} //namespace j2cpp

#endif //J2CPP_ANDROID_APPWIDGET_APPWIDGETHOSTVIEW_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
