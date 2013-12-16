/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.app.ExpandableListActivity
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_APP_EXPANDABLELISTACTIVITY_HPP_DECL
#define J2CPP_ANDROID_APP_EXPANDABLELISTACTIVITY_HPP_DECL


namespace j2cpp { namespace android { namespace app { class Activity; } } }
namespace j2cpp { namespace android { namespace view { class View; } } }
namespace j2cpp { namespace android { namespace view { class ContextMenu; } } }
namespace j2cpp { namespace android { namespace view { namespace View_ { class OnCreateContextMenuListener; } } } }
namespace j2cpp { namespace android { namespace view { namespace ContextMenu_ { class ContextMenuInfo; } } } }
namespace j2cpp { namespace android { namespace widget { class ExpandableListView; } } }
namespace j2cpp { namespace android { namespace widget { namespace ExpandableListView_ { class OnGroupCollapseListener; } } } }
namespace j2cpp { namespace android { namespace widget { namespace ExpandableListView_ { class OnChildClickListener; } } } }
namespace j2cpp { namespace android { namespace widget { namespace ExpandableListView_ { class OnGroupExpandListener; } } } }
namespace j2cpp { namespace android { namespace widget { class ExpandableListAdapter; } } }


#include <android/app/Activity.hpp>
#include <android/view/ContextMenu.hpp>
#include <android/view/View.hpp>
#include <android/widget/ExpandableListAdapter.hpp>
#include <android/widget/ExpandableListView.hpp>


namespace j2cpp {

namespace android { namespace app {

	class ExpandableListActivity;
	class ExpandableListActivity
		: public object<ExpandableListActivity>
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

		explicit ExpandableListActivity(jobject jobj)
		: object<ExpandableListActivity>(jobj)
		{
		}

		operator local_ref<android::app::Activity>() const;
		operator local_ref<android::view::View_::OnCreateContextMenuListener>() const;
		operator local_ref<android::widget::ExpandableListView_::OnChildClickListener>() const;
		operator local_ref<android::widget::ExpandableListView_::OnGroupCollapseListener>() const;
		operator local_ref<android::widget::ExpandableListView_::OnGroupExpandListener>() const;


		ExpandableListActivity();
		void onCreateContextMenu(local_ref< android::view::ContextMenu >  const&, local_ref< android::view::View >  const&, local_ref< android::view::ContextMenu_::ContextMenuInfo >  const&);
		jboolean onChildClick(local_ref< android::widget::ExpandableListView >  const&, local_ref< android::view::View >  const&, jint, jint, jlong);
		void onGroupCollapse(jint);
		void onGroupExpand(jint);
		void onContentChanged();
		void setListAdapter(local_ref< android::widget::ExpandableListAdapter >  const&);
		local_ref< android::widget::ExpandableListView > getExpandableListView();
		local_ref< android::widget::ExpandableListAdapter > getExpandableListAdapter();
		jlong getSelectedId();
		jlong getSelectedPosition();
		jboolean setSelectedChild(jint, jint, jboolean);
		void setSelectedGroup(jint);
	}; //class ExpandableListActivity

} //namespace app
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_APP_EXPANDABLELISTACTIVITY_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_APP_EXPANDABLELISTACTIVITY_HPP_IMPL
#define J2CPP_ANDROID_APP_EXPANDABLELISTACTIVITY_HPP_IMPL

namespace j2cpp {



android::app::ExpandableListActivity::operator local_ref<android::app::Activity>() const
{
	return local_ref<android::app::Activity>(get_jobject());
}

android::app::ExpandableListActivity::operator local_ref<android::view::View_::OnCreateContextMenuListener>() const
{
	return local_ref<android::view::View_::OnCreateContextMenuListener>(get_jobject());
}

android::app::ExpandableListActivity::operator local_ref<android::widget::ExpandableListView_::OnChildClickListener>() const
{
	return local_ref<android::widget::ExpandableListView_::OnChildClickListener>(get_jobject());
}

android::app::ExpandableListActivity::operator local_ref<android::widget::ExpandableListView_::OnGroupCollapseListener>() const
{
	return local_ref<android::widget::ExpandableListView_::OnGroupCollapseListener>(get_jobject());
}

android::app::ExpandableListActivity::operator local_ref<android::widget::ExpandableListView_::OnGroupExpandListener>() const
{
	return local_ref<android::widget::ExpandableListView_::OnGroupExpandListener>(get_jobject());
}


android::app::ExpandableListActivity::ExpandableListActivity()
: object<android::app::ExpandableListActivity>(
	call_new_object<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(0),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}


void android::app::ExpandableListActivity::onCreateContextMenu(local_ref< android::view::ContextMenu > const &a0, local_ref< android::view::View > const &a1, local_ref< android::view::ContextMenu_::ContextMenuInfo > const &a2)
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(1),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject(), a0, a1, a2);
}

jboolean android::app::ExpandableListActivity::onChildClick(local_ref< android::widget::ExpandableListView > const &a0, local_ref< android::view::View > const &a1, jint a2, jint a3, jlong a4)
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(2),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(2), 
		jboolean
	>(get_jobject(), a0, a1, a2, a3, a4);
}

void android::app::ExpandableListActivity::onGroupCollapse(jint a0)
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(3),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0);
}

void android::app::ExpandableListActivity::onGroupExpand(jint a0)
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(4),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(4), 
		void
	>(get_jobject(), a0);
}


void android::app::ExpandableListActivity::onContentChanged()
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(6),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(6), 
		void
	>(get_jobject());
}

void android::app::ExpandableListActivity::setListAdapter(local_ref< android::widget::ExpandableListAdapter > const &a0)
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(7),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(7), 
		void
	>(get_jobject(), a0);
}

local_ref< android::widget::ExpandableListView > android::app::ExpandableListActivity::getExpandableListView()
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(8),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(8), 
		local_ref< android::widget::ExpandableListView >
	>(get_jobject());
}

local_ref< android::widget::ExpandableListAdapter > android::app::ExpandableListActivity::getExpandableListAdapter()
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(9),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(9), 
		local_ref< android::widget::ExpandableListAdapter >
	>(get_jobject());
}

jlong android::app::ExpandableListActivity::getSelectedId()
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(10),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(10), 
		jlong
	>(get_jobject());
}

jlong android::app::ExpandableListActivity::getSelectedPosition()
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(11),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(11), 
		jlong
	>(get_jobject());
}

jboolean android::app::ExpandableListActivity::setSelectedChild(jint a0, jint a1, jboolean a2)
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(12),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(12), 
		jboolean
	>(get_jobject(), a0, a1, a2);
}

void android::app::ExpandableListActivity::setSelectedGroup(jint a0)
{
	return call_method<
		android::app::ExpandableListActivity::J2CPP_CLASS_NAME,
		android::app::ExpandableListActivity::J2CPP_METHOD_NAME(13),
		android::app::ExpandableListActivity::J2CPP_METHOD_SIGNATURE(13), 
		void
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(android::app::ExpandableListActivity,"android/app/ExpandableListActivity")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,0,"<init>","()V")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,1,"onCreateContextMenu","(Landroid/view/ContextMenu;Landroid/view/View;Landroid/view/ContextMenu$ContextMenuInfo;)V")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,2,"onChildClick","(Landroid/widget/ExpandableListView;Landroid/view/View;IIJ)Z")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,3,"onGroupCollapse","(I)V")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,4,"onGroupExpand","(I)V")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,5,"onRestoreInstanceState","(Landroid/os/Bundle;)V")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,6,"onContentChanged","()V")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,7,"setListAdapter","(Landroid/widget/ExpandableListAdapter;)V")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,8,"getExpandableListView","()Landroid/widget/ExpandableListView;")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,9,"getExpandableListAdapter","()Landroid/widget/ExpandableListAdapter;")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,10,"getSelectedId","()J")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,11,"getSelectedPosition","()J")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,12,"setSelectedChild","(IIZ)Z")
J2CPP_DEFINE_METHOD(android::app::ExpandableListActivity,13,"setSelectedGroup","(I)V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_APP_EXPANDABLELISTACTIVITY_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
