/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.preference.EditTextPreference
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_PREFERENCE_EDITTEXTPREFERENCE_HPP_DECL
#define J2CPP_ANDROID_PREFERENCE_EDITTEXTPREFERENCE_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace android { namespace content { class Context; } } }
namespace j2cpp { namespace android { namespace preference { class DialogPreference; } } }
namespace j2cpp { namespace android { namespace widget { class EditText; } } }
namespace j2cpp { namespace android { namespace util { class AttributeSet; } } }


#include <android/content/Context.hpp>
#include <android/preference/DialogPreference.hpp>
#include <android/util/AttributeSet.hpp>
#include <android/widget/EditText.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace android { namespace preference {

	class EditTextPreference;
	class EditTextPreference
		: public object<EditTextPreference>
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

		explicit EditTextPreference(jobject jobj)
		: object<EditTextPreference>(jobj)
		{
		}

		operator local_ref<android::preference::DialogPreference>() const;


		EditTextPreference(local_ref< android::content::Context > const&, local_ref< android::util::AttributeSet > const&, jint);
		EditTextPreference(local_ref< android::content::Context > const&, local_ref< android::util::AttributeSet > const&);
		EditTextPreference(local_ref< android::content::Context > const&);
		void setText(local_ref< java::lang::String >  const&);
		local_ref< java::lang::String > getText();
		jboolean shouldDisableDependents();
		local_ref< android::widget::EditText > getEditText();
	}; //class EditTextPreference

} //namespace preference
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_PREFERENCE_EDITTEXTPREFERENCE_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_PREFERENCE_EDITTEXTPREFERENCE_HPP_IMPL
#define J2CPP_ANDROID_PREFERENCE_EDITTEXTPREFERENCE_HPP_IMPL

namespace j2cpp {



android::preference::EditTextPreference::operator local_ref<android::preference::DialogPreference>() const
{
	return local_ref<android::preference::DialogPreference>(get_jobject());
}


android::preference::EditTextPreference::EditTextPreference(local_ref< android::content::Context > const &a0, local_ref< android::util::AttributeSet > const &a1, jint a2)
: object<android::preference::EditTextPreference>(
	call_new_object<
		android::preference::EditTextPreference::J2CPP_CLASS_NAME,
		android::preference::EditTextPreference::J2CPP_METHOD_NAME(0),
		android::preference::EditTextPreference::J2CPP_METHOD_SIGNATURE(0)
	>(a0, a1, a2)
)
{
}



android::preference::EditTextPreference::EditTextPreference(local_ref< android::content::Context > const &a0, local_ref< android::util::AttributeSet > const &a1)
: object<android::preference::EditTextPreference>(
	call_new_object<
		android::preference::EditTextPreference::J2CPP_CLASS_NAME,
		android::preference::EditTextPreference::J2CPP_METHOD_NAME(1),
		android::preference::EditTextPreference::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}



android::preference::EditTextPreference::EditTextPreference(local_ref< android::content::Context > const &a0)
: object<android::preference::EditTextPreference>(
	call_new_object<
		android::preference::EditTextPreference::J2CPP_CLASS_NAME,
		android::preference::EditTextPreference::J2CPP_METHOD_NAME(2),
		android::preference::EditTextPreference::J2CPP_METHOD_SIGNATURE(2)
	>(a0)
)
{
}


void android::preference::EditTextPreference::setText(local_ref< java::lang::String > const &a0)
{
	return call_method<
		android::preference::EditTextPreference::J2CPP_CLASS_NAME,
		android::preference::EditTextPreference::J2CPP_METHOD_NAME(3),
		android::preference::EditTextPreference::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0);
}

local_ref< java::lang::String > android::preference::EditTextPreference::getText()
{
	return call_method<
		android::preference::EditTextPreference::J2CPP_CLASS_NAME,
		android::preference::EditTextPreference::J2CPP_METHOD_NAME(4),
		android::preference::EditTextPreference::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< java::lang::String >
	>(get_jobject());
}






jboolean android::preference::EditTextPreference::shouldDisableDependents()
{
	return call_method<
		android::preference::EditTextPreference::J2CPP_CLASS_NAME,
		android::preference::EditTextPreference::J2CPP_METHOD_NAME(10),
		android::preference::EditTextPreference::J2CPP_METHOD_SIGNATURE(10), 
		jboolean
	>(get_jobject());
}

local_ref< android::widget::EditText > android::preference::EditTextPreference::getEditText()
{
	return call_method<
		android::preference::EditTextPreference::J2CPP_CLASS_NAME,
		android::preference::EditTextPreference::J2CPP_METHOD_NAME(11),
		android::preference::EditTextPreference::J2CPP_METHOD_SIGNATURE(11), 
		local_ref< android::widget::EditText >
	>(get_jobject());
}




J2CPP_DEFINE_CLASS(android::preference::EditTextPreference,"android/preference/EditTextPreference")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,0,"<init>","(Landroid/content/Context;Landroid/util/AttributeSet;I)V")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,1,"<init>","(Landroid/content/Context;Landroid/util/AttributeSet;)V")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,2,"<init>","(Landroid/content/Context;)V")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,3,"setText","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,4,"getText","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,5,"onBindDialogView","(Landroid/view/View;)V")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,6,"onAddEditTextToDialogView","(Landroid/view/View;Landroid/widget/EditText;)V")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,7,"onDialogClosed","(Z)V")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,8,"onGetDefaultValue","(Landroid/content/res/TypedArray;I)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,9,"onSetInitialValue","(ZLjava/lang/Object;)V")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,10,"shouldDisableDependents","()Z")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,11,"getEditText","()Landroid/widget/EditText;")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,12,"onSaveInstanceState","()Landroid/os/Parcelable;")
J2CPP_DEFINE_METHOD(android::preference::EditTextPreference,13,"onRestoreInstanceState","(Landroid/os/Parcelable;)V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_PREFERENCE_EDITTEXTPREFERENCE_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
