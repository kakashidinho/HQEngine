/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.database.DataSetObservable
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_DATABASE_DATASETOBSERVABLE_HPP_DECL
#define J2CPP_ANDROID_DATABASE_DATASETOBSERVABLE_HPP_DECL


namespace j2cpp { namespace android { namespace database { class Observable; } } }


#include <android/database/Observable.hpp>


namespace j2cpp {

namespace android { namespace database {

	class DataSetObservable;
	class DataSetObservable
		: public object<DataSetObservable>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)

		explicit DataSetObservable(jobject jobj)
		: object<DataSetObservable>(jobj)
		{
		}

		operator local_ref<android::database::Observable>() const;


		DataSetObservable();
		void notifyChanged();
		void notifyInvalidated();
	}; //class DataSetObservable

} //namespace database
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_DATABASE_DATASETOBSERVABLE_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_DATABASE_DATASETOBSERVABLE_HPP_IMPL
#define J2CPP_ANDROID_DATABASE_DATASETOBSERVABLE_HPP_IMPL

namespace j2cpp {



android::database::DataSetObservable::operator local_ref<android::database::Observable>() const
{
	return local_ref<android::database::Observable>(get_jobject());
}


android::database::DataSetObservable::DataSetObservable()
: object<android::database::DataSetObservable>(
	call_new_object<
		android::database::DataSetObservable::J2CPP_CLASS_NAME,
		android::database::DataSetObservable::J2CPP_METHOD_NAME(0),
		android::database::DataSetObservable::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}


void android::database::DataSetObservable::notifyChanged()
{
	return call_method<
		android::database::DataSetObservable::J2CPP_CLASS_NAME,
		android::database::DataSetObservable::J2CPP_METHOD_NAME(1),
		android::database::DataSetObservable::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject());
}

void android::database::DataSetObservable::notifyInvalidated()
{
	return call_method<
		android::database::DataSetObservable::J2CPP_CLASS_NAME,
		android::database::DataSetObservable::J2CPP_METHOD_NAME(2),
		android::database::DataSetObservable::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(android::database::DataSetObservable,"android/database/DataSetObservable")
J2CPP_DEFINE_METHOD(android::database::DataSetObservable,0,"<init>","()V")
J2CPP_DEFINE_METHOD(android::database::DataSetObservable,1,"notifyChanged","()V")
J2CPP_DEFINE_METHOD(android::database::DataSetObservable,2,"notifyInvalidated","()V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_DATABASE_DATASETOBSERVABLE_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION