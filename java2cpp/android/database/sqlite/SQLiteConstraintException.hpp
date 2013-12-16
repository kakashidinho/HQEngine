/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.database.sqlite.SQLiteConstraintException
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_DATABASE_SQLITE_SQLITECONSTRAINTEXCEPTION_HPP_DECL
#define J2CPP_ANDROID_DATABASE_SQLITE_SQLITECONSTRAINTEXCEPTION_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace android { namespace database { namespace sqlite { class SQLiteException; } } } }


#include <android/database/sqlite/SQLiteException.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace android { namespace database { namespace sqlite {

	class SQLiteConstraintException;
	class SQLiteConstraintException
		: public object<SQLiteConstraintException>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)

		explicit SQLiteConstraintException(jobject jobj)
		: object<SQLiteConstraintException>(jobj)
		{
		}

		operator local_ref<android::database::sqlite::SQLiteException>() const;


		SQLiteConstraintException();
		SQLiteConstraintException(local_ref< java::lang::String > const&);
	}; //class SQLiteConstraintException

} //namespace sqlite
} //namespace database
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_DATABASE_SQLITE_SQLITECONSTRAINTEXCEPTION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_DATABASE_SQLITE_SQLITECONSTRAINTEXCEPTION_HPP_IMPL
#define J2CPP_ANDROID_DATABASE_SQLITE_SQLITECONSTRAINTEXCEPTION_HPP_IMPL

namespace j2cpp {



android::database::sqlite::SQLiteConstraintException::operator local_ref<android::database::sqlite::SQLiteException>() const
{
	return local_ref<android::database::sqlite::SQLiteException>(get_jobject());
}


android::database::sqlite::SQLiteConstraintException::SQLiteConstraintException()
: object<android::database::sqlite::SQLiteConstraintException>(
	call_new_object<
		android::database::sqlite::SQLiteConstraintException::J2CPP_CLASS_NAME,
		android::database::sqlite::SQLiteConstraintException::J2CPP_METHOD_NAME(0),
		android::database::sqlite::SQLiteConstraintException::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



android::database::sqlite::SQLiteConstraintException::SQLiteConstraintException(local_ref< java::lang::String > const &a0)
: object<android::database::sqlite::SQLiteConstraintException>(
	call_new_object<
		android::database::sqlite::SQLiteConstraintException::J2CPP_CLASS_NAME,
		android::database::sqlite::SQLiteConstraintException::J2CPP_METHOD_NAME(1),
		android::database::sqlite::SQLiteConstraintException::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



J2CPP_DEFINE_CLASS(android::database::sqlite::SQLiteConstraintException,"android/database/sqlite/SQLiteConstraintException")
J2CPP_DEFINE_METHOD(android::database::sqlite::SQLiteConstraintException,0,"<init>","()V")
J2CPP_DEFINE_METHOD(android::database::sqlite::SQLiteConstraintException,1,"<init>","(Ljava/lang/String;)V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_DATABASE_SQLITE_SQLITECONSTRAINTEXCEPTION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
