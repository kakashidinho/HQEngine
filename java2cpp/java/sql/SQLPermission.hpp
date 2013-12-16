/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.sql.SQLPermission
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SQL_SQLPERMISSION_HPP_DECL
#define J2CPP_JAVA_SQL_SQLPERMISSION_HPP_DECL


namespace j2cpp { namespace java { namespace io { class Serializable; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace security { class BasicPermission; } } }
namespace j2cpp { namespace java { namespace security { class Guard; } } }


#include <java/io/Serializable.hpp>
#include <java/lang/String.hpp>
#include <java/security/BasicPermission.hpp>
#include <java/security/Guard.hpp>


namespace j2cpp {

namespace java { namespace sql {

	class SQLPermission;
	class SQLPermission
		: public object<SQLPermission>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)

		explicit SQLPermission(jobject jobj)
		: object<SQLPermission>(jobj)
		{
		}

		operator local_ref<java::security::BasicPermission>() const;
		operator local_ref<java::security::Guard>() const;
		operator local_ref<java::io::Serializable>() const;


		SQLPermission(local_ref< java::lang::String > const&);
		SQLPermission(local_ref< java::lang::String > const&, local_ref< java::lang::String > const&);
	}; //class SQLPermission

} //namespace sql
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_SQL_SQLPERMISSION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SQL_SQLPERMISSION_HPP_IMPL
#define J2CPP_JAVA_SQL_SQLPERMISSION_HPP_IMPL

namespace j2cpp {



java::sql::SQLPermission::operator local_ref<java::security::BasicPermission>() const
{
	return local_ref<java::security::BasicPermission>(get_jobject());
}

java::sql::SQLPermission::operator local_ref<java::security::Guard>() const
{
	return local_ref<java::security::Guard>(get_jobject());
}

java::sql::SQLPermission::operator local_ref<java::io::Serializable>() const
{
	return local_ref<java::io::Serializable>(get_jobject());
}


java::sql::SQLPermission::SQLPermission(local_ref< java::lang::String > const &a0)
: object<java::sql::SQLPermission>(
	call_new_object<
		java::sql::SQLPermission::J2CPP_CLASS_NAME,
		java::sql::SQLPermission::J2CPP_METHOD_NAME(0),
		java::sql::SQLPermission::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



java::sql::SQLPermission::SQLPermission(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1)
: object<java::sql::SQLPermission>(
	call_new_object<
		java::sql::SQLPermission::J2CPP_CLASS_NAME,
		java::sql::SQLPermission::J2CPP_METHOD_NAME(1),
		java::sql::SQLPermission::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}



J2CPP_DEFINE_CLASS(java::sql::SQLPermission,"java/sql/SQLPermission")
J2CPP_DEFINE_METHOD(java::sql::SQLPermission,0,"<init>","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::sql::SQLPermission,1,"<init>","(Ljava/lang/String;Ljava/lang/String;)V")

} //namespace j2cpp

#endif //J2CPP_JAVA_SQL_SQLPERMISSION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
