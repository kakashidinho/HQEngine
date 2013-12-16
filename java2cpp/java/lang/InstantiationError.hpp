/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.lang.InstantiationError
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_LANG_INSTANTIATIONERROR_HPP_DECL
#define J2CPP_JAVA_LANG_INSTANTIATIONERROR_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class IncompatibleClassChangeError; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/lang/IncompatibleClassChangeError.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace java { namespace lang {

	class InstantiationError;
	class InstantiationError
		: public object<InstantiationError>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)

		explicit InstantiationError(jobject jobj)
		: object<InstantiationError>(jobj)
		{
		}

		operator local_ref<java::lang::IncompatibleClassChangeError>() const;


		InstantiationError();
		InstantiationError(local_ref< java::lang::String > const&);
	}; //class InstantiationError

} //namespace lang
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_LANG_INSTANTIATIONERROR_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_LANG_INSTANTIATIONERROR_HPP_IMPL
#define J2CPP_JAVA_LANG_INSTANTIATIONERROR_HPP_IMPL

namespace j2cpp {



java::lang::InstantiationError::operator local_ref<java::lang::IncompatibleClassChangeError>() const
{
	return local_ref<java::lang::IncompatibleClassChangeError>(get_jobject());
}


java::lang::InstantiationError::InstantiationError()
: object<java::lang::InstantiationError>(
	call_new_object<
		java::lang::InstantiationError::J2CPP_CLASS_NAME,
		java::lang::InstantiationError::J2CPP_METHOD_NAME(0),
		java::lang::InstantiationError::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



java::lang::InstantiationError::InstantiationError(local_ref< java::lang::String > const &a0)
: object<java::lang::InstantiationError>(
	call_new_object<
		java::lang::InstantiationError::J2CPP_CLASS_NAME,
		java::lang::InstantiationError::J2CPP_METHOD_NAME(1),
		java::lang::InstantiationError::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



J2CPP_DEFINE_CLASS(java::lang::InstantiationError,"java/lang/InstantiationError")
J2CPP_DEFINE_METHOD(java::lang::InstantiationError,0,"<init>","()V")
J2CPP_DEFINE_METHOD(java::lang::InstantiationError,1,"<init>","(Ljava/lang/String;)V")

} //namespace j2cpp

#endif //J2CPP_JAVA_LANG_INSTANTIATIONERROR_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
