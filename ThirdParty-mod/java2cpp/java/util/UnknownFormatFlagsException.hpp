/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.util.UnknownFormatFlagsException
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_UNKNOWNFORMATFLAGSEXCEPTION_HPP_DECL
#define J2CPP_JAVA_UTIL_UNKNOWNFORMATFLAGSEXCEPTION_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace util { class IllegalFormatException; } } }


#include <java/lang/String.hpp>
#include <java/util/IllegalFormatException.hpp>


namespace j2cpp {

namespace java { namespace util {

	class UnknownFormatFlagsException;
	class UnknownFormatFlagsException
		: public object<UnknownFormatFlagsException>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)

		explicit UnknownFormatFlagsException(jobject jobj)
		: object<UnknownFormatFlagsException>(jobj)
		{
		}

		operator local_ref<java::util::IllegalFormatException>() const;


		UnknownFormatFlagsException(local_ref< java::lang::String > const&);
		local_ref< java::lang::String > getFlags();
		local_ref< java::lang::String > getMessage();
	}; //class UnknownFormatFlagsException

} //namespace util
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_UNKNOWNFORMATFLAGSEXCEPTION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_UNKNOWNFORMATFLAGSEXCEPTION_HPP_IMPL
#define J2CPP_JAVA_UTIL_UNKNOWNFORMATFLAGSEXCEPTION_HPP_IMPL

namespace j2cpp {



java::util::UnknownFormatFlagsException::operator local_ref<java::util::IllegalFormatException>() const
{
	return local_ref<java::util::IllegalFormatException>(get_jobject());
}


java::util::UnknownFormatFlagsException::UnknownFormatFlagsException(local_ref< java::lang::String > const &a0)
: object<java::util::UnknownFormatFlagsException>(
	call_new_object<
		java::util::UnknownFormatFlagsException::J2CPP_CLASS_NAME,
		java::util::UnknownFormatFlagsException::J2CPP_METHOD_NAME(0),
		java::util::UnknownFormatFlagsException::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}


local_ref< java::lang::String > java::util::UnknownFormatFlagsException::getFlags()
{
	return call_method<
		java::util::UnknownFormatFlagsException::J2CPP_CLASS_NAME,
		java::util::UnknownFormatFlagsException::J2CPP_METHOD_NAME(1),
		java::util::UnknownFormatFlagsException::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::util::UnknownFormatFlagsException::getMessage()
{
	return call_method<
		java::util::UnknownFormatFlagsException::J2CPP_CLASS_NAME,
		java::util::UnknownFormatFlagsException::J2CPP_METHOD_NAME(2),
		java::util::UnknownFormatFlagsException::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::lang::String >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(java::util::UnknownFormatFlagsException,"java/util/UnknownFormatFlagsException")
J2CPP_DEFINE_METHOD(java::util::UnknownFormatFlagsException,0,"<init>","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::util::UnknownFormatFlagsException,1,"getFlags","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::util::UnknownFormatFlagsException,2,"getMessage","()Ljava/lang/String;")

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_UNKNOWNFORMATFLAGSEXCEPTION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION