/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.nio.BufferUnderflowException
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NIO_BUFFERUNDERFLOWEXCEPTION_HPP_DECL
#define J2CPP_JAVA_NIO_BUFFERUNDERFLOWEXCEPTION_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class RuntimeException; } } }


#include <java/lang/RuntimeException.hpp>


namespace j2cpp {

namespace java { namespace nio {

	class BufferUnderflowException;
	class BufferUnderflowException
		: public object<BufferUnderflowException>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)

		explicit BufferUnderflowException(jobject jobj)
		: object<BufferUnderflowException>(jobj)
		{
		}

		operator local_ref<java::lang::RuntimeException>() const;


		BufferUnderflowException();
	}; //class BufferUnderflowException

} //namespace nio
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_NIO_BUFFERUNDERFLOWEXCEPTION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NIO_BUFFERUNDERFLOWEXCEPTION_HPP_IMPL
#define J2CPP_JAVA_NIO_BUFFERUNDERFLOWEXCEPTION_HPP_IMPL

namespace j2cpp {



java::nio::BufferUnderflowException::operator local_ref<java::lang::RuntimeException>() const
{
	return local_ref<java::lang::RuntimeException>(get_jobject());
}


java::nio::BufferUnderflowException::BufferUnderflowException()
: object<java::nio::BufferUnderflowException>(
	call_new_object<
		java::nio::BufferUnderflowException::J2CPP_CLASS_NAME,
		java::nio::BufferUnderflowException::J2CPP_METHOD_NAME(0),
		java::nio::BufferUnderflowException::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



J2CPP_DEFINE_CLASS(java::nio::BufferUnderflowException,"java/nio/BufferUnderflowException")
J2CPP_DEFINE_METHOD(java::nio::BufferUnderflowException,0,"<init>","()V")

} //namespace j2cpp

#endif //J2CPP_JAVA_NIO_BUFFERUNDERFLOWEXCEPTION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION