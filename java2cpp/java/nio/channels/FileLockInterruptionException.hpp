/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.nio.channels.FileLockInterruptionException
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NIO_CHANNELS_FILELOCKINTERRUPTIONEXCEPTION_HPP_DECL
#define J2CPP_JAVA_NIO_CHANNELS_FILELOCKINTERRUPTIONEXCEPTION_HPP_DECL


namespace j2cpp { namespace java { namespace io { class IOException; } } }


#include <java/io/IOException.hpp>


namespace j2cpp {

namespace java { namespace nio { namespace channels {

	class FileLockInterruptionException;
	class FileLockInterruptionException
		: public object<FileLockInterruptionException>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)

		explicit FileLockInterruptionException(jobject jobj)
		: object<FileLockInterruptionException>(jobj)
		{
		}

		operator local_ref<java::io::IOException>() const;


		FileLockInterruptionException();
	}; //class FileLockInterruptionException

} //namespace channels
} //namespace nio
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_NIO_CHANNELS_FILELOCKINTERRUPTIONEXCEPTION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_NIO_CHANNELS_FILELOCKINTERRUPTIONEXCEPTION_HPP_IMPL
#define J2CPP_JAVA_NIO_CHANNELS_FILELOCKINTERRUPTIONEXCEPTION_HPP_IMPL

namespace j2cpp {



java::nio::channels::FileLockInterruptionException::operator local_ref<java::io::IOException>() const
{
	return local_ref<java::io::IOException>(get_jobject());
}


java::nio::channels::FileLockInterruptionException::FileLockInterruptionException()
: object<java::nio::channels::FileLockInterruptionException>(
	call_new_object<
		java::nio::channels::FileLockInterruptionException::J2CPP_CLASS_NAME,
		java::nio::channels::FileLockInterruptionException::J2CPP_METHOD_NAME(0),
		java::nio::channels::FileLockInterruptionException::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



J2CPP_DEFINE_CLASS(java::nio::channels::FileLockInterruptionException,"java/nio/channels/FileLockInterruptionException")
J2CPP_DEFINE_METHOD(java::nio::channels::FileLockInterruptionException,0,"<init>","()V")

} //namespace j2cpp

#endif //J2CPP_JAVA_NIO_CHANNELS_FILELOCKINTERRUPTIONEXCEPTION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
