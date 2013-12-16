/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.io.FileNotFoundException
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_FILENOTFOUNDEXCEPTION_HPP_DECL
#define J2CPP_JAVA_IO_FILENOTFOUNDEXCEPTION_HPP_DECL


namespace j2cpp { namespace java { namespace io { class IOException; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/io/IOException.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace java { namespace io {

	class FileNotFoundException;
	class FileNotFoundException
		: public object<FileNotFoundException>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)

		explicit FileNotFoundException(jobject jobj)
		: object<FileNotFoundException>(jobj)
		{
		}

		operator local_ref<java::io::IOException>() const;


		FileNotFoundException();
		FileNotFoundException(local_ref< java::lang::String > const&);
	}; //class FileNotFoundException

} //namespace io
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_FILENOTFOUNDEXCEPTION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_FILENOTFOUNDEXCEPTION_HPP_IMPL
#define J2CPP_JAVA_IO_FILENOTFOUNDEXCEPTION_HPP_IMPL

namespace j2cpp {



java::io::FileNotFoundException::operator local_ref<java::io::IOException>() const
{
	return local_ref<java::io::IOException>(get_jobject());
}


java::io::FileNotFoundException::FileNotFoundException()
: object<java::io::FileNotFoundException>(
	call_new_object<
		java::io::FileNotFoundException::J2CPP_CLASS_NAME,
		java::io::FileNotFoundException::J2CPP_METHOD_NAME(0),
		java::io::FileNotFoundException::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



java::io::FileNotFoundException::FileNotFoundException(local_ref< java::lang::String > const &a0)
: object<java::io::FileNotFoundException>(
	call_new_object<
		java::io::FileNotFoundException::J2CPP_CLASS_NAME,
		java::io::FileNotFoundException::J2CPP_METHOD_NAME(1),
		java::io::FileNotFoundException::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



J2CPP_DEFINE_CLASS(java::io::FileNotFoundException,"java/io/FileNotFoundException")
J2CPP_DEFINE_METHOD(java::io::FileNotFoundException,0,"<init>","()V")
J2CPP_DEFINE_METHOD(java::io::FileNotFoundException,1,"<init>","(Ljava/lang/String;)V")

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_FILENOTFOUNDEXCEPTION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
