/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.io.DataInputStream
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_DATAINPUTSTREAM_HPP_DECL
#define J2CPP_JAVA_IO_DATAINPUTSTREAM_HPP_DECL


namespace j2cpp { namespace java { namespace io { class FilterInputStream; } } }
namespace j2cpp { namespace java { namespace io { class DataInput; } } }
namespace j2cpp { namespace java { namespace io { class InputStream; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/io/DataInput.hpp>
#include <java/io/FilterInputStream.hpp>
#include <java/io/InputStream.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace java { namespace io {

	class DataInputStream;
	class DataInputStream
		: public object<DataInputStream>
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
		J2CPP_DECLARE_METHOD(14)
		J2CPP_DECLARE_METHOD(15)
		J2CPP_DECLARE_METHOD(16)
		J2CPP_DECLARE_METHOD(17)
		J2CPP_DECLARE_METHOD(18)

		explicit DataInputStream(jobject jobj)
		: object<DataInputStream>(jobj)
		{
		}

		operator local_ref<java::io::FilterInputStream>() const;
		operator local_ref<java::io::DataInput>() const;


		DataInputStream(local_ref< java::io::InputStream > const&);
		jint read(local_ref< array<jbyte,1> >  const&);
		jint read(local_ref< array<jbyte,1> >  const&, jint, jint);
		jboolean readBoolean();
		jbyte readByte();
		jchar readChar();
		jdouble readDouble();
		jfloat readFloat();
		void readFully(local_ref< array<jbyte,1> >  const&);
		void readFully(local_ref< array<jbyte,1> >  const&, jint, jint);
		jint readInt();
		local_ref< java::lang::String > readLine();
		jlong readLong();
		jshort readShort();
		jint readUnsignedByte();
		jint readUnsignedShort();
		local_ref< java::lang::String > readUTF();
		static local_ref< java::lang::String > readUTF(local_ref< java::io::DataInput >  const&);
		jint skipBytes(jint);
	}; //class DataInputStream

} //namespace io
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_DATAINPUTSTREAM_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_IO_DATAINPUTSTREAM_HPP_IMPL
#define J2CPP_JAVA_IO_DATAINPUTSTREAM_HPP_IMPL

namespace j2cpp {



java::io::DataInputStream::operator local_ref<java::io::FilterInputStream>() const
{
	return local_ref<java::io::FilterInputStream>(get_jobject());
}

java::io::DataInputStream::operator local_ref<java::io::DataInput>() const
{
	return local_ref<java::io::DataInput>(get_jobject());
}


java::io::DataInputStream::DataInputStream(local_ref< java::io::InputStream > const &a0)
: object<java::io::DataInputStream>(
	call_new_object<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(0),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}


jint java::io::DataInputStream::read(local_ref< array<jbyte,1> > const &a0)
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(1),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(1), 
		jint
	>(get_jobject(), a0);
}

jint java::io::DataInputStream::read(local_ref< array<jbyte,1> > const &a0, jint a1, jint a2)
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(2),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(2), 
		jint
	>(get_jobject(), a0, a1, a2);
}

jboolean java::io::DataInputStream::readBoolean()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(3),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(3), 
		jboolean
	>(get_jobject());
}

jbyte java::io::DataInputStream::readByte()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(4),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(4), 
		jbyte
	>(get_jobject());
}

jchar java::io::DataInputStream::readChar()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(5),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(5), 
		jchar
	>(get_jobject());
}

jdouble java::io::DataInputStream::readDouble()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(6),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(6), 
		jdouble
	>(get_jobject());
}

jfloat java::io::DataInputStream::readFloat()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(7),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(7), 
		jfloat
	>(get_jobject());
}

void java::io::DataInputStream::readFully(local_ref< array<jbyte,1> > const &a0)
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(8),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(8), 
		void
	>(get_jobject(), a0);
}

void java::io::DataInputStream::readFully(local_ref< array<jbyte,1> > const &a0, jint a1, jint a2)
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(9),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(9), 
		void
	>(get_jobject(), a0, a1, a2);
}

jint java::io::DataInputStream::readInt()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(10),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(10), 
		jint
	>(get_jobject());
}

local_ref< java::lang::String > java::io::DataInputStream::readLine()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(11),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(11), 
		local_ref< java::lang::String >
	>(get_jobject());
}

jlong java::io::DataInputStream::readLong()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(12),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(12), 
		jlong
	>(get_jobject());
}

jshort java::io::DataInputStream::readShort()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(13),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(13), 
		jshort
	>(get_jobject());
}

jint java::io::DataInputStream::readUnsignedByte()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(14),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(14), 
		jint
	>(get_jobject());
}

jint java::io::DataInputStream::readUnsignedShort()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(15),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(15), 
		jint
	>(get_jobject());
}

local_ref< java::lang::String > java::io::DataInputStream::readUTF()
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(16),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(16), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > java::io::DataInputStream::readUTF(local_ref< java::io::DataInput > const &a0)
{
	return call_static_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(17),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(17), 
		local_ref< java::lang::String >
	>(a0);
}

jint java::io::DataInputStream::skipBytes(jint a0)
{
	return call_method<
		java::io::DataInputStream::J2CPP_CLASS_NAME,
		java::io::DataInputStream::J2CPP_METHOD_NAME(18),
		java::io::DataInputStream::J2CPP_METHOD_SIGNATURE(18), 
		jint
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(java::io::DataInputStream,"java/io/DataInputStream")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,0,"<init>","(Ljava/io/InputStream;)V")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,1,"read","([B)I")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,2,"read","([BII)I")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,3,"readBoolean","()Z")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,4,"readByte","()B")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,5,"readChar","()C")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,6,"readDouble","()D")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,7,"readFloat","()F")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,8,"readFully","([B)V")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,9,"readFully","([BII)V")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,10,"readInt","()I")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,11,"readLine","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,12,"readLong","()J")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,13,"readShort","()S")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,14,"readUnsignedByte","()I")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,15,"readUnsignedShort","()I")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,16,"readUTF","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,17,"readUTF","(Ljava/io/DataInput;)Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::io::DataInputStream,18,"skipBytes","(I)I")

} //namespace j2cpp

#endif //J2CPP_JAVA_IO_DATAINPUTSTREAM_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
