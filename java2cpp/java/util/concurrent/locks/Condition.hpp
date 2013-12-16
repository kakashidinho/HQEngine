/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.util.concurrent.locks.Condition
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_CONCURRENT_LOCKS_CONDITION_HPP_DECL
#define J2CPP_JAVA_UTIL_CONCURRENT_LOCKS_CONDITION_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace util { class Date; } } }
namespace j2cpp { namespace java { namespace util { namespace concurrent { class TimeUnit; } } } }


#include <java/lang/Object.hpp>
#include <java/util/Date.hpp>
#include <java/util/concurrent/TimeUnit.hpp>


namespace j2cpp {

namespace java { namespace util { namespace concurrent { namespace locks {

	class Condition;
	class Condition
		: public object<Condition>
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

		explicit Condition(jobject jobj)
		: object<Condition>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;


		void await();
		void awaitUninterruptibly();
		jlong awaitNanos(jlong);
		jboolean await(jlong, local_ref< java::util::concurrent::TimeUnit >  const&);
		jboolean awaitUntil(local_ref< java::util::Date >  const&);
		void signal();
		void signalAll();
	}; //class Condition

} //namespace locks
} //namespace concurrent
} //namespace util
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_CONCURRENT_LOCKS_CONDITION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_CONCURRENT_LOCKS_CONDITION_HPP_IMPL
#define J2CPP_JAVA_UTIL_CONCURRENT_LOCKS_CONDITION_HPP_IMPL

namespace j2cpp {



java::util::concurrent::locks::Condition::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

void java::util::concurrent::locks::Condition::await()
{
	return call_method<
		java::util::concurrent::locks::Condition::J2CPP_CLASS_NAME,
		java::util::concurrent::locks::Condition::J2CPP_METHOD_NAME(0),
		java::util::concurrent::locks::Condition::J2CPP_METHOD_SIGNATURE(0), 
		void
	>(get_jobject());
}

void java::util::concurrent::locks::Condition::awaitUninterruptibly()
{
	return call_method<
		java::util::concurrent::locks::Condition::J2CPP_CLASS_NAME,
		java::util::concurrent::locks::Condition::J2CPP_METHOD_NAME(1),
		java::util::concurrent::locks::Condition::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject());
}

jlong java::util::concurrent::locks::Condition::awaitNanos(jlong a0)
{
	return call_method<
		java::util::concurrent::locks::Condition::J2CPP_CLASS_NAME,
		java::util::concurrent::locks::Condition::J2CPP_METHOD_NAME(2),
		java::util::concurrent::locks::Condition::J2CPP_METHOD_SIGNATURE(2), 
		jlong
	>(get_jobject(), a0);
}

jboolean java::util::concurrent::locks::Condition::await(jlong a0, local_ref< java::util::concurrent::TimeUnit > const &a1)
{
	return call_method<
		java::util::concurrent::locks::Condition::J2CPP_CLASS_NAME,
		java::util::concurrent::locks::Condition::J2CPP_METHOD_NAME(3),
		java::util::concurrent::locks::Condition::J2CPP_METHOD_SIGNATURE(3), 
		jboolean
	>(get_jobject(), a0, a1);
}

jboolean java::util::concurrent::locks::Condition::awaitUntil(local_ref< java::util::Date > const &a0)
{
	return call_method<
		java::util::concurrent::locks::Condition::J2CPP_CLASS_NAME,
		java::util::concurrent::locks::Condition::J2CPP_METHOD_NAME(4),
		java::util::concurrent::locks::Condition::J2CPP_METHOD_SIGNATURE(4), 
		jboolean
	>(get_jobject(), a0);
}

void java::util::concurrent::locks::Condition::signal()
{
	return call_method<
		java::util::concurrent::locks::Condition::J2CPP_CLASS_NAME,
		java::util::concurrent::locks::Condition::J2CPP_METHOD_NAME(5),
		java::util::concurrent::locks::Condition::J2CPP_METHOD_SIGNATURE(5), 
		void
	>(get_jobject());
}

void java::util::concurrent::locks::Condition::signalAll()
{
	return call_method<
		java::util::concurrent::locks::Condition::J2CPP_CLASS_NAME,
		java::util::concurrent::locks::Condition::J2CPP_METHOD_NAME(6),
		java::util::concurrent::locks::Condition::J2CPP_METHOD_SIGNATURE(6), 
		void
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(java::util::concurrent::locks::Condition,"java/util/concurrent/locks/Condition")
J2CPP_DEFINE_METHOD(java::util::concurrent::locks::Condition,0,"await","()V")
J2CPP_DEFINE_METHOD(java::util::concurrent::locks::Condition,1,"awaitUninterruptibly","()V")
J2CPP_DEFINE_METHOD(java::util::concurrent::locks::Condition,2,"awaitNanos","(J)J")
J2CPP_DEFINE_METHOD(java::util::concurrent::locks::Condition,3,"await","(JLjava/util/concurrent/TimeUnit;)Z")
J2CPP_DEFINE_METHOD(java::util::concurrent::locks::Condition,4,"awaitUntil","(Ljava/util/Date;)Z")
J2CPP_DEFINE_METHOD(java::util::concurrent::locks::Condition,5,"signal","()V")
J2CPP_DEFINE_METHOD(java::util::concurrent::locks::Condition,6,"signalAll","()V")

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_CONCURRENT_LOCKS_CONDITION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
