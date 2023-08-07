#ifndef TCH_CXX_WRAPPER_UTILS_H
#define TCH_CXX_WRAPPER_UTILS_H

#include <c10/util/intrusive_ptr.h>


template<class T>
using UniqueIntrusivePtr = std::unique_ptr<c10::intrusive_ptr<T>>;

class MyIntrusiveTarget : public c10::intrusive_ptr_target {
public:
    MyIntrusiveTarget() {
        printf("create a target\n");
    }

    ~MyIntrusiveTarget() {
        printf("destroy a target\n");
    }
};

class IntrusivePtrContainer {
public:
    c10::intrusive_ptr<MyIntrusiveTarget> inner;

    IntrusivePtrContainer(c10::intrusive_ptr<MyIntrusiveTarget> value) : inner(value) {}

    IntrusivePtrContainer(const IntrusivePtrContainer &container) {
        inner = std::move(c10::intrusive_ptr<MyIntrusiveTarget>(container.inner));
    }
};

IntrusivePtrContainer new_target();

IntrusivePtrContainer
clone_target(IntrusivePtrContainer &src);


#endif //TCH_UTILS_H
