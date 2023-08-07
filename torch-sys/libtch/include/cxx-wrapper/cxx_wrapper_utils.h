#ifndef TCH_CXX_WRAPPER_UTILS_H
#define TCH_CXX_WRAPPER_UTILS_H

#include <c10/util/intrusive_ptr.h>

#define CREATE_CONTAINER_CLASS(CLASS_NAME_, WRAP_TYPE_) \
class CLASS_NAME_ { \
public: \
    c10::intrusive_ptr<WRAP_TYPE_> inner; \
 \
    explicit CLASS_NAME_(std::unique_ptr<WRAP_TYPE_> value) noexcept: inner(std::move(value)) {} \
 \
    CLASS_NAME_(const CLASS_NAME_ &container) { \
        inner = std::move(c10::intrusive_ptr<WRAP_TYPE_>(container.inner)); \
    } \
 \
    CLASS_NAME_ clone() const { \
        return CLASS_NAME_(*this); \
    } \
};

class MyIntrusiveTarget : public c10::intrusive_ptr_target {
public:
    MyIntrusiveTarget() {
        printf("create a target\n");
    }

    ~MyIntrusiveTarget() override {
        printf("destroy a target\n");
    }
};

CREATE_CONTAINER_CLASS(Wrapper, MyIntrusiveTarget)

#endif //TCH_CXX_WRAPPER_UTILS_H
