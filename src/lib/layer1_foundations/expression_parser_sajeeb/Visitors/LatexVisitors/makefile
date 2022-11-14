include compilers.mk
include compiler-flags.mk
include os-detect.mk
include auto_dependency_generation_variables.mk

TARGET_OBJECT_FILES = $(SRCS:.cpp=.o)

all: $(TARGET_OBJECT_FILES)

clean: clean_base_target

include clean_base_target.mk


include auto_dependency_generation_target.mk
