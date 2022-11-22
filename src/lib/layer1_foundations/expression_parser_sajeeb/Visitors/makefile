include compilers.mk
include compiler-flags.mk
include os-detect.mk
include auto_dependency_generation_variables.mk

TARGET_OBJECT_FILES = $(SRCS:.cpp=.o)

VISITOR_DIRECTORIES = $(wildcard *Visitors)

CLEAN_TARGETS = $(addsuffix Clean, $(VISITOR_DIRECTORIES))

all: $(TARGET_OBJECT_FILES) $(VISITOR_DIRECTORIES)

%Visitors: FORCE
	ln -sf $(shell pwd)/*.mk $(shell pwd)/$@
	$(MAKE) -C $@ all
	
clean: $(CLEAN_TARGETS) clean_base_target

include clean_base_target.mk

%Clean: FORCE
	-$(MAKE) -C $(subst Clean,,$@) clean

FORCE: ;

include auto_dependency_generation_target.mk
