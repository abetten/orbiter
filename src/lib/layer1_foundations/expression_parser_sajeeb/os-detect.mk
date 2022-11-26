#
# Detect OS and make necessary changes to flags
#

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
	OSFLAG += -D LINUX
	CPPFLAGS += -march=native -mtune=native
	LEXLIB += -lfl
endif
ifeq ($(UNAME_S),Darwin)
	OSFLAG += -D OSX
	CPPFLAGS += -I/opt/homebrew/opt/flex/include
	CPPFLAGS += -mmacosx-version-min=$(word 4, $(shell sw_vers))
	LEXLIB += -ll
	LDFLAGS += -L/usr/local/Cellar/flex/2.6.4_2/lib -L/usr/local/Cellar/bison/3.8.2/lib
endif

UNAME_P := $(shell uname -p)
ifeq ($(UNAME_P),x86_64)
	OSFLAG += -D AMD64
endif

ifneq ($(filter %86,$(UNAME_P)),)
	OSFLAG += -D IA32
endif
ifneq ($(filter arm%,$(UNAME_P)),)
	OSFLAG += -D ARM
endif