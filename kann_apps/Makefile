arch := kv3-2
cluster-system := cos
use-module := strict-flags opencl_linux opencl-kernel

cppflags += $(shell pkg-config --cflags kann-service)
cppflags += -g3 -O2 -std=c++14 -fPIC -Werror=pedantic
cppflags += -Iinclude

kann_lflags := $(shell pkg-config --libs kann-service)
# Common flags

all:
kann_multi_cnn-srcs := kann_multi_cnn.cpp
kann_multi_cnn-lflags := $(kann_lflags)
host-bin += kann_multi_cnn

include $(KALRAY_TOOLCHAIN_DIR)/share/make/Makefile.kalray
