CC := gcc
CCFLAGS := \
  -std=gnu99 \
  -Wall \
  -Wno-unused \
  -Werror \
  -g \
  -O0 \
  -Isrc \
  -Isrc/third_party \
  -Isrc/utils

LDFLAGS := \
  -lm \
  -lX11 \
  -lGL \
  -lGLU \
  -Llib \
  -ltensorflowlite_c

TEST_CCFLAGS := \
  -fsanitize=address \
  -fsanitize=undefined \
  -fno-sanitize-recover=all \
  -fsanitize=float-divide-by-zero \
  -fsanitize=float-cast-overflow \
  -fno-sanitize=null \
  -fno-sanitize=alignment \
  -fno-omit-frame-pointer

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)$*.d
TEST_DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)$*_test.d

BUILDDIR = build/
OBJDIR := $(BUILDDIR)obj/
BINDIR := $(BUILDDIR)bin/
DEPDIR := $(BUILDDIR)dep/
LIBDIR := $(BUILDDIR)lib/

OBJS := $(addprefix $(OBJDIR),$(subst .c,.o,$(SRCS)))
TEST_OBJS := $(addprefix $(OBJDIR),$(subst .c,.o,$(TEST_SRCS)))

.PHONY: all clean test

all: \
  $(BINDIR)file_utils_test \
  $(BINDIR)string_utils_test \
  $(BINDIR)yargs_test \
  $(BINDIR)app_main_test \
  $(BINDIR)tflite_main_test \
  $(BINDIR)tflite_face

clean:
	rm -rf $(BUILDDIR)

clean_src:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)
	rm -rf $(DEPDIR)

test: \
  run_file_utils_test \
  run_string_utils_test \
  run_yargs_test \
  run_tflite_main_test \
  run_app_main_test

$(OBJDIR)%.o: %.c $(DEPDIR)/%.d | $(DEPDIR)
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEPDIR)$*_test.d)
	$(CC) $(CCFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJDIR)%_test.o: %_test.c $(DEPDIR)/%.d | $(DEPDIR)
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEPDIR)$*.d)
	$(CC) $(CCFLAGS) $(TEST_CCFLAGS) $(TEST_DEPFLAGS) -c $< -o $@

$(BINDIR)file_utils_test: \
  $(OBJDIR)src/utils/file_utils_test.o \
  $(OBJDIR)src/utils/string_utils.o
	@mkdir -p $(dir $@) 
	$(CC) $(CCFLAGS) $(TEST_CCFLAGS) $^ -o $@

run_file_utils_test: $(BINDIR)file_utils_test
	$<

$(BINDIR)string_utils_test: \
  $(OBJDIR)src/utils/string_utils_test.o
	@mkdir -p $(dir $@) 
	$(CC) $(CCFLAGS) $(TEST_CCFLAGS) $^ -o $@

run_string_utils_test: $(BINDIR)string_utils_test
	$<

$(BINDIR)yargs_test: \
  $(OBJDIR)src/utils/string_utils.o \
  $(OBJDIR)src/utils/yargs_test.o
	@mkdir -p $(dir $@) 
	$(CC) $(CCFLAGS) $(TEST_CCFLAGS) $^ -o $@

run_yargs_test: $(BINDIR)yargs_test
	$<

$(BINDIR)app_main_test: \
 $(OBJDIR)src/app_main_test.o \
 $(OBJDIR)src/capture_main.o \
 $(OBJDIR)src/tflite_main.o \
 $(OBJDIR)src/window_main.o \
 $(OBJDIR)src/third_party/lodepng.o \
 $(OBJDIR)src/utils/file_utils.o \
 $(OBJDIR)src/utils/string_utils.o \
 $(OBJDIR)src/utils/yargs.o
	@mkdir -p $(dir $@) 
	$(CC) $(CCFLAGS) $(TEST_CCFLAGS) $^ -o $@ $(LDFLAGS)

run_app_main_test: $(BINDIR)app_main_test
	$<

$(BINDIR)tflite_main_test: \
 $(OBJDIR)src/tflite_main_test.o \
 $(OBJDIR)src/capture_main.o \
 $(OBJDIR)src/test_image_192.o \
 $(OBJDIR)src/utils/file_utils.o \
 $(OBJDIR)src/utils/string_utils.o \
 $(OBJDIR)src/utils/yargs.o
	@mkdir -p $(dir $@) 
	$(CC) $(CCFLAGS) $(TEST_CCFLAGS) $^ -o $@ $(LDFLAGS)

run_tflite_main_test: $(BINDIR)tflite_main_test
	$<

$(BINDIR)tflite_face: \
 $(OBJDIR)src/app_main.o \
 $(OBJDIR)src/capture_main.o \
 $(OBJDIR)src/main.o \
 $(OBJDIR)src/tflite_main.o \
 $(OBJDIR)src/window_main.o \
 $(OBJDIR)src/third_party/lodepng.o \
 $(OBJDIR)src/utils/file_utils.o \
 $(OBJDIR)src/utils/string_utils.o \
 $(OBJDIR)src/utils/yargs.o
	@mkdir -p $(dir $@) 
	$(CC) $^ -o $@ $(LDFLAGS)

$(DEPDIR): ; @mkdir -p $@

SRCS := $(shell find src/ -type f -name '*.c')
DEPFILES := $(SRCS:%.c=$(DEPDIR)/%.d)
$(DEPFILES):

include $(wildcard $(DEPFILES))