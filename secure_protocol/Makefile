# Makefile for secure_protocol with abstraction layers
# Supports Unix (default) and ARM Cortex-M4 targets

# Target selection (default to unix)
TARGET ?= unix

# Source files (platform-independent)
CORE_SOURCES = main.c devices.c crypto.c comm_common.c
DEPS = micro-ecc/uECC.c sha256/sha256.c

ifeq ($(TARGET),arm)
    # ARM toolchain
    # CC = arm-none-eabi-gcc
    CC = /opt/homebrew/opt/arm-none-eabi-gcc@9/bin/arm-none-eabi-gcc
    OBJCOPY = /opt/homebrew/opt/arm-none-eabi-binutils/bin/arm-none-eabi-objcopy
    
    # ARM Cortex-M4 specific flags
    MCU_FLAGS = -mcpu=cortex-m4 -mthumb -mfloat-abi=soft
    CFLAGS = $(MCU_FLAGS) -Os -Wall -fomit-frame-pointer -DTARGET_ARM -ffreestanding -nostdlib -g
    LDFLAGS = $(MCU_FLAGS) -nostdlib -T arm_linker.ld
    
    # micro-ecc optimization for ARM/Thumb-2
    UECC_FLAGS = -DuECC_OPTIMIZATION_LEVEL=3
    
    # Use ARM stub implementations
    EXT_SOURCES = comm_arm.c ext_support_arm.c arm_support.c uart.c
    OUTPUT_EOT = eot.elf
    OUTPUT_HOT = hot.elf
    
else
    # Unix target
    CC = gcc
    CFLAGS = -flto -Os -Wall -DTARGET_UNIX -g
    LDFLAGS = -flto
    UECC_FLAGS =
    EXT_SOURCES = comm_socket.c ext_support.c
    OUTPUT_EOT = eot
    OUTPUT_HOT = hot
endif

# All sources
SOURCES = $(CORE_SOURCES) $(EXT_SOURCES)

# Default target
.PHONY: all_platforms
all_platforms: unix arm

.PHONY: all
all: $(OUTPUT_EOT) $(OUTPUT_HOT)

# EOT build target
$(OUTPUT_EOT): $(SOURCES) $(DEPS)
	$(CC) $(CFLAGS) $(UECC_FLAGS) -DEOT_DEVICE -o $(OUTPUT_EOT) $(SOURCES) $(DEPS) $(LDFLAGS)
ifeq ($(TARGET),arm)
	$(OBJCOPY) -O binary $(OUTPUT_EOT) eot.bin
endif

# HOT build target
$(OUTPUT_HOT): $(SOURCES) $(DEPS)
	$(CC) $(CFLAGS) $(UECC_FLAGS) -o $(OUTPUT_HOT) $(SOURCES) $(DEPS) $(LDFLAGS)
ifeq ($(TARGET),arm)
	$(OBJCOPY) -O binary $(OUTPUT_HOT) hot.bin
endif

# Clean target
.PHONY: clean
clean:
	rm -f eot hot eot.elf hot.elf eot.bin hot.bin

# Debug build with symbols and no optimization
.PHONY: debug
debug: CFLAGS += -g -O0 -DDEBUG
debug: $(OUTPUT_EOT) $(OUTPUT_HOT)

# Build with custom timer/IO implementations
# Usage: make CUSTOM_EXT=timer_io.c
ifdef CUSTOM_EXT
$(OUTPUT_EOT): $(SOURCES) $(DEPS) $(CUSTOM_EXT)
	$(CC) $(CFLAGS) -DEOT_DEVICE -o $(OUTPUT_EOT) $(SOURCES) $(CUSTOM_EXT) $(DEPS) $(LDFLAGS)
$(OUTPUT_HOT): $(SOURCES) $(DEPS) $(CUSTOM_EXT)
	$(CC) $(CFLAGS) -o $(OUTPUT_HOT) $(SOURCES) $(CUSTOM_EXT) $(DEPS) $(LDFLAGS)
endif

# Convenience targets
.PHONY: arm
arm:
	$(MAKE) TARGET=arm all

.PHONY: unix
unix:
	$(MAKE) TARGET=unix all
