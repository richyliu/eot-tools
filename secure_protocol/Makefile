# Makefile for secure_protocol with abstraction layers
# Supports Unix (default) and ARM Cortex-M4 targets

# Target selection (default to unix)
TARGET ?= unix

# Source files (platform-independent)
CORE_SOURCES = main.c devices.c
DEPS = micro-ecc/uECC.c sha256/sha256.c

ifeq ($(TARGET),arm)
    # ARM toolchain
    CC = arm-none-eabi-gcc
    OBJCOPY = arm-none-eabi-objcopy
    
    # ARM Cortex-M4 specific flags
    MCU_FLAGS = -mcpu=cortex-m4 -mthumb -mfloat-abi=soft
    CFLAGS = $(MCU_FLAGS) -Os -Wall -fomit-frame-pointer -DTARGET_ARM -ffreestanding -nostdlib
    LDFLAGS = $(MCU_FLAGS) -nostdlib -T arm_linker.ld
    
    # micro-ecc optimization for ARM/Thumb-2
    UECC_FLAGS = -DuECC_OPTIMIZATION_LEVEL=3
    
    # Use ARM stub implementations
    EXT_SOURCES = comm_arm.c ext_timer_arm.c ext_io_arm.c ext_random_arm.c ext_utils_arm.c arm_support.c
    OUTPUT = main.elf
    
else
    # Unix target
    CC = gcc
    CFLAGS = -flto -Os -Wall -DTARGET_UNIX
    LDFLAGS = -flto
    UECC_FLAGS =
    EXT_SOURCES = comm_socket.c ext_timer.c ext_io.c ext_random.c ext_utils.c
    OUTPUT = main
endif

# All sources
SOURCES = $(CORE_SOURCES) $(EXT_SOURCES)

# Default target
.PHONY: all
all: $(OUTPUT)

# Main build target
$(OUTPUT): $(SOURCES) $(DEPS)
	$(CC) $(CFLAGS) $(UECC_FLAGS) -o $(OUTPUT) $(SOURCES) $(DEPS) $(LDFLAGS)
ifeq ($(TARGET),arm)
	$(OBJCOPY) -O binary main.elf main.bin
	$(OBJCOPY) -O ihex main.elf main.hex
endif

# Clean target
.PHONY: clean
clean:
	rm -f main main.elf main.bin main.hex

# Debug build with symbols and no optimization
.PHONY: debug
debug: CFLAGS += -g -O0 -DDEBUG
debug: $(OUTPUT)

# Build with custom timer/IO implementations
# Usage: make CUSTOM_EXT=timer_io.c
ifdef CUSTOM_EXT
$(OUTPUT): $(SOURCES) $(DEPS) $(CUSTOM_EXT)
	$(CC) $(CFLAGS) -o $(OUTPUT) $(SOURCES) $(CUSTOM_EXT) $(DEPS) $(LDFLAGS)
endif

# Convenience targets
.PHONY: arm
arm:
	$(MAKE) TARGET=arm

.PHONY: unix
unix:
	$(MAKE) TARGET=unix
