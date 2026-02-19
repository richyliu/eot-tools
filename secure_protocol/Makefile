# Makefile for secure_protocol with abstraction layers

# Default target
all: main

# Source files
SOURCES = main.c devices.c comm_socket.c ext_timer.c ext_io.c ext_random.c ext_utils.c
DEPS = micro-ecc/uECC.c sha256/sha256.c

# Compiler settings
CC = gcc
CFLAGS = -flto -Os -Wall
LDFLAGS = -flto

# Main target
main: $(SOURCES) $(DEPS)
	$(CC) $(CFLAGS) -o main $(SOURCES) $(DEPS) $(LDFLAGS)

# Clean target
clean:
	rm -f main

# Build with custom timer/IO implementations
# Usage: make CUSTOM_EXT=timer_io.c
ifdef CUSTOM_EXT
main: $(SOURCES) $(DEPS) $(CUSTOM_EXT)
	$(CC) $(CFLAGS) -o main $(SOURCES) $(CUSTOM_EXT) $(DEPS) $(LDFLAGS)
endif

# Debug build with symbols and no optimization
debug: CFLAGS = -g -O0 -Wall -DDEBUG
debug: main

.PHONY: all clean debug
