# Makefile for QEMU TCG plugins on macOS
# Note: This is specific to macOS. Building on Linux requires different 
# flags and header paths (e.g., using pkg-config --cflags --libs glib-2.0).

CC = clang
GLIB_CFLAGS = $(shell brew --prefix glib 2>/dev/null || echo /opt/homebrew)
INCLUDE_PATH = /opt/homebrew/include/

CFLAGS = -fPIC -shared \
         -Wl,-undefined,dynamic_lookup \
         -I $(INCLUDE_PATH) \
         -isystem $(GLIB_CFLAGS)/include/glib-2.0 \
         -isystem $(GLIB_CFLAGS)/lib/glib-2.0/include

LDFLAGS = -L$(GLIB_CFLAGS)/lib -lglib-2.0

all: libinscount.dylib

libinscount.dylib: inscount.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f libinscount.dylib
