all: main

main: main.c devices.c
	gcc -flto -Os -o main $^ micro-ecc/uECC.c sha256/sha256.c

clean:
	rm -f main
