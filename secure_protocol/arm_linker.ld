/*
 * ARM Cortex-M4 Linker Script for bare metal compilation
 * This is a minimal stub script - modify for your specific hardware.
 */

MEMORY
{
    /* Default memory regions - adjust for your target hardware */
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 256K
    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 64K
}

SECTIONS
{
    .text :
    {
        *(.vectors)
        *(.text)
        *(.text.*)
        *(.rodata)
        *(.rodata.*)
        . = ALIGN(4);
    } > FLASH

    .data :
    {
        _sdata = .;
        *(.data)
        *(.data.*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss :
    {
        _sbss = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    _estack = ORIGIN(RAM) + LENGTH(RAM);
}
