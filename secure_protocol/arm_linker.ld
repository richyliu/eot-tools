/* Define memory regions */
MEMORY
{
  FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 512K
  RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 128K
}

/* Define output sections */
SECTIONS
{
  .text : 
  {
    PROVIDE(_text_start = .);
    *(.vectors)      /* Vector table */
    *(.text)         /* Code */
    *(.text.*)
    *(.rodata)       /* Read-only data */
    PROVIDE(_text_end = .);
  } > FLASH

  .data : 
  {
    PROVIDE(_sdata = .);    /* Start of .data in RAM (VMA) */
    *(.data)               /* Initialized data */
    *(.data.*)
    PROVIDE(_edata = .);    /* End of .data in RAM */
  } > RAM AT > FLASH

  /* _sidata is where .data is loaded in flash (LMA) */
  PROVIDE(_sidata = LOADADDR(.data));

  .bss : 
  {
    PROVIDE(_sbss = .);    /* Start of .bss */
    *(.bss)                /* Uninitialized data */
    *(.bss.*)
    *(COMMON)
    PROVIDE(_ebss = .);    /* End of .bss */
  } > RAM

  /* Stack and other sections */
  _estack = ORIGIN(RAM) + LENGTH(RAM);
}
