# Why `link_libc = true` is required

Without this flag, `cuInit` crashes with SIGSEGV inside `calloc`. Here is exactly why.

## The fork: two different `_start` implementations

`link_libc` controls which startup code runs. In `std/start.zig`:

```zig
if (builtin.link_libc and @hasDecl(root, "main")) {
    @export(&main, .{ .name = "main" });  // glibc's _start will call this
} else {
    // Zig provides its own _start → posixCallMainAndExit
}
```

**Without `link_libc`**: Zig's `_start` calls `posixCallMainAndExit`, which calls `std.os.linux.tls.initStatic(phdrs)`.

**With `link_libc`**: Zig exports a C-ABI `main`, and glibc's `_start` calls `__libc_start_main → main`. Zig's TLS init is skipped entirely:

```zig
// std/start.zig
if (!builtin.single_threaded and !builtin.link_libc) {
    std.os.linux.tls.initStatic(phdrs);  // skipped when link_libc = true
}
```

## What `initStatic` does

`std/os/linux/tls.zig` computes the address for Zig's TLS block and then calls:

```zig
// tls.zig:247
const rc = @call(.always_inline, linux.syscall2, .{ .arch_prctl, linux.ARCH.SET_FS, addr });
```

This issues `arch_prctl(ARCH_SET_FS, zig_tls_addr)`, overwriting the `fs` base register.

## Why overwriting `fs` breaks glibc

On x86-64, `fs` is the thread pointer. Every TLS variable access compiles to an `fs`-relative load:

```asm
mov %fs:(-0x60), %rax    ; load thread_arena (glibc's malloc arena pointer)
```

The Linux dynamic linker (`ld-linux-x86-64.so.2`) calls `ARCH_SET_FS` during process startup to point `fs` at its own TLS layout, which includes:

- `tcbhead_t` (the thread control block) at `fs:0`
- glibc's malloc arena pointer (`thread_arena`) at a fixed negative offset
- pthread-internal state

When Zig calls `ARCH_SET_FS` with its own address, all those offsets now point into Zig's TLS block instead. glibc's TLS state becomes garbage.

## Why `calloc` is the crash site

`cuInit` allocates memory internally using `calloc`. glibc's `calloc` calls `__libc_malloc`, which accesses `thread_arena` via TLS to find the current thread's malloc arena. After Zig's `ARCH_SET_FS`, that offset points into Zig's allocator state rather than a `malloc_state *`. glibc dereferences it as a pointer, hits unmapped memory, and SIGSEGV.

The GDB backtrace confirms this:

```
#0  calloc ()           from /usr/lib/libc.so.6
#1  ?? ()               from /usr/lib/libcuda.so.1
#2  ?? ()               from /usr/lib/libcuda.so.1
#3  cuInit ()           from /usr/lib/libcuda.so.1
#4  root.Context.init ()
#5  main.main ()
```

## Why `link_libc` fixes it

With `link_libc = true`:

1. Zig skips `initStatic` / `ARCH_SET_FS` entirely.
2. `ld.so` and glibc's `__libc_start_main` cooperate to set up one unified TLS layout.
3. Zig's 262 KB TLS block (from `.tdata`/`.tbss`) is placed within that layout based on the `PT_TLS` ELF segment — it is still there, just at the correct offsets.
4. `fs` is never clobbered, so glibc's `thread_arena` and all other TLS variables remain valid for the lifetime of the process.
