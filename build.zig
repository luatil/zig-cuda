const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // -----------------------------------------------------------------------
    // Step 1: compile the CUDA kernel to PTX via nvcc.
    // We use --ptx so only device code is emitted — no host object file.
    // -----------------------------------------------------------------------
    const nvcc_ptx = b.addSystemCommand(&.{
        "nvcc",
        "--ptx",
        "-arch=native",
        "-I/opt/cuda/include",
    });
    nvcc_ptx.addArg("-o");
    const ptx_file = nvcc_ptx.addOutputFileArg("add.ptx");
    nvcc_ptx.addFileArg(b.path("src/add.cu"));

    // -----------------------------------------------------------------------
    // Step 2: embed the PTX in a generated Zig module using @embedFile.
    // addWriteFiles creates a scratch directory; putting both the PTX copy
    // and the Zig wrapper in the same directory makes @embedFile work.
    // -----------------------------------------------------------------------
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(ptx_file, "add.ptx");
    const embed_zig = wf.add("cuda_ptx.zig",
        \\pub const bytes = @embedFile("add.ptx");
    );

    // -----------------------------------------------------------------------
    // Step 3: build the zig_cuda module and wire in the embedded PTX.
    // -----------------------------------------------------------------------
    const mod = b.addModule("zig_cuda", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });
    mod.addAnonymousImport("cuda_ptx", .{
        .root_source_file = embed_zig,
    });

    // -----------------------------------------------------------------------
    // Step 4: build the executable and link the CUDA Driver library.
    // We use the stub libcuda.so from the toolkit for link-time resolution;
    // the real libcuda.so is supplied by the NVIDIA driver at runtime.
    // -----------------------------------------------------------------------
    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zig_cuda", .module = mod },
        },
    });

    // link_libc is required. Without it, Zig's debug build places ~262 KB of
    // static TLS in .tbss, which overwrites the glibc fs_base layout that the
    // dynamic linker sets up. That causes cuInit → calloc to SIGSEGV. With
    // libc linked, Zig delegates to glibc's allocator and TLS drops to ~24 B.
    root_mod.link_libc = true;
    root_mod.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
    root_mod.linkSystemLibrary("cuda", .{});

    const exe = b.addExecutable(.{
        .name = "zig_cuda",
        .root_module = root_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
