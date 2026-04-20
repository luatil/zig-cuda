const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Compile the CUDA kernel to an object file via nvcc
    const nvcc = b.addSystemCommand(&.{
        "nvcc",
        "-c",
        "-arch=native",
        "--compiler-options",
        "-fPIC",
        "-I/opt/cuda/include",
    });
    nvcc.addArg("-o");
    const cuda_obj = nvcc.addOutputFileArg("add.o");
    nvcc.addFileArg(b.path("src/add.cu"));

    // The zig_cuda module exposes the CUDA bindings
    const mod = b.addModule("zig_cuda", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zig_cuda", .module = mod },
        },
    });

    // Link the CUDA object file and runtime library into the root module
    root_mod.addObjectFile(cuda_obj);
    root_mod.addLibraryPath(.{ .cwd_relative = "/opt/cuda/lib64" });
    root_mod.linkSystemLibrary("cudart", .{});
    root_mod.linkSystemLibrary("stdc++", .{});
    root_mod.addRPath(.{ .cwd_relative = "/opt/cuda/lib64" });

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
