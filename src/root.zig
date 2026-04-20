const std = @import("std");

// The PTX source is embedded at compile time by the build system.
const cuda_ptx = @import("cuda_ptx");

// ---------------------------------------------------------------------------
// CUDA Driver API types
// ---------------------------------------------------------------------------

pub const CUdevice = c_int;
pub const CUdeviceptr = u64;

pub const CUresult = enum(c_int) {
    success = 0,
    _,

    pub fn check(self: CUresult) !void {
        if (self != .success) {
            std.debug.print("CUDA Driver error: {d}\n", .{@intFromEnum(self)});
            return error.CudaDriverError;
        }
    }
};

pub const CUctx_st = opaque {};
pub const CUmod_st = opaque {};
pub const CUfunc_st = opaque {};
pub const CUstream_st = opaque {};

pub const CUcontext = *CUctx_st;
pub const CUmodule = *CUmod_st;
pub const CUfunction = *CUfunc_st;

// ---------------------------------------------------------------------------
// CUDA Driver API extern declarations
// (versioned names are the real symbols in libcuda.so)
// ---------------------------------------------------------------------------

extern fn cuInit(flags: c_uint) CUresult;
extern fn cuDeviceGet(device: *CUdevice, ordinal: c_int) CUresult;
extern fn cuCtxCreate_v2(pctx: *CUcontext, flags: c_uint, dev: CUdevice) CUresult;
extern fn cuCtxDestroy_v2(ctx: CUcontext) CUresult;
extern fn cuModuleLoadData(module: *CUmodule, image: *const anyopaque) CUresult;
extern fn cuModuleUnload(hmod: CUmodule) CUresult;
extern fn cuModuleGetFunction(hfunc: *CUfunction, hmod: CUmodule, name: [*:0]const u8) CUresult;
extern fn cuMemAlloc_v2(dptr: *CUdeviceptr, bytesize: usize) CUresult;
extern fn cuMemFree_v2(dptr: CUdeviceptr) CUresult;
extern fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const anyopaque, size: usize) CUresult;
extern fn cuMemcpyDtoH_v2(dst: *anyopaque, src: CUdeviceptr, size: usize) CUresult;
extern fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: ?*CUstream_st,
    kernelParams: [*]?*anyopaque,
    extra: ?[*]?*anyopaque,
) CUresult;

// ---------------------------------------------------------------------------
// Higher-level Context that owns a CUDA context and the loaded kernel module
// ---------------------------------------------------------------------------

pub const Context = struct {
    ctx: CUcontext,
    module: CUmodule,
    add_fn: CUfunction,
    sub_fn: CUfunction,

    pub fn init() !Context {
        try cuInit(0).check();

        var device: CUdevice = undefined;
        try cuDeviceGet(&device, 0).check();

        var ctx: CUcontext = undefined;
        try cuCtxCreate_v2(&ctx, 0, device).check();
        errdefer _ = cuCtxDestroy_v2(ctx);

        // Load the kernel from the PTX bytes embedded at compile time.
        var module: CUmodule = undefined;
        try cuModuleLoadData(&module, @ptrCast(cuda_ptx.bytes)).check();
        errdefer _ = cuModuleUnload(module);

        var add_fn: CUfunction = undefined;
        try cuModuleGetFunction(&add_fn, module, "add_kernel").check();

        var sub_fn: CUfunction = undefined;
        try cuModuleGetFunction(&sub_fn, module, "sub_kernel").check();

        return .{ .ctx = ctx, .module = module, .add_fn = add_fn, .sub_fn = sub_fn };
    }

    pub fn deinit(self: Context) void {
        _ = cuModuleUnload(self.module);
        _ = cuCtxDestroy_v2(self.ctx);
    }

    /// Element-wise GPU addition. All slices must have the same length.
    pub fn add(self: Context, a: []const f32, b: []const f32, c: []f32) !void {
        std.debug.assert(a.len == b.len and b.len == c.len);
        const n = a.len;
        const size = n * @sizeOf(f32);

        // Allocate device buffers
        var d_a: CUdeviceptr = undefined;
        var d_b: CUdeviceptr = undefined;
        var d_c: CUdeviceptr = undefined;
        try cuMemAlloc_v2(&d_a, size).check();
        defer _ = cuMemFree_v2(d_a);
        try cuMemAlloc_v2(&d_b, size).check();
        defer _ = cuMemFree_v2(d_b);
        try cuMemAlloc_v2(&d_c, size).check();
        defer _ = cuMemFree_v2(d_c);

        // Upload inputs
        try cuMemcpyHtoD_v2(d_a, @ptrCast(a.ptr), size).check();
        try cuMemcpyHtoD_v2(d_b, @ptrCast(b.ptr), size).check();

        // Launch kernel: pass pointers-to-args as required by the Driver API
        const block_size: c_uint = 256;
        const grid_size: c_uint = @intCast((n + block_size - 1) / block_size);
        var n_arg: c_int = @intCast(n);
        var params = [_]?*anyopaque{
            @ptrCast(&d_a),
            @ptrCast(&d_b),
            @ptrCast(&d_c),
            @ptrCast(&n_arg),
        };
        try cuLaunchKernel(
            self.add_fn,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            null,
            &params,
            null,
        ).check();

        // Download result (implicit sync — no stream means default stream)
        try cuMemcpyDtoH_v2(@ptrCast(c.ptr), d_c, size).check();
    }

    /// Element-wise GPU subtraction. All slices must have the same length.
    pub fn sub(self: Context, a: []const f32, b: []const f32, c: []f32) !void {
        std.debug.assert(a.len == b.len and b.len == c.len);
        const n = a.len;
        const size = n * @sizeOf(f32);

        // Allocate device buffers
        var d_a: CUdeviceptr = undefined;
        var d_b: CUdeviceptr = undefined;
        var d_c: CUdeviceptr = undefined;
        try cuMemAlloc_v2(&d_a, size).check();
        defer _ = cuMemFree_v2(d_a);
        try cuMemAlloc_v2(&d_b, size).check();
        defer _ = cuMemFree_v2(d_b);
        try cuMemAlloc_v2(&d_c, size).check();
        defer _ = cuMemFree_v2(d_c);

        // Upload inputs
        try cuMemcpyHtoD_v2(d_a, @ptrCast(a.ptr), size).check();
        try cuMemcpyHtoD_v2(d_b, @ptrCast(b.ptr), size).check();

        // Launch kernel: pass pointers-to-args as required by the Driver API
        const block_size: c_uint = 256;
        const grid_size: c_uint = @intCast((n + block_size - 1) / block_size);
        var n_arg: c_int = @intCast(n);
        var params = [_]?*anyopaque{
            @ptrCast(&d_a),
            @ptrCast(&d_b),
            @ptrCast(&d_c),
            @ptrCast(&n_arg),
        };
        try cuLaunchKernel(
            self.sub_fn,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            null,
            &params,
            null,
        ).check();

        // Download result (implicit sync — no stream means default stream)
        try cuMemcpyDtoH_v2(@ptrCast(c.ptr), d_c, size).check();
    }
};
