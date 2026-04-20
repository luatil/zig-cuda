const std = @import("std");

extern fn cuda_add(h_a: [*]const f32, h_b: [*]const f32, h_c: [*]f32, n: c_int) c_int;

/// Adds two slices element-wise on the GPU. All slices must have the same length.
pub fn add(a: []const f32, b: []const f32, c: []f32) !void {
    std.debug.assert(a.len == b.len and b.len == c.len);
    const err = cuda_add(a.ptr, b.ptr, c.ptr, @intCast(a.len));
    if (err != 0) {
        std.debug.print("CUDA error code: {d}\n", .{err});
        return error.CudaError;
    }
}
