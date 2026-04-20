const std = @import("std");
const cuda = @import("zig_cuda");

pub fn main(init: std.process.Init) !void {
    const n = 1024 * 1024;
    // Three 4 MB arrays would overflow the default 8 MB stack; use the heap.
    var arena = init.arena.allocator();

    const a = try arena.alloc(f32, n);
    const b = try arena.alloc(f32, n);
    const c = try arena.alloc(f32, n);

    // a[i] = i, b[i] = n - i  =>  c[i] should equal n for every i
    for (0..n) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(n - i);
    }

    const ctx = try cuda.Context.init();
    defer ctx.deinit();

    try ctx.add(a, b, c);

    // Verify all results
    for (c) |v| {
        if (v != @as(f32, n)) {
            std.debug.print("Result mismatch: got {d}\n", .{v});
            return error.WrongResult;
        }
    }

    std.debug.print("CUDA add kernel ran successfully! n={d}, c[0..4] = {d:.0} {d:.0} {d:.0} {d:.0}\n", .{ n, c[0], c[1], c[2], c[3] });
}
