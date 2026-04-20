const std = @import("std");
const cuda = @import("zig_cuda");

pub fn main() !void {
    const n = 1024;
    var a: [n]f32 = undefined;
    var b: [n]f32 = undefined;
    var c: [n]f32 = undefined;

    // a[i] = i, b[i] = n - i, so c[i] should equal n for all i
    for (0..n) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(n - i);
    }

    try cuda.add(&a, &b, &c);

    // Verify results
    var ok = true;
    for (c) |v| {
        if (v != @as(f32, n)) {
            ok = false;
            break;
        }
    }

    if (ok) {
        std.debug.print("CUDA add kernel ran successfully! c[0..4] = {d:.0} {d:.0} {d:.0} {d:.0}\n", .{ c[0], c[1], c[2], c[3] });
    } else {
        std.debug.print("Result mismatch!\n", .{});
        return error.WrongResult;
    }
}
