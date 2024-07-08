const std = @import("std");
const tensor = @import("tensor.zig");

pub const Tensor = tensor.Tensor;

test "Run all Tensor tests" {
    std.testing.refAllDecls(tensor);
}
