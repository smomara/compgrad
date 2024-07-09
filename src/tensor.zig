const std = @import("std");

pub const Layout = struct {
    total_size: usize = 0,

    pub fn Tensor(comptime layout: *Layout, comptime size: usize) type {
        const tensor_offset = layout.total_size;
        layout.total_size += size;

        return struct {
            const VecT = @Vector(size, f32);
            pub const offset: usize = tensor_offset;

            values: VecT,

            pub fn init(comptime values: [size]f32) @This() {
                return .{ .values = values };
            }

            pub fn add(comptime self: @This(), comptime other: anytype) @This() {
                if (@typeName(@TypeOf(other)) != @typeName(@This())) @compileError("must add two tensors");
                return .{ .values = self.values + other.values };
            }

            pub fn mul(comptime self: @This(), comptime other: anytype) @This() {
                if (@typeName(@TypeOf(other)) != @typeName(@This())) @compileError("must mul two tensors");
                return .{ .values = self.values + other.values };
            }
        };
    }
};

test "vectors" {
    comptime {
        var layout: Layout = Layout{};
        const t1: type = layout.Tensor(4);
        const t2: type = layout.Tensor(4);

        if (t1.offset != 0) @compileError("v1 offset != 0");
        if (t2.offset != 4) @compileError("v2 offset != 4");

        const a = t1.init([_]f32{ 1, 2, 3, 4 });
        const b = t2.init([_]f32{ 5, 6, 7, 8 });

        const c = a.add(b);
        if (!std.mem.eql(f32, &@as([4]f32, c.values), &[_]f32{ 6, 8, 10, 12 })) @compileError("a + b != c");
    }
}
