const std = @import("std");
const math = std.math;
const expect = std.testing.expect;
const assert = std.debug.assert;

fn Tensor(comptime T: type, comptime len: comptime_int) type {
    comptime {
        switch (T) {
            f32 => {},
            f64 => {},
            else => @compileError("Value must be instantiated with a float type"),
        }
    }

    const VecT = @Vector(len, T);

    return struct {
        data: VecT,
        grad: VecT,
        op: Op,
        left: ?*Tensor(T, len),
        right: ?*Tensor(T, len),
        aux: VecT,

        const Op = enum {
            None,
            Add,
            Mul,
            Pow,
            Relu,
        };

        const vector_zeroes: VecT = @splat(0);
        const vector_ones: VecT = @splat(1);
        const vector_neg_ones: VecT = @splat(-1);

        pub fn init(data: anytype) Tensor(T, len) {
            const DataT = @TypeOf(data);
            const vec = switch (@typeInfo(DataT)) {
                .Float, .ComptimeFloat, .Int, .ComptimeInt => @as(VecT, @splat(data)),
                .Array => |arr_info| if (arr_info.len == len) @as(VecT, data) else {
                    @compileError("Array length must match Tensor length");
                },
                .Vector => |vec_info| if (vec_info.len == len) data else {
                    @compileError("Vector length must match Tensor length");
                },
                else => @compileError("Unsupported data type"),
            };

            return .{
                .data = vec,
                .grad = vector_zeroes,
                .op = .None,
                .left = null,
                .right = null,
                .aux = vector_zeroes,
            };
        }

        pub fn add(comptime self: *Tensor(T, len), comptime other: anytype) Tensor(T, len) {
            const OtherT = @TypeOf(other);
            const ResultT = struct { data: VecT, right: ?*Tensor(T, len) };
            var result: ResultT = undefined;

            switch (@typeInfo(OtherT)) {
                .Pointer => |ptr_info| switch (ptr_info.child) {
                    Tensor(T, len) => {
                        result = .{ .data = self.data + other.data, .right = other };
                    },
                    else => @compileError("Unsupported type for addition"),
                },
                .Float, .ComptimeFloat, .Int, .ComptimeInt => {
                    var new_tensor = init(other);
                    result = .{
                        .data = self.data + @as(VecT, @splat(other)),
                        .right = &new_tensor,
                    };
                },
                .Array => |arr_info| if (arr_info.len == len) {
                    var new_tensor = init(other);
                    result = .{
                        .data = self.data + @as(VecT, other),
                        .right = &new_tensor,
                    };
                } else {
                    @compileError("Array length must match Tensor length");
                },
                .Vector => |vec_info| if (vec_info.len == len) {
                    var new_tensor = init(other);
                    result = .{
                        .data = self.data + other,
                        .right = &new_tensor,
                    };
                } else {
                    @compileError("Vector length must match Tensor length");
                },
                else => @compileError("Unsupported type for addition"),
            }

            return .{
                .data = result.data,
                .grad = vector_zeroes,
                .op = .Add,
                .left = self,
                .right = result.right,
                .aux = vector_zeroes,
            };
        }

        pub fn mul(comptime self: *Tensor(T, len), comptime other: anytype) Tensor(T, len) {
            const OtherT = @TypeOf(other);
            const ResultT = struct { data: VecT, right: ?*Tensor(T, len) };
            var result: ResultT = undefined;

            switch (@typeInfo(OtherT)) {
                .Pointer => |ptr_info| switch (ptr_info.child) {
                    Tensor(T, len) => {
                        result = .{ .data = self.data * other.data, .right = other };
                    },
                    else => @compileError("Unsupported type for multiplication"),
                },
                .Float, .ComptimeFloat, .Int, .ComptimeInt => {
                    var new_tensor = init(other);
                    result = .{
                        .data = self.data * @as(VecT, @splat(other)),
                        .right = &new_tensor,
                    };
                },
                .Array => |arr_info| if (arr_info.len == len) {
                    var new_tensor = init(other);
                    result = .{
                        .data = self.data * @as(VecT, other),
                        .right = &new_tensor,
                    };
                } else {
                    @compileError("Array length must match Tensor length");
                },
                .Vector => |vec_info| if (vec_info.len == len) {
                    var new_tensor = init(other);
                    result = .{
                        .data = self.data * other,
                        .right = &new_tensor,
                    };
                } else {
                    @compileError("Vector length must match Tensor length");
                },
                else => @compileError("Unsupported type for multiplication"),
            }

            return .{
                .data = result.data,
                .grad = vector_zeroes,
                .op = .Mul,
                .left = self,
                .right = result.right,
                .aux = vector_zeroes,
            };
        }

        pub fn pow(comptime self: *Tensor(T, len), comptime exponent: anytype) Tensor(T, len) {
            const ExponentT = @TypeOf(exponent);
            const exp_vec = switch (@typeInfo(ExponentT)) {
                .Pointer => |ptr_info| switch (ptr_info.child) {
                    Tensor(T, len) => exponent.data,
                    else => @compileError("Unsupported type for exponent"),
                },
                .Float, .ComptimeFloat, .Int, .ComptimeInt => @as(VecT, @splat(exponent)),
                .Array => |arr_info| if (arr_info.len == len) @as(VecT, exponent) else {
                    @compileError("Array length must match Tensor length");
                },
                .Vector => |vec_info| if (vec_info.len == len) exponent else {
                    @compileError("Vector length must match Tensor length");
                },
                else => @compileError("Unsupported type for exponent"),
            };

            const result = @exp2(exp_vec * @log2(self.data));
            return .{
                .data = result,
                .grad = vector_zeroes,
                .op = .Pow,
                .left = self,
                .right = null,
                .aux = exp_vec,
            };
        }

        pub fn neg(comptime self: *Tensor(T, len)) Tensor(T, len) {
            var tensor_neg_ones = init(vector_neg_ones);
            return self.mul(&tensor_neg_ones);
        }

        pub fn sub(comptime self: *Tensor(T, len), comptime other: anytype) Tensor(T, len) {
            const OtherT = @TypeOf(other);
            const ResultT = struct { data: VecT, right: ?*Tensor(T, len) };
            var result: ResultT = undefined;

            switch (@typeInfo(OtherT)) {
                .Pointer => |ptr_info| switch (ptr_info.child) {
                    Tensor(T, len) => {
                        var other_neg = other.neg();
                        result = .{ .data = self.data + other_neg.data, .right = &other_neg };
                    },
                    else => @compileError("Unsupported type for subtraction"),
                },
                .Float, .ComptimeFloat, .Int, .ComptimeInt => {
                    var new_tensor = init(other);
                    var other_neg = new_tensor.neg();
                    result = .{
                        .data = self.data + other_neg.data,
                        .right = &other_neg,
                    };
                },
                .Array => |arr_info| if (arr_info.len == len) {
                    var new_tensor = init(other);
                    var other_neg = new_tensor.neg();
                    result = .{
                        .data = self.data + other_neg.data,
                        .right = &other_neg,
                    };
                } else {
                    @compileError("Array length must match Tensor length");
                },
                .Vector => |vec_info| if (vec_info.len == len) {
                    var new_tensor = init(other);
                    var other_neg = new_tensor.neg();
                    result = .{
                        .data = self.data + other_neg.data,
                        .right = &other_neg,
                    };
                } else {
                    @compileError("Vector length must match Tensor length");
                },
                else => @compileError("Unsupported type for subtraction"),
            }

            return .{
                .data = result.data,
                .grad = vector_zeroes,
                .op = .Add, // Subtraction is implemented as addition with negation
                .left = self,
                .right = result.right,
                .aux = vector_zeroes,
            };
        }

        pub fn div(comptime self: *Tensor(T, len), comptime other: anytype) Tensor(T, len) {
            const OtherT = @TypeOf(other);
            const ResultT = struct { data: VecT, right: ?*Tensor(T, len) };
            var result: ResultT = undefined;

            switch (@typeInfo(OtherT)) {
                .Pointer => |ptr_info| switch (ptr_info.child) {
                    Tensor(T, len) => {
                        var other_inv = other.pow(-1);
                        result = .{ .data = self.data * other_inv.data, .right = &other_inv };
                    },
                    else => @compileError("Unsupported type for division"),
                },
                .Float, .ComptimeFloat, .Int, .ComptimeInt => {
                    var new_tensor = init(other);
                    var other_inv = new_tensor.pow(-1);
                    result = .{
                        .data = self.data * other_inv.data,
                        .right = &other_inv,
                    };
                },
                .Array => |arr_info| if (arr_info.len == len) {
                    var new_tensor = init(other);
                    var other_inv = new_tensor.pow(-1);
                    result = .{
                        .data = self.data * other_inv.data,
                        .right = &other_inv,
                    };
                } else {
                    @compileError("Array length must match Tensor length");
                },
                .Vector => |vec_info| if (vec_info.len == len) {
                    var new_tensor = init(other);
                    var other_inv = new_tensor.pow(-1);
                    result = .{
                        .data = self.data * other_inv.data,
                        .right = &other_inv,
                    };
                } else {
                    @compileError("Vector length must match Tensor length");
                },
                else => @compileError("Unsupported type for division"),
            }

            return .{
                .data = result.data,
                .grad = vector_zeroes,
                .op = .Mul, // Division is implemented as multiplication with inverse
                .left = self,
                .right = result.right,
                .aux = vector_zeroes,
            };
        }

        pub fn relu(comptime self: *Tensor(T, len)) Tensor(T, len) {
            const zero_vector: VecT = @splat(0);
            return .{
                .data = @max(self.data, zero_vector),
                .grad = vector_zeroes,
                .op = .Relu,
                .left = self,
                .right = null,
                .aux = vector_zeroes,
            };
        }

        pub fn backward(comptime self: *Tensor(T, len)) void {
            self.grad = vector_ones;
            backprop(self);
        }

        fn backprop(comptime node: *Tensor(T, len)) void {
            switch (node.op) {
                .Add => {
                    if (node.left) |left| {
                        left.grad += node.grad;
                        backprop(left);
                    }
                    if (node.right) |right| {
                        right.grad += node.grad;
                        backprop(right);
                    }
                },
                .Mul => {
                    if (node.left) |left| {
                        if (node.right) |right| {
                            left.grad += right.data * node.grad;
                            right.grad += left.data * node.grad;
                            backprop(left);
                            backprop(right);
                        }
                    }
                },
                .Pow => {
                    if (node.left) |left| {
                        const ones_vector: VecT = @splat(1);
                        left.grad += node.aux * @exp2((node.aux - ones_vector) * @log2(left.data)) * node.grad;
                        backprop(left);
                    }
                },
                .Relu => {
                    if (node.left) |left| {
                        const zeroes_vector: VecT = @splat(0);
                        left.grad += @select(T, node.data > zeroes_vector, node.grad, zeroes_vector);
                        backprop(left);
                    }
                },
                .None => {},
            }
        }

        test "Tensor init" {
            comptime {
                // Test init with scalar values
                const t1 = Tensor(T, len).init(2);
                try expect(@reduce(.And, t1.data == @as(VecT, @splat(2))));

                const t2 = Tensor(T, len).init(2.5);
                try expect(@reduce(.And, t2.data == @as(VecT, @splat(2.5))));

                // Test init with arrays
                const t3 = Tensor(T, len).init([_]T{ 1, 2, 3, 4 });
                try expect(@reduce(.And, t3.data == @Vector(4, T){ 1, 2, 3, 4 }));

                // Test init with vectors
                const v: VecT = @Vector(4, T){ 5, 6, 7, 8 };
                const t4 = Tensor(T, len).init(v);
                try expect(@reduce(.And, t4.data == v));
            }
        }

        test "Tensor add" {
            comptime {
                var a = Tensor(T, len).init(2);

                // Test tensor + tensor
                var b = Tensor(T, len).init(3);
                var c = a.add(&b);
                try expect(@reduce(.And, c.data == @as(VecT, @splat(5))));

                // Test tensor + scalar
                c = a.add(3);
                try expect(@reduce(.And, c.data == @as(VecT, @splat(5))));

                // Test tensor + array
                c = a.add([_]T{ 1, 2, 3, 4 });
                try expect(@reduce(.And, c.data == @Vector(4, T){ 3, 4, 5, 6 }));

                // Test tensor + vector
                const v: VecT = @Vector(4, T){ 1, 2, 3, 4 };
                c = a.add(v);
                try expect(@reduce(.And, c.data == @Vector(4, T){ 3, 4, 5, 6 }));
            }
        }

        test "Tensor mul" {
            comptime {
                var a = Tensor(T, len).init(2);

                // Test tensor * tensor
                var b = Tensor(T, len).init(3);
                var c = a.mul(&b);
                try expect(@reduce(.And, c.data == @as(VecT, @splat(6))));

                // Test tensor * scalar
                c = a.mul(3);
                try expect(@reduce(.And, c.data == @as(VecT, @splat(6))));

                // Test tensor * array
                c = a.mul([_]T{ 1, 2, 3, 4 });
                try expect(@reduce(.And, c.data == @Vector(4, T){ 2, 4, 6, 8 }));

                // Test tensor * vector
                const v: VecT = @Vector(4, T){ 1, 2, 3, 4 };
                c = a.mul(v);
                try expect(@reduce(.And, c.data == @Vector(4, T){ 2, 4, 6, 8 }));
            }
        }

        test "Tensor pow" {
            comptime {
                var a = Tensor(T, len).init(2);

                // Test tensor ^ scalar
                var b = a.pow(3);
                try expect(@reduce(.And, @abs(b.data - @as(VecT, @splat(8))) < @as(VecT, @splat(1e-6))));

                // Test tensor ^ array
                b = a.pow([_]T{ 1, 2, 3, 4 });
                try expect(@reduce(.And, @abs(b.data - @Vector(4, T){ 2, 4, 8, 16 }) < @as(VecT, @splat(1e-6))));

                // Test tensor ^ vector
                const v: VecT = @Vector(4, T){ 1, 2, 3, 4 };
                b = a.pow(v);
                try expect(@reduce(.And, @abs(b.data - @Vector(4, T){ 2, 4, 8, 16 }) < @as(VecT, @splat(1e-6))));

                // Test tensor ^ tensor
                var c = Tensor(T, len).init([_]T{ 1, 2, 3, 4 });
                b = a.pow(&c);
                try expect(@reduce(.And, @abs(b.data - @Vector(4, T){ 2, 4, 8, 16 }) < @as(VecT, @splat(1e-6))));
            }
        }

        test "Tensor neg" {
            comptime {
                var a = Tensor(T, len).init(2);
                const b = a.neg();
                try expect(@reduce(.And, b.data == @as(VecT, @splat(-2))));
            }
        }

        test "Tensor sub" {
            comptime {
                var a = Tensor(T, len).init(5);

                // Test tensor - tensor
                var b = Tensor(T, len).init(3);
                var c = a.sub(&b);
                try expect(@reduce(.And, c.data == @as(VecT, @splat(2))));

                // Test tensor - scalar
                c = a.sub(3);
                try expect(@reduce(.And, c.data == @as(VecT, @splat(2))));

                // Test tensor - array
                c = a.sub([_]T{ 1, 2, 3, 4 });
                try expect(@reduce(.And, c.data == @Vector(4, T){ 4, 3, 2, 1 }));

                // Test tensor - vector
                const v: VecT = @Vector(4, T){ 1, 2, 3, 4 };
                c = a.sub(v);
                try expect(@reduce(.And, c.data == @Vector(4, T){ 4, 3, 2, 1 }));
            }
        }

        test "Tensor div" {
            comptime {
                var a = Tensor(T, len).init(6);

                // Test tensor / tensor
                var b = Tensor(T, len).init(2);
                var c = a.div(&b);
                try expect(@reduce(.And, @abs(c.data - @as(VecT, @splat(3))) < @as(VecT, @splat(1e-6))));

                // Test tensor / scalar
                c = a.div(2);
                try expect(@reduce(.And, @abs(c.data - @as(VecT, @splat(3))) < @as(VecT, @splat(1e-6))));

                // Test tensor / array
                c = a.div([_]T{ 1, 2, 3, 4 });
                try expect(@reduce(.And, @abs(c.data - @Vector(4, T){ 6, 3, 2, 1.5 }) < @as(VecT, @splat(1e-6))));

                // Test tensor / vector
                const v: VecT = @Vector(4, T){ 1, 2, 3, 4 };
                c = a.div(v);
                try expect(@reduce(.And, @abs(c.data - @Vector(4, T){ 6, 3, 2, 1.5 }) < @as(VecT, @splat(1e-6))));
            }
        }

        test "Tensor relu" {
            comptime {
                var a = Tensor(T, len).init(@Vector(4, T){ -2, -1, 0, 1 });
                const b = a.relu();
                try expect(@reduce(.And, b.data == @Vector(4, T){ 0, 0, 0, 1 }));
            }
        }

        test "Tensor backward" {
            comptime {
                var a = Tensor(T, len).init(2);
                var b = Tensor(T, len).init(3);
                var c = a.mul(&b);
                var d = c.add(&b);
                var e = d.pow(2);

                e.backward();

                try expect(@reduce(.And, @abs(a.grad - @as(VecT, @splat(54))) < @as(VecT, @splat(1e-6))));
                try expect(@reduce(.And, @abs(b.grad - @as(VecT, @splat(54))) < @as(VecT, @splat(1e-6))));
                try expect(@reduce(.And, @abs(c.grad - @as(VecT, @splat(18))) < @as(VecT, @splat(1e-6))));
                try expect(@reduce(.And, @abs(d.grad - @as(VecT, @splat(18))) < @as(VecT, @splat(1e-6))));
                try expect(@reduce(.And, @abs(e.grad - @as(VecT, @splat(1))) < @as(VecT, @splat(1e-6))));
            }
        }
    };
}

test "Value struct" {
    comptime std.testing.refAllDecls(Tensor(f32, 4));
    comptime std.testing.refAllDecls(Tensor(f64, 4));
}
