const std = @import("std");
const math = std.math;
const expect = std.testing.expect;
const assert = std.debug.assert;

fn Value(comptime T: type) type {
    comptime {
        switch (T) {
            f32 => {},
            f64 => {},
            else => @compileError("Value must be instantiated with a float type"),
        }
    }

    return struct {
        data: T,
        grad: T,
        op: Op,
        left: ?*@This(),
        right: ?*@This(),
        aux: T,

        const Op = enum {
            None,
            Add,
            Mul,
            Pow,
            Relu,
        };

        pub fn init(data: T) Value(T) {
            return .{
                .data = data,
                .grad = 0,
                .op = .None,
                .left = null,
                .right = null,
                .aux = 0,
            };
        }

        pub fn add(comptime self: *Value(T), comptime other: *Value(T)) Value(T) {
            return .{
                .data = self.data + other.data,
                .grad = 0,
                .op = .Add,
                .left = self,
                .right = other,
                .aux = 0,
            };
        }

        pub fn mul(comptime self: *Value(T), comptime other: *Value(T)) Value(T) {
            return .{
                .data = self.data * other.data,
                .grad = 0,
                .op = .Mul,
                .left = self,
                .right = other,
                .aux = 0,
            };
        }

        pub fn pow(comptime self: *Value(T), comptime exponent: T) Value(T) {
            return .{
                .data = std.math.pow(T, self.data, exponent),
                .grad = 0,
                .op = .Pow,
                .left = self,
                .right = null,
                .aux = exponent,
            };
        }

        pub fn neg(comptime self: *Value(T)) Value(T) {
            var one_neg = init(-1);
            return self.mul(&one_neg);
        }

        pub fn sub(comptime self: *Value(T), comptime other: *Value(T)) Value(T) {
            var other_neg = other.neg();
            return self.add(&other_neg);
        }

        pub fn div(comptime self: *Value(T), comptime other: *Value(T)) Value(T) {
            var other_inv = other.pow(-1);
            return self.mul(&other_inv);
        }

        pub fn relu(comptime self: *Value(T)) Value(T) {
            return .{
                .data = if (self.data > 0) self.data else 0,
                .grad = 0,
                .op = .Relu,
                .left = self,
                .right = null,
                .aux = 0,
            };
        }

        pub fn backward(comptime self: *Value(T)) void {
            self.grad = 1;
            backprop(self);
        }

        fn backprop(comptime node: *Value(T)) void {
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
                            backprop(left);
                        }
                    }
                    if (node.right) |right| {
                        if (node.left) |left| {
                            right.grad += left.data * node.grad;
                            backprop(right);
                        }
                    }
                },
                .Pow => {
                    if (node.left) |left| {
                        left.grad += (node.aux * std.math.pow(T, left.data, node.aux - 1)) * node.grad;
                        backprop(left);
                    }
                },
                .Relu => {
                    if (node.left) |left| {
                        left.grad += if (node.data > 0) node.grad else 0;
                        backprop(left);
                    }
                },
                .None => {},
            }
        }

        test "Value init" {
            comptime {
                const v = Value(T).init(5);
                try expect(v.data == 5);
                try expect(v.grad == 0);
                try expect(v.op == .None);
                try expect(v.left == null);
                try expect(v.right == null);
                try expect(v.aux == 0);
            }
        }

        test "Value add" {
            comptime {
                var a = Value(T).init(2);
                var b = Value(T).init(3);
                const c = a.add(&b);
                try expect(c.data == 5);
                try expect(c.op == .Add);
                try expect(c.left.?.data == 2);
                try expect(c.right.?.data == 3);
            }
        }

        test "Value mul" {
            comptime {
                var a = Value(T).init(2);
                var b = Value(T).init(3);
                const c = a.mul(&b);
                try expect(c.data == 6);
                try expect(c.op == .Mul);
                try expect(c.left.?.data == 2);
                try expect(c.right.?.data == 3);
            }
        }

        test "Value exp" {
            comptime {
                var a = Value(T).init(2);
                const b = a.pow(@as(T, 3));
                try expect(b.data == @as(T, 8));
                try expect(b.op == .Pow);
                try expect(b.left.?.data == 2);
                try expect(b.right == null);
            }
        }

        test "Value neg" {
            comptime {
                var a = Value(T).init(2);
                const b = a.neg();
                try expect(b.data == -2);
                try expect(b.op == .Mul);
                try expect(b.left.?.data == 2);
                try expect(b.right.?.data == -1);
            }
        }

        test "Value sub" {
            comptime {
                var a = Value(T).init(5);
                var b = Value(T).init(3);
                const c = a.sub(&b);
                try expect(c.data == 2);
                try expect(c.op == .Add);
                try expect(c.left.?.data == 5);
                try expect(c.right.?.data == -3);
            }
        }

        test "Value div" {
            comptime {
                var a = Value(T).init(6);
                var b = Value(T).init(2);
                const c = a.div(&b);
                try expect(c.data == 3);
                try expect(c.op == .Mul);
                try expect(c.left.?.data == 6);
                try expect(c.right.?.data == 0.5);
            }
        }

        test "Value relu" {
            comptime {
                var a = Value(T).init(-2);
                var b = Value(T).init(3);
                const c = a.relu();
                const d = b.relu();
                try expect(c.data == 0);
                try expect(d.data == 3);
                try expect(c.op == .Relu);
                try expect(d.op == .Relu);
                try expect(c.left.?.data == -2);
                try expect(d.left.?.data == 3);
            }
        }

        test "Value backward" {
            comptime {
                var a = Value(T).init(2);
                var b = Value(T).init(3);
                var c = a.mul(&b);
                var d = c.add(&b);
                var e = d.pow(2);

                e.backward();

                // expected values obtained using pytorch
                //
                //     import torch
                //
                //     torch.set_grad_enabled(True)
                //
                //     a = torch.tensor(2.0, requires_grad=True)
                //     b = torch.tensor(3.0, requires_grad=True)
                //
                //     c = a * b
                //     c.retain_grad()
                //
                //     d = c + b
                //     d.retain_grad()
                //
                //     e = d ** 2
                //     e.retain_grad()
                //
                //     e.backward()
                //
                //     print(a.grad.item()) # 54
                //     print(b.grad.item()) # 54
                //     print(c.grad.item()) # 18
                //     print(d.grad.item()) # 18
                //     print(e.grad.item()) # 1

                try expect(a.grad == 54);
                try expect(b.grad == 54);
                try expect(c.grad == 18);
                try expect(d.grad == 18);
                try expect(e.grad == 1);
            }
        }
    };
}

test "Value struct" {
    comptime std.testing.refAllDecls(Value(f32));
    comptime std.testing.refAllDecls(Value(f64));
}
