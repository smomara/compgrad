const std = @import("std");
const math = std.math;
const expect = std.testing.expect;

fn ValueEngine(comptime T: type) type {
    return struct {
        const Value = struct {
            data: T,
            op: Op,
            left: ?*const Value,
            right: ?*const Value,
            aux: T,

            const Op = enum {
                None,
                Add,
                Mul,
                Pow,
                Relu,
            };

            pub fn init(comptime data: T) Value {
                return .{
                    .data = data,
                    .op = .None,
                    .left = null,
                    .right = null,
                    .aux = 0,
                };
            }

            pub fn add(comptime self: Value, comptime other: Value) Value {
                return .{
                    .data = self.data + other.data,
                    .op = .Add,
                    .left = &self,
                    .right = &other,
                    .aux = 0,
                };
            }

            pub fn mul(comptime self: Value, comptime other: Value) Value {
                return .{
                    .data = self.data * other.data,
                    .op = .Mul,
                    .left = &self,
                    .right = &other,
                    .aux = 0,
                };
            }

            pub fn pow(comptime self: Value, comptime exponent: T) Value {
                return .{
                    .data = std.math.pow(T, self.data, exponent),
                    .op = .Pow,
                    .left = &self,
                    .right = null,
                    .aux = exponent,
                };
            }

            pub fn neg(comptime self: Value) Value {
                return self.mul(init(-1));
            }

            pub fn sub(comptime self: Value, comptime other: Value) Value {
                return self.add(other.neg());
            }

            pub fn div(comptime self: Value, comptime other: Value) Value {
                return self.mul(other.pow(-1));
            }

            pub fn relu(comptime self: Value) Value {
                return .{
                    .data = if (self.data > 0) self.data else 0,
                    .op = .Relu,
                    .left = &self,
                    .right = null,
                    .aux = 0,
                };
            }
        };

        fn countNodes(comptime root: *const Value) usize {
            var count: usize = 1;
            if (root.left) |left| count += countNodes(left);
            if (root.right) |right| count += countNodes(right);
            return count;
        }

        fn GradientResult(comptime root: Value) type {
            const node_count = countNodes(&root);
            return struct {
                values: [node_count]Value,
                grads: [node_count]T,

                pub fn getGrad(comptime self: GradientResult(root), comptime value: Value) T {
                    inline for (self.values, self.grads) |v, grad| {
                        if (std.meta.eql(&v, &value)) {
                            return grad;
                        }
                    }
                    unreachable;
                }
            };
        }

        pub fn backward(comptime root: Value) GradientResult(root) {
            var result: GradientResult(root) = undefined;
            comptime var index: usize = 0;

            comptime populateValues(&root, &result.values, &index);
            comptime computeGradients(&result.values, &result.grads);

            return result;
        }

        fn populateValues(comptime node: *const Value, comptime values: []Value, comptime index: *usize) void {
            values[index.*] = node.*;
            index.* += 1;

            if (node.left) |left| comptime populateValues(left, values, index);
            if (node.right) |right| comptime populateValues(right, values, index);
        }

        fn computeGradients(comptime values: []Value, comptime grads: []T) void {
            inline for (grads) |*grad| {
                grad.* = 0;
            }
            grads[grads.len - 1] = 1; // Set gradient of the root node

            comptime var i: usize = values.len;
            inline while (i > 0) : (i -= 1) {
                const v = &values[i - 1];
                const grad = grads[i - 1];

                switch (v.op) {
                    .Add => {
                        if (v.left) |left| {
                            const left_index = comptime findIndex(values, left);
                            grads[left_index] += grad;
                        }
                        if (v.right) |right| {
                            const right_index = comptime findIndex(values, right);
                            grads[right_index] += grad;
                        }
                    },
                    .Mul => {
                        if (v.left) |left| {
                            const left_index = comptime findIndex(values, left);
                            grads[left_index] += v.right.?.data * grad;
                        }
                        if (v.right) |right| {
                            const right_index = comptime findIndex(values, right);
                            grads[right_index] += v.left.?.data * grad;
                        }
                    },
                    .Pow => {
                        if (v.left) |left| {
                            const left_index = comptime findIndex(values, left);
                            grads[left_index] += (v.aux * std.math.pow(T, left.data, v.aux - 1)) * grad;
                        }
                    },
                    .Relu => {
                        if (v.left) |left| {
                            const left_index = comptime findIndex(values, left);
                            grads[left_index] += if (v.data > 0) grad else 0;
                        }
                    },
                    .None => {},
                }
            }
        }

        fn findIndex(comptime values: []const Value, comptime node: *const Value) usize {
            inline for (values, 0..) |v, i| {
                if (std.meta.eql(&v, node)) {
                    return i;
                }
            }
            unreachable;
        }

        test "Value init" {
            comptime {
                const v = Value.init(5);
                try expect(v.data == 5);
                try expect(v.op == .None);
                try expect(v.left == null);
                try expect(v.right == null);
                try expect(v.aux == 0);
            }
        }

        test "Value add" {
            comptime {
                const a = Value.init(2);
                const b = Value.init(3);
                const c = a.add(b);
                try expect(c.data == 5);
                try expect(c.op == .Add);
                try expect(c.left.?.data == 2);
                try expect(c.right.?.data == 3);
            }
        }

        test "Value mul" {
            comptime {
                const a = Value.init(2);
                const b = Value.init(3);
                const c = a.mul(b);
                try expect(c.data == 6);
                try expect(c.op == .Mul);
                try expect(c.left.?.data == 2);
                try expect(c.right.?.data == 3);
            }
        }

        test "Value pow" {
            comptime {
                const a = Value.init(2);
                const b = a.pow(3);
                try expect(b.data == 8);
                try expect(b.op == .Pow);
                try expect(b.left.?.data == 2);
                try expect(b.right == null);
                try expect(b.aux == 3);
            }
        }

        test "backward" {
            comptime {
                const a = Value.init(2);
                const b = Value.init(3);
                const c = a.mul(b);
                const d = c.add(b);
                const e = d.pow(2);

                const result = backward(e);

                @compileLog(result.getGrad(a));
                @compileLog(result.getGrad(b));
                @compileLog(result.getGrad(c));
                @compileLog(result.getGrad(d));
                @compileLog(result.getGrad(d));
            }
        }
    };
}

test "Value struct" {
    inline for (.{ f32, f64 }) |F| {
        const E = ValueEngine(F);
        comptime std.testing.refAllDecls(E);
    }
}
