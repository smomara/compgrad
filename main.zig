const std = @import("std");
const math = std.math;
const expect = std.testing.expect;

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
        left: ?*const @This(),
        right: ?*const @This(),
        aux: T,

        const Op = enum {
            None,
            Add,
            Mul,
            Pow,
            Relu,
        };

        pub fn init(data: T) @This() {
            return .{
                .data = data,
                .grad = 0,
                .op = .None,
                .left = null,
                .right = null,
                .aux = 0,
            };
        }

        pub fn add(comptime self: @This(), comptime other: @This()) @This() {
            return .{
                .data = self.data + other.data,
                .grad = 0,
                .op = .Add,
                .left = &self,
                .right = &other,
                .aux = 0,
            };
        }

        pub fn mul(comptime self: @This(), comptime other: @This()) @This() {
            return .{
                .data = self.data * other.data,
                .grad = 0,
                .op = .Mul,
                .left = &self,
                .right = &other,
                .aux = 0,
            };
        }

        pub fn pow(comptime self: @This(), comptime exponent: T) @This() {
            return .{
                .data = std.math.pow(T, self.data, exponent),
                .grad = 0,
                .op = .Pow,
                .left = &self,
                .right = null,
                .aux = exponent,
            };
        }

        pub fn neg(comptime self: @This()) @This() {
            return self.mul(init(-1));
        }

        pub fn sub(comptime self: @This(), comptime other: @This()) @This() {
            return self.add(other.neg());
        }

        pub fn div(comptime self: @This(), comptime other: @This()) @This() {
            return self.mul(other.pow(-1));
        }

        pub fn relu(comptime self: @This()) @This() {
            return .{
                .data = if (self.data > 0) self.data else 0,
                .grad = 0,
                .op = .Relu,
                .left = &self,
                .right = null,
                .aux = 0,
            };
        }

        pub fn backward(comptime self: *@This()) void {
            const nodes = get_nodes(comptime self);
            self.grad = 1;

            var i: usize = nodes.len;
            while (i > 0) : (i -= 1) {
                const v = &nodes[i - 1];
                switch (v.op) {
                    .Add => {
                        if (v.left) |left| left.grad += v.grad;
                        if (v.right) |right| right.grad += v.grad;
                    },
                    .Mul => {
                        if (v.left) |left| left.grad += v.right.?.data * v.grad;
                        if (v.right) |right| right.grad += v.left.?.data * v.grad;
                    },
                    .Pow => {
                        if (v.left) |left| left.grad += (v.aux * std.math.pow(T, v.left.?.data, v.aux - 1)) * v.grad;
                    },
                    .Relu => {
                        if (v.left) |left| left.grad += if (v.data > 0) v.grad else 0;
                    },
                    .None => {},
                }
            }
        }

        fn count_nodes(comptime root: *@This()) usize {
            var count: usize = 1;
            if (root.left) |left| count += count_nodes(left);
            if (root.right) |right| count += count_nodes(right);
            return count;
        }

        fn get_nodes(comptime root: *@This()) []@This() {
            const node_count = comptime count_nodes(root);
            var nodes: [node_count]@This() = undefined;
            var index: usize = 0;
            var stack: [node_count]*const @This() = undefined;
            var stack_top: usize = 0;

            stack[stack_top] = root;
            stack_top += 1;

            while (stack_top > 0) {
                stack_top -= 1;
                const node = stack[stack_top];

                if (node.right) |right| {
                    stack[stack_top] = right;
                    stack_top += 1;
                }
                if (node.left) |left| {
                    stack[stack_top] = left;
                    stack_top += 1;
                }

                nodes[index] = node.*;
                index += 1;
            }

            return &nodes;
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
                const b = Value(T).init(3);
                const c = a.add(b);
                try expect(c.data == 5);
                try expect(c.op == .Add);
                try expect(c.left.?.data == 2);
                try expect(c.right.?.data == 3);
            }
        }

        test "Value mul" {
            comptime {
                const a = Value(T).init(2);
                const b = Value(T).init(3);
                const c = a.mul(b);
                try expect(c.data == 6);
                try expect(c.op == .Mul);
                try expect(c.left.?.data == 2);
                try expect(c.right.?.data == 3);
            }
        }

        test "Value exp" {
            comptime {
                const a = Value(T).init(2);
                const b = a.pow(@as(T, 3));
                try expect(b.data == @as(T, 8));
                try expect(b.op == .Pow);
                try expect(b.left.?.data == 2);
                try expect(b.right == null);
            }
        }

        test "Value neg" {
            comptime {
                const a = Value(T).init(2);
                const b = a.neg();
                try expect(b.data == -2);
                try expect(b.op == .Mul);
                try expect(b.left.?.data == 2);
                try expect(b.right.?.data == -1);
            }
        }

        test "Value sub" {
            comptime {
                const a = Value(T).init(5);
                const b = Value(T).init(3);
                const c = a.sub(b);
                try expect(c.data == 2);
                try expect(c.op == .Add);
                try expect(c.left.?.data == 5);
                try expect(c.right.?.data == -3);
            }
        }

        test "Value div" {
            comptime {
                const a = Value(T).init(6);
                const b = Value(T).init(2);
                const c = a.div(b);
                try expect(c.data == 3);
                try expect(c.op == .Mul);
                try expect(c.left.?.data == 6);
                try expect(c.right.?.data == 0.5);
            }
        }

        test "Value relu" {
            comptime {
                const a = Value(T).init(-2);
                const b = Value(T).init(3);
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
            var a = Value(T).init(2);
            var b = Value(T).init(3);
            var c = a.mul(b);
            var d = c.add(b);
            var e = d.pow(2);

            e.backward();
        }
    };
}

test "Value struct" {
    comptime std.testing.refAllDecls(Value(f32));
    comptime std.testing.refAllDecls(Value(f64));
}
