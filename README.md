# CompGrad

CompGrad is a lightweight, high-performance tensor library implemented in Zig. It leverages Zig's powerful comptime features to provide efficient tensor operations with automatic differentiation, all known at compile-time. Say goodbye to pesky runtime errors! (but get used to compiler errors)

## Features

- **Compile-time Tensor Operations**: All tensor operations and gradients are computed at compile-time, leading to highly optimized runtime performance.
- **Automatic Differentiation**: Built-in support for automatic differentiation (autograd) for all operations.
- **Vectorization**: Utilizes Zig's vector types for efficient SIMD operations out of the box.
- **Generic Over Float Types**: Supports f16, f32, f64, and f128 float types.
- **Basic Tensor Operations**: Addition, subtraction, multiplication, division, power, and ReLU activation.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/smomara/compgrad.git
    ```
2. Build the library:
    ```
    cd compgrad
    zig build
    ```

This will generate a `libtensor.a` static library in the `lib/` directory.

## Usage

Here's a simple exapmle of using CompGrad:

```zig
const std = @import("std");
const Tensor = @import("tensor").Tensor;

pub fn main() !void {
    var a = Tensor(f32, 4).init([_]f32{ 1, 2, 3, 4 });
    var b = Tensor(f32, 4).init(2);
    var c = a.mul(&b);
    c.backward();

    std.debug.print("Result: {any}\n", .{c.data});
    std.debug.print("Gradient of a: {any}\n", .{a.grad});
    std.debug.print("Gradient of b: {any}\n", .{
}
```

To use CompGrad in your project:

1. Copy `libtensor.a` to your project's lib directory.
2. Copy `tensor.zig` to your project's source directory.
3. In your `build.zig`, add:
   ```zig
   exe.addObjectFile(.{ .path = "lib/libtensor.a" });
   exe.addIncludePath(.{ .path = "src" });
   ```

## Roadmap

- Expand operation set to support more deep learning operations (convolutions, pooling, etc.)
- Implement common activation functions (sigmoid, tanh, etc.)
- Add support for higher-dimensional tensors
- Implement basic neural network layers (Linear, Conv2D, etc.)
- Optimize memory usage and performance

## Inspiration

CompGrad is inspired by:
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [tinygrad](https://github.com/tinygrad/tinygrad) by George Hotz

I aim to bring the simplicity and educational value of these projects to the Zig ecosystem, while leveraging Zig's unique features for performance and safety.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

Do literally whatever you want to this code.
