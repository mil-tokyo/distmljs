# Generates shader/webgpu/autogen/binary_*.glsl

import os

package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dst_dir = os.path.join(package_root, "shader/webgpu/autogen")
os.makedirs(dst_dir, exist_ok=True)


DTYPE_TO_SCALAR_TYPE = {
    "float32": "float",
    "int32": "int",
    "uint8": "uint",
    "bool": "uint",
}

D_F = ["float32"]
D_FI = ["float32", "int32"]
D_FIU = ["float32", "int32", "uint8"]
D_A = ["float32", "int32", "uint8", "bool"]


def generateDecomposeDim(dim) -> str:
    source = "uint dec_tmp = i;\n"
    for d in range(dim-1, -1, -1):
        source += f"uint dim{d} = dec_tmp % outShape{d};\n"
        if d > 0:
            source += f"dec_tmp = dec_tmp / outShape{d};\n"
    return source

# about pow
# pow(-1.5, 2) cases error in GLSL, but it is useful in normalization algorithm.
# implementation: pow(abs(-1.5), 2)


for name, op, supported_types in sorted([
    ["add", "lhs + rhs", D_FIU],
    ["sub", "lhs - rhs", D_FIU],
    ["mul", "lhs * rhs", D_FIU],
    ["div", "lhs / rhs", D_FIU],
    ["pow", "pow(abs(lhs), rhs)", D_F],
    ["sigmoidBackprop", "(1.0 - lhs) * lhs * rhs", D_F],
    ["reluBackprop", "lhs > 0.0 ? rhs : 0.0", D_F],
]):
    for t in supported_types:
        for dim in range(8):
            kernel_name = f"binary_{name}_{t}_{dim}"

            scalar_type = DTYPE_TO_SCALAR_TYPE[t]
            if dim > 0:
                source = f"""#version 450

    // Auto-generated by tools/generate_webgputensor_glsl_binary_op.py

    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

    layout(std430, set = 0, binding = 0) readonly buffer arrayLhs {{
    {scalar_type} numbers[];
    }} array_lhs;

    layout(std430, set = 0, binding = 1) readonly buffer arrayRhs {{
    {scalar_type} numbers[];
    }} array_rhs;

    layout(std430, set = 0, binding = 2) buffer arrayC {{
    {scalar_type} numbers[];
    }} array_c;

    layout(std430, set = 0, binding = 3) readonly buffer Meta {{
    uint len;
    uint {",".join([f"outShape{d}" for d in range(dim)])};
    uint {",".join([f"lhsStride{d}" for d in range(dim)])};
    uint {",".join([f"rhsStride{d}" for d in range(dim)])};
    }} meta;

    void main() {{
    uint len = meta.len;
    uint {",".join([f"outShape{d}=meta.outShape{d}" for d in range(dim)])};
    uint {",".join([f"lhsStride{d}=meta.lhsStride{d}" for d in range(dim)])};
    uint {",".join([f"rhsStride{d}=meta.rhsStride{d}" for d in range(dim)])};
    for (uint i = gl_GlobalInvocationID.x; i < len; i += 4096) {{
        {generateDecomposeDim(dim)}

        {scalar_type} lhs = array_lhs.numbers[{"+".join([f"dim{d}*lhsStride{d}" for d in range(dim)])}];
        {scalar_type} rhs = array_rhs.numbers[{"+".join([f"dim{d}*rhsStride{d}" for d in range(dim)])}];
        array_c.numbers[i] = {op};
    }}
    }}

    """
            else:

                source = f"""#version 450

    // Auto-generated by tools/generate_webgputensor_glsl_binary_op.py

    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

    layout(std430, set = 0, binding = 0) readonly buffer arrayLhs {{
    {scalar_type} numbers[];
    }} array_lhs;

    layout(std430, set = 0, binding = 1) readonly buffer arrayRhs {{
    {scalar_type} numbers[];
    }} array_rhs;

    layout(std430, set = 0, binding = 2) buffer arrayC {{
    {scalar_type} numbers[];
    }} array_c;

    layout(std430, set = 0, binding = 3) readonly buffer Meta {{
    uint len;
    }} meta;

    void main() {{
    uint len = meta.len;
    for (uint i = gl_GlobalInvocationID.x; i < len; i += 4096) {{
        {scalar_type} lhs = array_lhs.numbers[0];
        {scalar_type} rhs = array_rhs.numbers[0];
        array_c.numbers[i] = {op};
    }}
    }}

    """
            with open(os.path.join(dst_dir, kernel_name + ".glsl"), "w") as f:
                f.write(source)
