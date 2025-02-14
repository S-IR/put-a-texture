dxc hlsl\vert.hlsl -T vs_6_0 -Zi -E main -spirv -Fo  test.vert.spv
dxc hlsl\frag.hlsl -T ps_6_0 -Zi -E main -spirv -Fo  test.frag.spv

dxc hlsl\vert.hlsl -T vs_6_0  -E main -Fo  test.vert.dxil
dxc hlsl\frag.hlsl -T ps_6_0  -E main -Fo  test.frag.dxil