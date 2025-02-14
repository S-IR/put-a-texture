package main
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"
import "core:time"

import sdl "vendor:sdl3"

window: ^sdl.Window
device: ^sdl.GPUDevice
pipeline: ^sdl.GPUGraphicsPipeline
sampler: ^sdl.GPUSampler

float2 :: [2]f32
float3 :: [3]f32
float4 :: [4]f32
FragUBO :: float4
PositionTextureVertex :: struct {
	x, y, z: f32,
	u, v:    f32,
}

sdl_panic_if :: proc(cond: bool, message: string = "") {
	if cond {
		if len(message) > 0 {
			fmt.println(message)
		}
		fmt.println(sdl.GetError())
		os.exit(1)
	}

}
positions := [4]float3{{-1, 1, 0}, {1, 1, 0}, {1, -1, 0}, {-1, -1, 0}}
texCoords := [4]float2{{0, 0}, {4, 0}, {4, 4}, {0, 4}}
#assert(len(positions) == len(texCoords))

indices := [6]u16{0, 1, 2, 0, 2, 3}
main :: proc() {

	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)

		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not freed: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			if len(track.bad_free_array) > 0 {
				fmt.eprintf("=== %v incorrect frees: ===\n", len(track.bad_free_array))
				for entry in track.bad_free_array {
					fmt.eprintf("- %p @ %v\n", entry.memory, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}
	window = sdl.CreateWindow("test", 1920, 1080, {.FULLSCREEN})
	defer sdl.DestroyWindow(window)

	sdl_panic_if(sdl.Init({.VIDEO}) == false)
	device = sdl.CreateGPUDevice({.DXIL}, true, nil)
	defer sdl.DestroyGPUDevice(device)

	sdl_panic_if(sdl.ClaimWindowForGPUDevice(device, window) == false)

	vertexShader := load_shader(
	device,
	"test.vert.dxil",
	{
		samplerCount = 0,
		UBOs         = 1, // For MatrixTransform
		SBOs         = 0,
		STOs         = 0,
	},
	)

	// For fragment shader (needs sampler in space2 and uniform buffer in space3):
	fragmentShader := load_shader(
	device,
	"test.frag.dxil",
	{
		samplerCount = 1, // For the texture sampler
		UBOs         = 1, // For MultiplyColor
		SBOs         = 0,
		STOs         = 0,
	},
	)

	sdl_panic_if(vertexShader == nil || fragmentShader == nil)


	pipelineInfo := sdl.GPUGraphicsPipelineCreateInfo {
		target_info = {
			num_color_targets = 1,
			color_target_descriptions = raw_data(
				[]sdl.GPUColorTargetDescription {
					{format = sdl.GetGPUSwapchainTextureFormat(device, window)},
				},
			),
		},
		vertex_input_state = {
			num_vertex_buffers = 2,
			vertex_buffer_descriptions = raw_data(
				[]sdl.GPUVertexBufferDescription {
					{
						slot = 0,
						input_rate = .VERTEX,
						instance_step_rate = 0,
						pitch = size_of(float3),
					},
					{
						slot = 1,
						input_rate = .VERTEX,
						instance_step_rate = 0,
						pitch = size_of(float2),
					},
				},
			),
			num_vertex_attributes = 2,
			vertex_attributes = raw_data(
				[]sdl.GPUVertexAttribute {
					{buffer_slot = 0, format = .FLOAT2, location = 0, offset = 0},
					{buffer_slot = 1, format = .FLOAT2, location = 1, offset = 0},
				},
			),
		},
		primitive_type = .TRIANGLELIST,
		vertex_shader = vertexShader,
		fragment_shader = fragmentShader,
	}

	pipeline = sdl.CreateGPUGraphicsPipeline(device, pipelineInfo)
	fmt.println("before pipeline", sdl.GetError())

	sdl_panic_if(pipeline == nil)

	sdl.ReleaseGPUShader(device, vertexShader)
	sdl.ReleaseGPUShader(device, fragmentShader)


	sampler = sdl.CreateGPUSampler(
		device,
		sdl.GPUSamplerCreateInfo {
			min_filter = .NEAREST,
			mag_filter = .NEAREST,
			mipmap_mode = .NEAREST,
			address_mode_u = .CLAMP_TO_EDGE,
			address_mode_v = .CLAMP_TO_EDGE,
			address_mode_w = .CLAMP_TO_EDGE,
		},
	)

	sdl_panic_if(sampler == nil)
	defer sdl.ReleaseGPUSampler(device, sampler) // Add this

	texture := load_image("ravioli.bmp")
	defer sdl.ReleaseGPUTexture(device, texture)


	indexBuffer := sdl.CreateGPUBuffer(
		device,
		sdl.GPUBufferCreateInfo{usage = {.INDEX}, size = size_of(indices)},
	)
	defer sdl.ReleaseGPUBuffer(device, indexBuffer)

	vertexPosBuffer := sdl.CreateGPUBuffer(
		device,
		sdl.GPUBufferCreateInfo{usage = {.VERTEX}, size = size_of(positions)},
	)
	defer sdl.ReleaseGPUBuffer(device, vertexPosBuffer)

	vertexTexCoords := sdl.CreateGPUBuffer(
		device,
		sdl.GPUBufferCreateInfo{usage = {.VERTEX}, size = size_of(texCoords)},
	)
	defer sdl.ReleaseGPUBuffer(device, vertexTexCoords)

	load_into_gpu_buffer(indexBuffer, raw_data(&indices), size_of(indices))
	load_into_gpu_buffer(vertexPosBuffer, raw_data(&positions), size_of(positions))
	load_into_gpu_buffer(vertexTexCoords, raw_data(&texCoords), size_of(texCoords))


	e: sdl.Event
	quit := false

	for !quit {

		defer free_all(context.temp_allocator)

		e: sdl.Event
		frameStart := time.now()

		for sdl.PollEvent(&e) {
			#partial switch e.type {
			case .QUIT:
				quit = true
				break
			case .KEY_DOWN:
				if e.key.key == sdl.K_ESCAPE {
					quit = true
				}
			case:
				continue
			}
		}


		cmdBuf := sdl.AcquireGPUCommandBuffer(device)
		if cmdBuf == nil do continue
		defer sdl_panic_if(sdl.SubmitGPUCommandBuffer(cmdBuf) == false)

		swapTexture: ^sdl.GPUTexture
		if !sdl.WaitAndAcquireGPUSwapchainTexture(cmdBuf, window, &swapTexture, nil, nil) {
			continue
		}


		colorTargetInfo := sdl.GPUColorTargetInfo {
			texture     = swapTexture,
			clear_color = {0.3, 0.2, 0.7, 1.0},
			load_op     = .CLEAR,
			store_op    = .STORE,
		}


		renderPass := sdl.BeginGPURenderPass(cmdBuf, &colorTargetInfo, 1, nil)
		sdl_panic_if(renderPass == nil)


		sdl.BindGPUGraphicsPipeline(renderPass, pipeline)


		sdl.BindGPUVertexBuffers(
			renderPass,
			0,
			raw_data(
				[]sdl.GPUBufferBinding {
					{buffer = vertexPosBuffer, offset = 0},
					{buffer = vertexTexCoords, offset = 0},
				},
			),
			2,
		)

		sdl.BindGPUIndexBuffer(renderPass, {buffer = indexBuffer, offset = 0}, ._16BIT)
		assert(texture != nil && sampler != nil)
		sdl.BindGPUFragmentSamplers(
			renderPass,
			0,
			raw_data([]sdl.GPUTextureSamplerBinding{{texture = texture, sampler = sampler}}),
			1,
		)
		sdl.DrawGPUIndexedPrimitives(renderPass, len(indices), 1, 0, 0, 0)
		sdl.EndGPURenderPass(renderPass)


	}

}

load_into_gpu_buffer :: proc(gpuBuffer: ^sdl.GPUBuffer, data: rawptr, size: uint) {

	assert(data != nil && size > 0)
	transferBuffer := sdl.CreateGPUTransferBuffer(
		device,
		sdl.GPUTransferBufferCreateInfo{usage = .UPLOAD, size = u32(size)},
	)

	infoPtr := sdl.MapGPUTransferBuffer(device, transferBuffer, true)
	sdl.memcpy(infoPtr, data, size)
	sdl.UnmapGPUTransferBuffer(device, transferBuffer)


	uploadCmdBuf := sdl.AcquireGPUCommandBuffer(device)
	copyPass := sdl.BeginGPUCopyPass(uploadCmdBuf)
	sdl.UploadToGPUBuffer(
		copyPass,
		sdl.GPUTransferBufferLocation{offset = 0, transfer_buffer = transferBuffer},
		sdl.GPUBufferRegion{buffer = gpuBuffer, offset = 0, size = u32(size)},
		true,
	)
	sdl.EndGPUCopyPass(copyPass)
	sdl_panic_if(sdl.SubmitGPUCommandBuffer(uploadCmdBuf) == false)

}
ShaderBufferCounts :: struct {
	samplerCount, UBOs, SBOs, STOs: u32,
}


load_shader :: proc(
	device: ^sdl.GPUDevice,
	shaderPath: string,
	using bufferCounts: ShaderBufferCounts = {},
) -> ^sdl.GPUShader {

	stage: sdl.GPUShaderStage
	if strings.contains(shaderPath, ".vert") {
		stage = .VERTEX
	} else if strings.contains(shaderPath, ".frag") {
		stage = .FRAGMENT
	} else {
		panic(
			fmt.tprintf("Shader suffix is neither .vert or .frag, shader path is %s", shaderPath),
		)
	}

	format := sdl.GetGPUShaderFormats(device)
	entrypoint: cstring
	if format >= {.SPIRV} || format >= {.DXIL} {
		entrypoint = "main"
	} else {
		panic("unsupported backend shader format")
	}

	codeSize: uint
	code := sdl.LoadFile(strings.clone_to_cstring(shaderPath, context.temp_allocator), &codeSize)
	sdl_panic_if(code == nil)
	defer sdl.free(code)

	return sdl.CreateGPUShader(
		device,
		sdl.GPUShaderCreateInfo {
			code = transmute([^]u8)(code),
			code_size = codeSize,
			entrypoint = entrypoint,
			format = format,
			stage = stage,
			num_samplers = samplerCount,
			num_uniform_buffers = UBOs,
			num_storage_buffers = SBOs,
			num_storage_textures = STOs,
		},
	)

}
load_image :: proc(path: string) -> ^sdl.GPUTexture {
	if !strings.has_suffix(path, ".bmp") {
		panic(fmt.tprintf("image type not supported : %v", path))
	}

	img := sdl.LoadBMP(strings.clone_to_cstring(path, allocator = context.temp_allocator))

	fmt.printfln("%s : %s", #location(), sdl.GetError())
	texture := sdl.CreateGPUTexture(
		device,
		sdl.GPUTextureCreateInfo {
			type = .D2,
			format = .R8G8B8A8_UNORM,
			width = u32(img.w),
			height = u32(img.h),
			layer_count_or_depth = 1,
			num_levels = 1,
			usage = {.SAMPLER, .COLOR_TARGET},
		},
	)
	fmt.printfln("%s : %s", #location(), sdl.GetError())

	// Create a transfer buffer for the image data
	transferBuffer := sdl.CreateGPUTransferBuffer(
	device,
	sdl.GPUTransferBufferCreateInfo {
		usage = .UPLOAD,
		size  = u32(img.w * img.h * 4), // Assuming RGBA8 format (4 bytes per pixel)
	},
	)
	defer sdl.ReleaseGPUTransferBuffer(device, transferBuffer)
	fmt.printfln("%s : %s", #location(), sdl.GetError())

	// Map the transfer buffer and copy the image data
	data := sdl.MapGPUTransferBuffer(device, transferBuffer, true)
	sdl.memcpy(data, img.pixels, uint(size_of(u8) * img.w * img.h * 4))
	sdl.UnmapGPUTransferBuffer(device, transferBuffer)

	// Upload the data to the texture
	cmdBuf := sdl.AcquireGPUCommandBuffer(device)
	copyPass := sdl.BeginGPUCopyPass(cmdBuf)
	fmt.printfln("%s : %s", #location(), sdl.GetError())

	sdl.UploadToGPUTexture(
		copyPass,
		{offset = 0, transfer_buffer = transferBuffer},
		{texture = texture, w = u32(img.w), h = u32(img.h), d = 1},
		true,
	)

	// sdl.UploadToGPUTexture(
	// 	copyPass,
	// 	{
	// 		transfer_buffer = transferBuffer,
	// 		offset          = 0, /* Zeros out the rest */
	// 	},
	// 	{texture = texture, w = u32(img.w), h = u32(img.h), d = 1},
	// 	false,
	// )

	fmt.printfln("%s : %s", #location(), sdl.GetError())

	sdl.EndGPUCopyPass(copyPass)
	sdl_panic_if(sdl.SubmitGPUCommandBuffer(cmdBuf) == false)
	fmt.printfln("%s : %s", #location(), sdl.GetError())

	return texture
}
