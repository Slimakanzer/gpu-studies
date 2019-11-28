.include "gpr_alloc.inc"

.hsa_code_object_version 2,1
.hsa_code_object_isa

.GPR_ALLOC_BEGIN
	kernarg = 0
	gid_x = 2
	flt_min = 0
	pad_value = 0
	wgroup_size = 64
	.SGPR_ALLOC_FROM 5
	.SGPR_ALLOC s
	.SGPR_ALLOC p
	.SGPR_ALLOC t
	.SGPR_ALLOC w
	.SGPR_ALLOC goffset_x
	.SGPR_ALLOC base_in, 2
	.SGPR_ALLOC base_out, 2
	.SGPR_ALLOC lpad_exec, 2
	.SGPR_ALLOC rpad_exec, 2
	.SGPR_ALLOC init_exec, 2

	.VGPR_ALLOC_FROM 0
	.VGPR_ALLOC tid
	.VGPR_ALLOC vindex
	.VGPR_ALLOC vgrid_id_x
	.VGPR_ALLOC voffset
	.VGPR_ALLOC vpool_item
	.VGPR_ALLOC vpool_item_max
	.VGPR_ALLOC vaddr, 2
.GPR_ALLOC_END

.text
.p2align 8
.amdgpu_hsa_kernel hello_world

hello_world:

  .amd_kernel_code_t
	is_ptr64 = 1
	enable_sgpr_kernarg_segment_ptr = 1
	enable_sgpr_workgroup_id_x = 1
	kernarg_segment_byte_size = 32
	compute_pgm_rsrc2_user_sgpr = 2
	granulated_workitem_vgpr_count = .AUTO_VGPR_GRANULATED_COUNT
	granulated_wavefront_sgpr_count = .AUTO_SGPR_GRANULATED_COUNT
	wavefront_sgpr_count = .AUTO_SGPR_COUNT
	workitem_vgpr_count = .AUTO_VGPR_COUNT
  .end_amd_kernel_code_t

  // read kernel arguments:
  // s[base_in:base_in+1] = *in
  // s[base_out:base_out+1] = *out
  // s[w] = out width
  // s[s] = kernel size
  // s[p] = padding size
  // s[t] = stride size
  s_load_dwordx2    s[base_in:base_in+1], s[kernarg:kernarg+1], 0x00
  s_load_dwordx2    s[base_out:base_out+1], s[kernarg:kernarg+1], 0x08
  s_load_dword      s[w], s[kernarg:kernarg+1], 0x10
  s_load_dword      s[s], s[kernarg:kernarg+1], 0x14
  s_load_dword      s[p], s[kernarg:kernarg+1], 0x18
  s_load_dword      s[t], s[kernarg:kernarg+1], 0x1c
  
  v_mov_b32         v[vpool_item_max], flt_min           // set initial value to flt_min
  s_mov_b64         s[init_exec:init_exec+1], exec       // save initial exec mask
  s_mul_i32         s[goffset_x], s[gid_x], wgroup_size  // calculate group start base index
  v_add_i32         v[vgrid_id_x], v[tid], s[goffset_x]  // calculate id in grid 
  s_waitcnt         0
  
  v_mul_u32_u24     v[vindex], v[vgrid_id_x], s[t]             // calculate intial array index
  v_sub_i32         v[vindex], v[vindex], s[p]          //
  
  v_mov_b32         v[vaddr+1], s[base_in+1]
  v_lshlrev_b32     v[voffset], 2, v[vindex]
  v_add_u32         v[vaddr], v[voffset], s[base_in]

  pooling_loop:
    v_cmp_ge_i32      s[lpad_exec:lpad_exec+1], v[vindex], 0
    v_cmp_lt_i32      s[rpad_exec:rpad_exec+1], v[vindex], s[w] // similary (vindex < w - 1)

    // If vindex is valid (vindex > 0 && vindex < w - 1) then load value from input buffer
    // Else set value as padding default value (0.0f)
    s_and_b64         exec, s[lpad_exec:lpad_exec+1], s[rpad_exec:rpad_exec+1]
      flat_load_dword   v[vpool_item], v[vaddr:vaddr+1]
    s_not_b64         exec, exec
      v_mov_b32         v[vpool_item], pad_value
    s_mov_b64         exec, s[init_exec:init_exec+1] // return initial exec mask

    v_add_u32         v[vindex], v[vindex], 1
    v_add_co_u32      v[vaddr], vcc, v[vaddr], 4
    v_addc_co_u32     v[vaddr+1], vcc, v[vaddr+1], 0, vcc
    s_waitcnt         0

    v_max_f32         v[vpool_item_max], v[vpool_item_max], v[vpool_item] // calculate max value

    s_sub_u32         s[s], s[s], 1  //
    s_cmp_gt_u32      s[s], 0        // if s[s] > 0 then goto `pooling_loop`
    s_cbranch_scc1    pooling_loop   //

  // v[vaddr:vaddr+1] = &out[i]
  v_lshlrev_b32     v[voffset], 2, v[vgrid_id_x]
  v_add_co_u32      v[vaddr], vcc, s[base_out], v[voffset]
  v_mov_b32         v[vaddr+1], s[base_out+1]
  v_addc_co_u32     v[vaddr+1], vcc, v[vaddr+1], 0, vcc

  flat_store_dword  v[vaddr:vaddr+1], v[vpool_item_max]
  s_endpgm