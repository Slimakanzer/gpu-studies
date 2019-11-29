.include "gpr_alloc.inc"

.hsa_code_object_version 2,1
.hsa_code_object_isa

.GPR_ALLOC_BEGIN
  kernarg = 0
  gid_x = 2
  flt_min = 0xffffffff
  gsize = 256
  // s[0:1] = kernarg addr
  // s[2] = gid_x
  .SGPR_ALLOC_FROM 3
  .SGPR_ALLOC tmp
  .SGPR_ALLOC base_in, 2
  .SGPR_ALLOC base_out, 2
  .SGPR_ALLOC n
  .SGPR_ALLOC c
  .SGPR_ALLOC h
  .SGPR_ALLOC w
  .SGPR_ALLOC r
  .SGPR_ALLOC s
  .SGPR_ALLOC pad_h
  .SGPR_ALLOC pad_w
  .SGPR_ALLOC stride_h
  .SGPR_ALLOC stride_w
  .SGPR_ALLOC out_h
  .SGPR_ALLOC out_w
  .SGPR_ALLOC in_desc, 4
  .SGPR_ALLOC out_desc, 4
  .SGPR_ALLOC base_offset
  .SGPR_ALLOC base_out_offset
  .SGPR_ALLOC soffset
  .SGPR_ALLOC soffset_tmp
  .SGPR_ALLOC sout_offset
  .SGPR_ALLOC loop_filter_x
  .SGPR_ALLOC loop_filter_y
  .SGPR_ALLOC loop_x
  .SGPR_ALLOC step_y
  .SGPR_ALLOC step_x
  .SGPR_ALLOC step_next_out_row
  .SGPR_ALLOC step_out

  .VGPR_ALLOC_FROM 0
  .VGPR_ALLOC tid
  .VGPR_ALLOC voffset
  .VGPR_ALLOC voffset_out
  .VGPR_ALLOC vpool_item
  .VGPR_ALLOC vpool_item_max
.GPR_ALLOC_END

.text
.p2align 8
.amdgpu_hsa_kernel hello_world

hello_world:

  .amd_kernel_code_t
    is_ptr64 = 1
    enable_sgpr_kernarg_segment_ptr = 1
    enable_sgpr_workgroup_id_x = 1
    kernarg_segment_byte_size = 64
    compute_pgm_rsrc2_user_sgpr = 2
    granulated_workitem_vgpr_count = .AUTO_VGPR_GRANULATED_COUNT
    granulated_wavefront_sgpr_count = .AUTO_SGPR_GRANULATED_COUNT
    wavefront_sgpr_count = .AUTO_SGPR_COUNT
    workitem_vgpr_count = .AUTO_VGPR_COUNT
  .end_amd_kernel_code_t

  // read kernel arguments:
  // s[base_in:base_in+1] = *in
  // s[base_out:base_out+1] = *out
  // s[n] = batch size
  // s[c] = input depth
  // s[h] = input height
  // s[w] = input width
  // s[r] = kernel height
  // s[s] = kernel width
  // s[pad_h] = padding height
  // s[pad_w] = padding width
  // s[stride_h] = stride input tensor height
  // s[stride_w] = stride input tensor width
  // s[out_h] = output height
  // s[out_w] = output width
  s_load_dwordx2    s[base_in:base_in+1], s[kernarg:kernarg+1], 0x00
  s_load_dwordx2    s[base_out:base_out+1], s[kernarg:kernarg+1], 0x08
  s_load_dwordx8    s[n:pad_w], s[kernarg:kernarg+1], 0x10
  s_load_dwordx4    s[stride_h:out_w], s[kernarg:kernarg+1], 0x30

  v_mov_b32         v[vpool_item_max], flt_min           // set initial value to flt_min
  v_lshlrev_b32     v[voffset_out], 2, v[tid]
  s_waitcnt         0

  .GPR_REUSE tid, vindex_x
  v_mul_u32_u24     v[vindex_x], v[vindex_x], s[stride_w]

  // setup base offset
  s_mul_i32         s[base_offset], s[gid_x], s[w]
  s_mul_i32         s[base_offset], s[base_offset], s[h]
  s_lshl_b32        s[base_offset], s[base_offset], 2

  s_mul_i32         s[base_out_offset], s[gid_x], s[out_w]
  s_mul_i32         s[base_out_offset], s[base_out_offset], s[out_h]
  s_lshl_b32        s[base_out_offset], s[base_out_offset], 2

  // setup steps
  s_mul_i32         s[step_next_out_row], s[w], s[stride_h]
  s_lshl_b32        s[step_next_out_row], s[step_next_out_row], 2
  s_lshl_b32        s[step_out], s[out_w], 2

  s_mul_i32         s[step_x], s[stride_w], gsize * 4

  // setup buffer descriptor
  s_mul_i32         s[in_desc+2], s[w], 4       // setup buffer size (width * sizeof(float))
  s_mov_b32         s[in_desc+3], 0x00020000

  s_mul_i32         s[out_desc+2], s[out_w], 4       // setup buffer size (width * sizeof(float))
  s_mov_b32         s[out_desc+3], 0x00020000

  // setup initial states
  .GPR_REUSE out_h, loop_y
  s_mul_i32         s[step_y], s[w], 4
  v_lshlrev_b32     v[voffset], 2, v[vindex_x]  // setup Voffset

  pooling_loop_y:
    s_mov_b32         s[soffset], 0
    s_mov_b32         s[sout_offset], 0
    s_mov_b32         s[loop_x], s[out_w]

    pooling_loop_x:
      s_add_u32         s[in_desc], s[base_in], s[base_offset]
      s_addc_u32        s[in_desc+1], s[base_in+1], 0

      s_add_u32         s[out_desc], s[base_out], s[base_out_offset]
      s_addc_u32        s[out_desc+1], s[base_out+1], 0
    
      s_mov_b32         s[loop_filter_y], s[r]

      pooling_loop_filter_y:
        s_mov_b32         s[soffset_tmp], s[soffset]
        s_mov_b32         s[loop_filter_x], s[s]

        pooling_loop_filter_x:
          buffer_load_dword v[vpool_item], v[voffset], s[in_desc:in_desc+3], s[soffset_tmp] offen
          s_sub_u32         s[loop_filter_x], s[loop_filter_x], 1
          s_add_u32         s[soffset_tmp], s[soffset_tmp], 4
          s_waitcnt         0

          v_max_f32         v[vpool_item_max], v[vpool_item_max], v[vpool_item] // calculate max value
          s_cmp_gt_u32      s[loop_filter_x], 0             // if s[loop_filter_x] > 0 then goto `pooling_loop`
          s_cbranch_scc1    pooling_loop_filter_x    //

        s_add_u32         s[in_desc], s[in_desc], s[step_y]
        s_addc_u32        s[in_desc+1], s[in_desc+1], 0

        s_sub_u32         s[loop_filter_y], s[loop_filter_y], 1
        s_cmp_gt_u32      s[loop_filter_y], 0
        s_cbranch_scc1    pooling_loop_filter_y

      buffer_store_dword  v[vpool_item_max], v[voffset_out], s[out_desc:out_desc+3], s[sout_offset], offen

      s_add_u32         s[soffset], s[soffset], s[step_x]
      s_add_u32         s[sout_offset], s[sout_offset], gsize * 4

      s_sub_i32         s[loop_x], s[loop_x], gsize
      s_cmp_gt_i32      s[loop_x], 0
      s_cbranch_scc1    pooling_loop_x

    // to the next output row
    s_add_u32         s[base_offset], s[base_offset], s[step_next_out_row]
    s_add_u32         s[base_out_offset], s[base_out_offset], s[step_out]

    s_sub_u32         s[loop_y], s[loop_y], 1
    s_cmp_gt_u32      s[loop_y], 0
    s_cbranch_scc1    pooling_loop_y

  s_endpgm
