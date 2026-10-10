"""Level-specific NKI guidance derived from checker categories and local references.

Category names below are stable labels from the feedback schema. This module does not
classify errors or change checker behavior; it turns known failure modes into generation
constraints and a short private self-check.
"""

GUIDES = {
    1: """Level 1 average pooling — use the installed NKI 0.6.0 API verified against reference_level1.py.
- Exact APIs: nl.ndarray(shape, dtype, buffer=..., name='', address=None); tile.ap(pattern) where pattern is a list of [stride, count] pairs; nl.sum(x, axis, dtype=None, keepdims=False); nisa.tensor_scalar(dst, data, op0, operand0, ...); nisa.dma_copy(dst, src, ...).
- Do not use nl.Layout, nl.OpCode, ndarray(layout=...), or guessed tensor_scalar keywords such as dtype=. Do not replace the operator with an invented opcode. For average scaling, use op0=nl.multiply and operand0=1.0/(p*p), with dst and data as named arguments.
- Minimal verified pooling body for input in_tensor of shape (C,H,W), with p dividing H and W:
  C,H,W = in_tensor.shape
  out = nl.ndarray((C,H//p,W//p), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
  in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=in_tile, src=in_tensor)
  pool_view = in_tile.ap([[H*W,C], [p*W,H//p], [p,W//p], [W,p], [1,p]])
  sums = nl.sum(pool_view, axis=[3,4])
  scaled = nl.ndarray(sums.shape, dtype=sums.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=scaled, data=sums, op0=nl.multiply, operand0=1.0/(p*p))
  nisa.dma_copy(dst=out, src=scaled)
  return out
- The access pattern presents logical shape (C,H//p,W//p,p,p); reduce only the final two pool axes. Keep the input tile in SBUF and returned output in shared HBM. Match the reference's exact argument names and buffer sequence.
- Checker-to-constraint: If feedback reports an unknown API name or unexpected keyword => use only the verified names and arguments; DMA_SIZE_MISMATCH/TILE_SHAPE_MISMATCH => copy matching shapes; PARTITION_LIMIT/OUT_OF_BOUNDS => derive all extents from C,H,W and stay within limits; NUMERICAL_MISMATCH => reduce both pool axes and divide by p*p.
Before emitting code, privately verify the output shape, access-pattern strides, reduction axes, normalization, and every DMA shape. Return only the requested Python code block.""",
    3: """Level 3 shape and dataflow plan (single-tile matmul):
- lhsT has shape (K,M), rhs has shape (K,N), and result has shape (M,N). The first dimension K is the contraction/partition dimension in both inputs.
- Keep every SBUF and PSUM tile explicitly 2-D: lhs (K,M), rhs (K,N), PSUM/output (M,N). Do not allocate a 1-D PSUM from lhsT.shape[1:].
- Use nisa.dma_copy HBM->SBUF for each operand; nisa.nc_matmul(dst=psum, stationary=lhs_sbuf, moving=rhs_sbuf); nisa.tensor_copy PSUM->SBUF; then nisa.dma_copy SBUF->shared_hbm output. Use keyword arguments for nc_matmul.
- For this level's supplied shape (K=128,M=64,N=512), each operand and result fits one tile under the checker-provided limits; do not pad dimensions or invent a K loop. For other levels, tile each dimension from its actual bounds.
- Checker-to-constraint: TILE_RANK => make each on-chip tile 2-D (partition, free); BUFFER_PLACEMENT => operands in SBUF and nc_matmul destination in PSUM; DMA_SIZE_MISMATCH/TILE_SHAPE_MISMATCH/OUT_OF_BOUNDS => copy matching in-range slices; NUMERICAL_MISMATCH => preserve K contraction and (M,N) result indexing.
Before emitting code, privately verify all tile shapes, buffer regions, the (M,N) result, and the copy sequence. Return only the requested Python code block.""",
}


def guide_for_level(level):
    """Return only guidance relevant to this level."""
    return GUIDES.get(level, "")
