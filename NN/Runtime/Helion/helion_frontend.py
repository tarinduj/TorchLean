"""Check the supported Helion host-code pattern and translate device IR into Lean."""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import textwrap

import torch

_printer_spec = importlib.util.spec_from_file_location(
    "op_printer", pathlib.Path(__file__).with_name("op_printer.py")
)
op_printer = importlib.util.module_from_spec(_printer_spec)
_printer_spec.loader.exec_module(op_printer)

refuse, invalid = op_printer.refuse, op_printer.invalid


def _indent(text: str, spaces: int) -> str:
    return "\n".join(" " * spaces + line for line in text.splitlines())


def _tile_args(blocks: list[int], bounds: list[str]) -> str:
    """Print a tile's block sizes and loop bounds, the first arguments of the tile ops: `[B0, B1] ![d0, d1]`."""
    return f"[{', '.join(f'B{b}' for b in blocks)}] ![{', '.join(bounds)}]"


def _doc_comment(text: str) -> str:
    """Lay out `text` as a Lean doc comment whose lines fit in the printer's width."""
    lines = textwrap.wrap(
        text, width=op_printer.WIDTH - 7, break_long_words=False, break_on_hyphens=False
    )
    lines = ["/-- " + lines[0]] + ["    " + line for line in lines[1:]]
    return "\n".join(lines) + " -/"


def _is_attr(expr, object_name: str, attribute: str) -> bool:
    """Check for a named attribute such as `x.size` or `hl.tile`."""
    return (
        isinstance(expr, ast.Attribute)
        and expr.attr == attribute
        and isinstance(expr.value, ast.Name)
        and expr.value.id == object_name
    )


def _assigned_names(statement) -> list[str] | None:
    """Get assigned names so the host check can track size variables and the output.

    Return None unless this is one assignment to a name or a tuple of distinct names.
    """
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    if isinstance(target, ast.Name):
        elements = [target]
    elif isinstance(target, ast.Tuple):
        elements = target.elts
    else:
        return None
    if not elements or any(not isinstance(element, ast.Name) for element in elements):
        return None
    names = [element.id for element in elements]
    return names if len(set(names)) == len(names) else None


class Emitter(op_printer.FxEmitter):
    """Check a captured Helion kernel and print its device IR as a Lean definition."""

    def __init__(self, kernel, example_args: tuple, bridge):
        """Capture device IR and validate the inputs and host-code pattern.

        Require tensor arguments with symbolic sizes, named by `op_printer.input_name` and `size_name`.
        The shared emitter replaces `bridge._DIM_NAMES` with this kernel's size-name map.
        """
        if kernel.settings.static_shapes:
            refuse("only static_shapes=False is supported")
        try:
            bound_kernel = kernel.bind(tuple(example_args))
        except Exception as error:
            first_line = str(error).splitlines()[0] if str(error) else ""
            invalid(f"Helion could not bind the kernel: {type(error).__name__}: {first_line}")
        self.fn_name = kernel.fn.__name__
        self.host_function = bound_kernel.host_function
        self.helion_env = bound_kernel.env
        self.device_ir = self.host_function.device_ir
        params = [arg.arg for arg in self.host_function.definition.args.args]
        if len(params) != len(example_args):
            refuse("only calls that pass every parameter are supported")
        # Python argument name -> Lean input name, in signature order.
        self.lean_inputs = {name: op_printer.input_name(i) for i, name in enumerate(params)}
        # Name each size symbol by first appearance; sizes Helion shares between inputs share a name.
        dimension_names, self.input_shapes = {}, {}
        for name, fake in zip(params, bound_kernel.fake_args):
            if not isinstance(fake, torch.Tensor):
                refuse("only tensor parameters are supported")
            if not all(isinstance(size, torch.SymInt) for size in fake.shape):
                refuse("only symbolic sizes are supported")
            for size in fake.shape:
                dimension_names.setdefault(size.node.expr, op_printer.size_name(len(dimension_names)))
            self.input_shapes[self.lean_inputs[name]] = [dimension_names[size.node.expr] for size in fake.shape]
        self.dim_names = list(dimension_names.values())
        self.block_ids = {
            block.var.node.expr: block.block_id for block in self.helion_env.block_sizes
        }
        block_names = {symbol: f"B{block_id}" for symbol, block_id in self.block_ids.items()}
        super().__init__(bridge, dimension_names | block_names)
        self.output_name = self._validate_host_and_find_output()
        self.store_count = 0
        self.in_bounds = []   # loop bound ≤ tensor size, for every load and store; trivial ones omitted
        self.full_slices = []  # axis size ≤ block size, for every full slice (`:`)

    def _validate_host_and_find_output(self) -> str:
        """Check this host-code pattern and return the allocated output's name:

            m, n = x.size()
            result = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                ...
            return result
        """
        body = [
            statement
            for statement in self.host_function.definition.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
            )
        ]
        if (
            len(body) < 3
            or not isinstance(body[-2], ast.For)
            or not isinstance(body[-1], ast.Return)
        ):
            refuse("only host code that ends with one grid loop followed by `return` is supported")
        *prelude, grid_loop, return_statement = body
        used_names = set(self.lean_inputs)
        size_names = set()
        output_name = None
        for statement in prelude:
            names = _assigned_names(statement)
            if (
                names is None
                or not isinstance(statement.value, ast.Call)
                or used_names & set(names)
            ):
               th
            used_names.update(names)
            if self._is_input_size_read(statement.value):
                size_names.update(names)
            elif (
                isinstance(statement.targets[0], ast.Name)
                and output_name is None
                and self._is_output_allocation(statement.value, size_names)
            ):
                output_name = names[0]
            else:
                refuse("only reading sizes and allocating the output are supported in host code")
        if not self._is_grid_iterator(grid_loop.iter, size_names):
            refuse("only grid loops over `hl.tile` of input sizes are supported")
        if any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and (node.id in self.lean_inputs or node.id == output_name)
            for node in ast.walk(grid_loop)
        ):
            refuse("assigning to an argument or the output in the grid loop is not supported")
        if output_name is None or not (
            isinstance(return_statement.value, ast.Name)
            and return_statement.value.id == output_name
        ):
            refuse("only returning the `torch.empty` output is supported")
        return output_name

    def _is_input_size_read(self, call) -> bool:
        """Check `x.size()` or `x.size(<integer literal>)` for an input `x` in the host code."""
        if not any(_is_attr(call.func, name, "size") for name in self.lean_inputs) or call.keywords:
            return False
        if not call.args:
            return True
        return (
            len(call.args) == 1
            and isinstance(call.args[0], ast.Constant)
            and type(call.args[0].value) is int
        )

    def _is_output_allocation(self, call, size_names: set[str]) -> bool:
        """Accept `torch.empty([m])` when `m` came from `x.size()`.

        `dtype=x.dtype` and `device=x.device` are the only supported keywords.
        """
        shape = call.args[0] if len(call.args) == 1 else None
        return (
            _is_attr(call.func, "torch", "empty")
            and isinstance(shape, (ast.List, ast.Tuple))
            and all(
                isinstance(element, ast.Name) and element.id in size_names
                for element in shape.elts
            )
            and all(
                keyword.arg in ("dtype", "device")
                and any(_is_attr(keyword.value, name, keyword.arg) for name in self.lean_inputs)
                for keyword in call.keywords
            )
        )

    def _is_grid_iterator(self, iterator, size_names: set[str]) -> bool:
        """Accept `hl.tile(m)` or `hl.tile([m, n])` when each size came from an input's `size()`."""
        if not (
            isinstance(iterator, ast.Call)
            and _is_attr(iterator.func, "hl", "tile")
            and not iterator.keywords
            and len(iterator.args) == 1
        ):
            return False
        argument = iterator.args[0]
        sizes = argument.elts if isinstance(argument, (ast.List, ast.Tuple)) else [argument]
        return bool(sizes) and all(
            isinstance(size, ast.Name) and size.id in size_names for size in sizes
        )

    def _block_id_of(self, value) -> int:
        """Look up the block number Helion assigned to this symbolic tile index."""
        if isinstance(value, torch.SymInt) and value.node.expr in self.block_ids:
            return self.block_ids[value.node.expr]
        refuse("only tiled axes are supported")

    def _loop_bounds(self, blocks: list[int]) -> list[str]:
        """Return the sizes the loops of `blocks` run to: Helion's mask bounds for their tiles."""
        return [self.sym(self.helion_env.block_sizes[block].size) for block in blocks]

    def _record_in_bounds(self, bounds: list[str], dimensions: list[str]):
        """Record that every masked-in lane lies inside the tensor: `bound ≤ size` on each axis."""
        if len(bounds) != len(dimensions):
            invalid(f"tile of rank {len(bounds)} on a tensor of rank {len(dimensions)}")
        for bound, size in zip(bounds, dimensions, strict=True):
            if bound != size and f"{bound} ≤ {size}" not in self.in_bounds:
                self.in_bounds.append(f"{bound} ≤ {size}")

    def _tile_layout(self, ir_node) -> tuple[list[int], list[str], list[str]]:
        """Find each tile axis's block ID, loop bound and tile index.

        Record coverage requirements for full slices (`:`).
        """
        # Get each axis's block ID from the captured shape, since `:` has no tile variable.
        blocks = [self._block_id_of(size) for size in ir_node.meta["val"].shape]
        tiles = []
        for block in blocks:
            info = self.helion_env.block_sizes[block]
            # Helion allocates the block of a full slice with `allocate_reduction_dimension`, which
            # sets its `reduction` flag even when nothing is reduced over it.
            if info.reduction:
                # Represent the full slice (`:`) with one block starting at index 0.
                # Require that block to cover the axis, e.g. `d1 ≤ B1`.
                bound = self.sym(info.size)
                if f"{bound} ≤ B{block}" not in self.full_slices:
                    self.full_slices.append(f"{bound} ≤ B{block}")
                tiles.append("0")
            else:
                tiles.append(f"tile{block}")
        return blocks, self._loop_bounds(blocks), tiles

    def _indexed_tile_layout(self, ir_node, indices, context: dict) -> tuple[list[int], list[str], list[str]]:
        """Check that each index is an enclosing loop's tile or a full slice (`:`).

        Return the block IDs, loop bounds and tile indices for the load or store.
        """
        blocks, bounds, tiles = self._tile_layout(ir_node)
        if len(indices) != len(blocks):
            refuse("only indexing every axis with a tile or `:` is supported")
        for index, block, tile_index in zip(indices, blocks, tiles):
            if isinstance(index, slice):
                # Only full slices (`:`) are supported; their block starts at tile index 0.
                if index != slice(None):
                    refuse("only full slices `:` are supported")
                if tile_index != "0":
                    invalid("expected Helion to cover a `:` axis with one whole-axis block")
            elif not isinstance(index, torch.fx.Node):
                refuse("only tiles and `:` are supported as indices")
            # A tile index must be the current tile of an enclosing loop over that axis's block.
            elif self._block_id_of(index.meta["val"]) != block or block not in context["tiles"]:
                invalid("index is not the tile of an enclosing loop for that axis")
        return blocks, bounds, tiles

    def aten(self, ir_node, values: dict) -> str:
        """Translate masked `amax`; send other ATen ops to the shared printer."""
        if op_printer.op_name(ir_node) != "amax.default":
            return super().aten(ir_node, values)
        operand = ir_node.args[0]
        dims = ir_node.args[1] if len(ir_node.args) > 1 else ir_node.kwargs.get("dim", [])
        keepdim = ir_node.args[2] if len(ir_node.args) > 2 else ir_node.kwargs.get("keepdim", False)
        # Accept only an amax whose operand is `_mask_to(…, -inf)`: ℝ has no -inf, so the `_mask_to`
        # and the `amax` become one `maskedMax`.
        if (
            not isinstance(operand, torch.fx.Node)
            or op_printer.op_name(operand) != "_mask_to"
            or operand.args[1] != float("-inf")
        ):
            refuse("only amax over an input masked to -inf is supported")
        rank = len(operand.meta["val"].shape)
        if len(dims) != 1:
            refuse("only amax over one axis is supported")
        if not -rank <= dims[0] < rank:
            invalid(f"amax axis {dims[0]} is out of range for a rank-{rank} tensor")
        axis = dims[0] % rank  # Convert negative axis numbering to nonnegative.
        blocks, bounds, tiles = self._tile_layout(operand)
        original = values[operand.args[0]]  # The tile before `_mask_to`; `maskedMax` skips its masked lanes.
        call = f"(maskedMax {_tile_args(blocks, bounds)} ![{', '.join(tiles)}] {axis} {original})"
        # Restore the singleton axis when keepdim=True.
        return self.reshape(call, self.shape(ir_node)) if keepdim else call

    def node(self, ir_node, values: dict, context: dict):
        """Translate a Helion-specific IR node into Lean.

        Tile loops, carried values, loads, and stores are handled here. ATen
        tensor operations go through the shared FX walker and op printer.
        """
        args = ir_node.args
        op = op_printer.op_name(ir_node)
        if op == "_get_symnode":
            values[ir_node] = self.sym(ir_node.meta["val"])
        elif op == "_new_var":
            values[ir_node] = values[args[0]]
        elif op == "_host_tensor":
            if args[0] not in self.lean_inputs and args[0] != self.output_name:
                refuse("only the arguments and the output are supported in device code")
            values[ir_node] = self.lean_inputs.get(args[0], ("OUT",))
        elif op == "full":
            if args[1] != 0:
                refuse("only zero accumulators are supported")
            values[ir_node] = f"(Tensor.zeros [{', '.join(self.shape(ir_node))}])"
        elif op == "_for_loop":
            values[ir_node] = self._emit_inner_loop(ir_node, values, context["tiles"])
        elif op == "getitem":
            if args[1] != 0:
                refuse("only loops with one result are supported")
            values[ir_node] = values[args[0]]
        elif op == "_phi":
            before, after = args[0], args[1]
            loop = after.args[0] if op_printer.op_name(after) == "getitem" else None
            if (
                loop is None
                or op_printer.op_name(loop) != "_for_loop"
                or loop.args[3][0] is not before
            ):
                refuse("only phis for a loop's carried value are supported")
            block_id = self.device_ir.graphs[loop.args[0]].block_ids[0]
            name = f"acc{block_id}"
            context["lets"].append(f"let {name} := {op_printer.unparen(values[args[1]])}")
            values[ir_node] = name
        elif op == "_mask_to":
            if args[1] == float("-inf"):
                users = list(ir_node.users)
                if len(users) != 1 or op_printer.op_name(users[0]) != "amax.default":
                    refuse("_mask_to(-inf) is supported only as the input of amax")
                values[ir_node] = None   # printed by its amax
            elif args[1] == 0:
                blocks, bounds, tiles = self._tile_layout(ir_node)
                values[ir_node] = f"(maskTo {_tile_args(blocks, bounds)} ![{', '.join(tiles)}] 0 {values[args[0]]})"
            else:
                refuse("only _mask_to with fill 0 or -inf is supported")
        elif op == "load":
            source, indices = args[0], args[1]
            if values[source] not in self.input_shapes:
                refuse("only loads from an argument are supported")
            if len(args) > 2 and args[2] is not None:
                refuse("only loads without an extra mask are supported")
            blocks, bounds, tiles = self._indexed_tile_layout(ir_node, indices, context)
            self._record_in_bounds(bounds, self.shape(source))
            values[ir_node] = f"(loadTile {_tile_args(blocks, bounds)} 0 {values[source]} ![{', '.join(tiles)}])"
        elif op == "store":
            destination, indices, value = args[0], args[1], args[2]
            if context["state"] is None:
                refuse("only stores in the grid body are supported")
            if values[destination] != ("OUT",):
                refuse("only stores into the output are supported")
            if len(args) > 3 and args[3] is not None:
                refuse("only stores without an extra mask are supported")
            blocks, bounds, tiles = self._indexed_tile_layout(value, indices, context)
            if [b for b, t in zip(blocks, tiles) if t != "0"] != self.grid_blocks:
                refuse("only stores at the grid tiles, in order, are supported")
            self.output_shape = self.shape(destination)
            self._record_in_bounds(bounds, self.output_shape)
            self.store_count += 1
            context["state"] = (
                f"(storeTile {_tile_args(blocks, bounds)} {context['state']} ![{', '.join(tiles)}] {values[value]})"
            )
        else:
            refuse(f"device op {op} is not supported")

    def _emit_inner_loop(self, ir_node, values: dict, enclosing_tiles: set[int]) -> str:
        """Emit a Lean `forTiles` expression for an inner `hl.tile` loop that updates one value."""
        graph_id, begin, end, initial_values = ir_node.args[:4]
        loop_graph = self.device_ir.graphs[graph_id]
        if len(loop_graph.block_ids) != 1 or len(end) != 1:
            refuse("only inner loops over one axis are supported")
        if list(begin) != [0]:
            refuse("only inner loops from 0 are supported")
        if len(initial_values) != 1:
            refuse("only inner loops carrying one value are supported")
        (block_id,) = loop_graph.block_ids
        extent = self.sym(end[0].meta["val"])
        if extent not in self.dim_names:
            refuse("only inner loops up to an input size are supported")
        accumulator = f"acc{block_id}"
        placeholders = loop_graph.graph.find_nodes(op="placeholder")
        if len(placeholders) != len(initial_values):
            invalid(f"inner loop body has {len(placeholders)} inputs but the loop passes {len(initial_values)}")
        placeholder = placeholders[0]
        body = self.emit_graph(
            loop_graph.graph,
            {**values, placeholder: accumulator},
            {"tiles": enclosing_tiles | {block_id}, "state": None, "lets": []},
        )
        return (
            f"(forTiles {extent} B{block_id} {values[initial_values[0]]} "
            f"fun tile{block_id} {accumulator} =>\n{_indent(body, 2)})"
        )

    def emit(self, name: str, source: str) -> str:
        """Emit the complete Lean module for the outer `hl.tile` grid loop and its body."""
        if len(self.device_ir.grid_block_ids) != 1:
            refuse("only one grid loop is supported")
        self.grid_blocks = list(self.device_ir.grid_block_ids[0])
        if len(self.device_ir.root_ids) != 1:
            refuse("only one root graph is supported")
        root = self.device_ir.graphs[self.device_ir.root_ids[0]].graph
        body = self.emit_graph(root, {}, {"tiles": set(self.grid_blocks), "state": "out", "lets": []})
        # Only the root graph, where each full slice is one block, is translated. Helion runs it
        # for configs whose `reduction_loops` entries are `None`; other configs run the
        # reduction-loop graphs, which split full slices into chunks.
        if (
            any(type(graph).__name__ == "ReductionLoopGraphInfo" for graph in self.device_ir.graphs)
            and not self.full_slices
        ):
            invalid("device IR has reduction-loop graphs but the root graph has no full slice")
        if self.store_count == 0:
            invalid(f"the kernel never stores into {self.output_name}")
        if self.store_count > 1:
            refuse("only one store into the output is supported")
        output_shape = ", ".join(self.output_shape)
        block_parameters = " ".join(f"B{block}" for block in sorted(self.block_ids.values()))
        positivity_hypotheses = [
            f"(h{dimension} : 0 < {dimension})" for dimension in sorted(self.hyps)
        ]
        signature = op_printer.print_def(
            "noncomputable def kernel",
            [
                f"({' '.join(self.dim_names)} {block_parameters} : ℕ)",
                *positivity_hypotheses,
                f"(out0 : Tensor ℝ [{output_shape}])",
                *(f"({name} : Tensor ℝ [{', '.join(shape)}])" for name, shape in self.input_shapes.items()),
            ],
            f"Tensor ℝ [{output_shape}]",
        )
        inputs = ", ".join(f"`{name}`" for name in self.input_shapes)
        # One `forTiles` per grid block, outermost first; each loop passes the output to the next.
        grid_loops = "".join(
            " " * (2 * depth + 2)
            + f"forTiles {self.sym(self.helion_env.block_sizes[block].size)} B{block} "
            + f"{'out0' if depth == 0 else 'out'} fun tile{block} out =>\n"
            for depth, block in enumerate(self.grid_blocks)
        )
        # With full slices, Helion also has reduction-loop graphs that other configs run.
        scope = " This is Helion's root device IR, run by configs without reduction loops." if self.full_slices else ""
        docstring = _doc_comment(
            f"Run the Helion `{self.fn_name}` kernel on {inputs}, writing into the output buffer `out0`.{scope}"
        )
        if any(len(line) > op_printer.WIDTH for line in docstring.splitlines()):
            docstring = _doc_comment(
                f"Run the Helion kernel on {inputs}, writing into the output buffer `out0`.{scope}"
            )
        return (
            f"-- Generated from {source} by NN/Runtime/Helion/helion_frontend.py; do not edit.\n"
            "import NN.Runtime.Helion.Tile\n\n"
            "open Spec TorchLean TorchLean.Helion\n\n"
            f"namespace {name}\n\n"
            f"{docstring}\n"
            f"{signature}\n"
            f"{grid_loops}"
            f"{op_printer.wrap_lines(_indent(body, 2 * len(self.grid_blocks) + 2))}\n\n"
            f"end {name}\n"
        )
