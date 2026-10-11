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

refuse = op_printer.refuse


def _indent(text: str, spaces: int) -> str:
    return "\n".join(" " * spaces + line for line in text.splitlines())


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

    def __init__(self, kernel, example: torch.Tensor, bridge):
        """Capture device IR and validate the input and host-code pattern.

        Require symbolic sizes and one tensor argument, named by `op_printer.input_name` and `size_name`.
        The shared emitter replaces `bridge._DIM_NAMES` with this kernel's size-name map.
        """
        if kernel.settings.static_shapes:
            refuse("kernel must set static_shapes=False so sizes stay symbolic")
        try:
            bound_kernel = kernel.bind((example,))
        except Exception as error:
            first_line = str(error).splitlines()[0] if str(error) else ""
            refuse(f"Helion could not bind the kernel: {type(error).__name__}: {first_line}")
        self.fn_name = kernel.fn.__name__
        self.host_function = bound_kernel.host_function
        self.helion_env = bound_kernel.env
        self.device_ir = self.host_function.device_ir
        params = [arg.arg for arg in self.host_function.definition.args.args]
        if len(params) != 1:
            refuse(f"kernel takes {params}; one tensor argument for now")
        self.input_name = params[0]
        self.lean_input = op_printer.input_name(0)
        self.dim_names = [op_printer.size_name(axis) for axis in range(example.dim())]
        sizes = list(bound_kernel.fake_args[0].shape)
        if not all(isinstance(size, torch.SymInt) for size in sizes):
            refuse(f"a size of {self.input_name} is static")
        self.block_ids = {
            block.var.node.expr: block.block_id for block in self.helion_env.block_sizes
        }
        dimension_names = {size.node.expr: name for size, name in zip(sizes, self.dim_names)}
        block_names = {symbol: f"B{block_id}" for symbol, block_id in self.block_ids.items()}
        super().__init__(bridge, dimension_names | block_names)
        self.output_name = self._validate_host_and_find_output()
        self.store_count = 0

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
            refuse("host code must end with one grid loop followed by `return`")
        *prelude, grid_loop, return_statement = body
        used_names = {self.input_name}
        size_names = set()
        output_name = None
        for statement in prelude:
            names = _assigned_names(statement)
            if (
                names is None
                or not isinstance(statement.value, ast.Call)
                or used_names & set(names)
            ):
                refuse(f"host statement not in the supported pattern: {ast.unparse(statement)}")
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
                refuse(f"host statement not in the supported pattern: {ast.unparse(statement)}")
        if not self._is_grid_iterator(grid_loop.iter, size_names):
            refuse(f"grid loop must be `hl.tile(size)`: {ast.unparse(grid_loop.iter)}")
        if any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id in (self.input_name, output_name)
            for node in ast.walk(grid_loop)
        ):
            refuse("the grid loop rebinds the argument or the output")
        if output_name is None or not (
            isinstance(return_statement.value, ast.Name)
            and return_statement.value.id == output_name
        ):
            refuse("the kernel must return the tensor it allocates with torch.empty")
        return output_name

    def _is_input_size_read(self, call) -> bool:
        """Check `x.size()` or `x.size(<integer literal>)` in the host code."""
        if not _is_attr(call.func, self.input_name, "size") or call.keywords:
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
                and _is_attr(keyword.value, self.input_name, keyword.arg)
                for keyword in call.keywords
            )
        )

    def _is_grid_iterator(self, iterator, size_names: set[str]) -> bool:
        """Accept `hl.tile(m)` in `for tile_m in hl.tile(m)` when `m` came from `x.size()`."""
        return (
            isinstance(iterator, ast.Call)
            and _is_attr(iterator.func, "hl", "tile")
            and not iterator.keywords
            and len(iterator.args) == 1
            and isinstance(iterator.args[0], ast.Name)
            and iterator.args[0].id in size_names
        )

    def _block_id_of(self, value) -> int:
        """Look up the block number Helion assigned to this symbolic tile index."""
        if isinstance(value, torch.SymInt) and value.node.expr in self.block_ids:
            return self.block_ids[value.node.expr]
        refuse(f"{value} is not a tile")

    def _check_tile_extents(self, blocks: list[int], dimensions: list[str], operation: str):
        """Require each tile loop's extent to equal the size of the tensor axis it indexes.

        For `x[tile_m, tile_n]` with shape `[M, N]`, the loop extents must be
        exactly `[M, N]`. This also applies to stores.
        """
        extents = [self.sym(self.helion_env.block_sizes[block].size) for block in blocks]
        if extents != dimensions:
            refuse(f"{operation} tiles axes of size {dimensions} with loops over {extents}")

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
            if args[0] not in (self.input_name, self.output_name):
                refuse(f"host tensor {args[0]} is neither the argument nor the output")
            values[ir_node] = self.lean_input if args[0] == self.input_name else ("OUT",)
        elif op == "full":
            if args[1] != 0:
                refuse(f"full with value {args[1]}; only zero accumulators for now")
            values[ir_node] = f"(Tensor.zeros [{', '.join(self.shape(ir_node))}])"
        elif op == "_for_loop":
            values[ir_node] = self._emit_inner_loop(ir_node, values, context["tiles"])
        elif op == "getitem":
            if args[1] != 0:
                refuse("loop with more than one result")
            values[ir_node] = values[args[0]]
        elif op == "_phi":
            before, after = args[0], args[1]
            loop = after.args[0] if op_printer.op_name(after) == "getitem" else None
            if (
                loop is None
                or op_printer.op_name(loop) != "_for_loop"
                or loop.args[3][0] is not before
            ):
                refuse("phi must merge a loop's initial value with that loop's result")
            block_id = self.device_ir.graphs[loop.args[0]].block_ids[0]
            name = f"acc{block_id}"
            context["lets"].append(f"let {name} := {op_printer.unparen(values[args[1]])}")
            values[ir_node] = name
        elif op == "load":
            source, indices = args[0], args[1]
            if (
                values[source] != self.lean_input
                or (len(args) > 2 and args[2] is not None)
                or len(indices) != 2
            ):
                refuse("load must read a whole 2-D tile of the argument without an extra mask")
            row_block, column_block = (self._block_id_of(index.meta["val"]) for index in indices)
            if not {row_block, column_block} <= context["tiles"]:
                refuse("load index is not a tile of an enclosing loop")
            self._check_tile_extents([row_block, column_block], self.shape(source), "load")
            values[ir_node] = (
                f"(loadTile B{row_block} B{column_block} 0 {values[source]} "
                f"tile{row_block} tile{column_block})"
            )
        elif op == "store":
            destination, indices, value = args[0], args[1], args[2]
            if (
                context["state"] is None
                or values[destination] != ("OUT",)
                or (len(args) > 3 and args[3] is not None)
                or len(indices) != 1
            ):
                refuse("store must write the output at the grid tile, in the grid body")
            block_id = self._block_id_of(indices[0].meta["val"])
            if block_id != self.grid_block:
                refuse("store index is not the grid tile")
            self._check_tile_extents([block_id], self.shape(destination), "store")
            self.store_count += 1
            context["state"] = (
                f"(storeTile {context['state']} tile{block_id} {values[value]})"
            )
        else:
            refuse(f"unsupported op in device IR: {op}")

    def _emit_inner_loop(self, ir_node, values: dict, enclosing_tiles: set[int]) -> str:
        """Emit a Lean `forTiles` expression for an inner `hl.tile` loop that updates one value."""
        graph_id, begin, end, initial_values = ir_node.args[:4]
        loop_graph = self.device_ir.graphs[graph_id]
        if (
            len(loop_graph.block_ids) != 1
            or list(begin) != [0]
            or len(end) != 1
            or len(initial_values) != 1
        ):
            refuse("inner loop must be one hl.tile from 0 carrying one value")
        (block_id,) = loop_graph.block_ids
        extent = self.sym(end[0].meta["val"])
        if extent not in self.dim_names:
            refuse(f"inner loop runs to {extent}, not an input size")
        accumulator = f"acc{block_id}"
        placeholders = [node for node in loop_graph.graph.nodes if node.op == "placeholder"]
        if len(placeholders) != 1:
            refuse("inner loop body must read only its carried value from outside")
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
        if any(type(graph).__name__ == "ReductionLoopGraphInfo" for graph in self.device_ir.graphs):
            refuse(
                "device IR has alternative reduction-loop graphs chosen by config; "
                "only the root graph is translated"
            )
        if [len(blocks) for blocks in self.device_ir.grid_block_ids] != [1]:
            refuse(f"grid {self.device_ir.grid_block_ids}; a 1-D grid for now")
        grid_block = self.device_ir.grid_block_ids[0][0]
        self.grid_block = grid_block
        if len(self.device_ir.root_ids) != 1:
            refuse(f"device IR has {len(self.device_ir.root_ids)} root graphs; expected one")
        root = self.device_ir.graphs[self.device_ir.root_ids[0]].graph
        body = self.emit_graph(root, {}, {"tiles": {grid_block}, "state": "out", "lets": []})
        if self.store_count != 1:
            refuse(f"expected one store into {self.output_name}, found {self.store_count}")
        output_node = next(
            node
            for node in root.nodes
            if op_printer.op_name(node) == "_host_tensor" and node.args[0] == self.output_name
        )
        output_shape = ", ".join(self.shape(output_node))
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
                f"({self.lean_input} : Tensor ℝ [{', '.join(self.dim_names)}])",
            ],
            f"Tensor ℝ [{output_shape}]",
        )
        self.output_shape = self.shape(output_node)
        docstring = _doc_comment(
            f"Run the Helion `{self.fn_name}` kernel on `{self.lean_input}`, writing into the output buffer `out0`."
        )
        if any(len(line) > op_printer.WIDTH for line in docstring.splitlines()):
            docstring = _doc_comment(
                f"Run the Helion kernel on `{self.lean_input}`, writing into the output buffer `out0`."
            )
        return (
            f"-- Generated from {source} by NN/Runtime/Helion/helion_frontend.py; do not edit.\n"
            "import NN.Runtime.Helion.Tile\n\n"
            "open Spec TorchLean TorchLean.Helion\n\n"
            f"namespace {name}\n\n"
            f"{docstring}\n"
            f"{signature}\n"
            f"  forTiles {self.sym(self.helion_env.block_sizes[grid_block].size)} "
            f"B{grid_block} out0 fun tile{grid_block} out =>\n"
            f"{op_printer.wrap_lines(_indent(body, 4))}\n\n"
            f"end {name}\n"
        )
