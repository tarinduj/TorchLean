"""Check the supported Helion host-code pattern and translate device IR into Lean."""
from __future__ import annotations

import ast
import dataclasses
import enum
import pathlib
import sys
import textwrap

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))  # the sibling module `fx_emitter`
import fx_emitter  # noqa: E402
from fx_emitter import App, Atom, Context, Element, Kind, LoweredValue, Tup  # noqa: E402

refuse, invalid = fx_emitter.refuse, fx_emitter.invalid


class Role(enum.Enum):
    """Which side of the goal a predicate goes on."""
    CONDITION = enum.auto()  # assumed by the goal
    OBLIGATION = enum.auto()  # proved by the goal


@dataclasses.dataclass
class Predicate:
    """A Lean predicate over sizes that collects facts, e.g. `InBounds d0 d1 d2 d3 := d0 ≤ d2 ∧ d1 ≤ d3`."""
    name: str
    doc: str
    role: Role
    params: list[str]
    facts: list[str] = dataclasses.field(default_factory=list)  # e.g. `d0 ≤ d2`, each once

    def add(self, fact: str):
        if fact not in self.facts:
            self.facts.append(fact)


def _indent(text: str, spaces: int) -> str:
    return "\n".join(" " * spaces + line for line in text.splitlines())


def _component(name: str, i: int, count: int) -> str:
    """Print how to read result `i` from a loop's final value.

    Python:
        for tile_n in hl.tile(n):
            acc_x = acc_x + x[tile_m, tile_n].sum(1)
            acc_y = acc_y + y[tile_m, tile_n].sum(1)

    Captured graph:
        loop = _for_loop(body, [0], [n], [acc_x, acc_y])   # body returns [next_x, next_y]
        getitem(loop, 0)
        getitem(loop, 1)

    Lean:
        let acc1 := forTiles ... (zeros, zeros) ...
        acc1.1   -- final acc_x
        acc1.2   -- final acc_y

    Lean has only pairs, so a triple `(a, b, c)` is really `(a, (b, c))`, and its parts are
    `.1`, `.2.1` and `.2.2`.
    """
    if count == 1:
        return name
    return name + ".2" * i + ("" if i == count - 1 else ".1")


def _tile_args(blocks: list[int], bounds: list[str]) -> tuple[Atom, Atom]:
    """Return a tile's block sizes and loop bounds, the first two arguments of the tile ops: `[B0, B1]`, `![d0, d1]`."""
    return Atom(f"[{', '.join(f'B{b}' for b in blocks)}]"), Atom(f"![{', '.join(bounds)}]")


def _tiles(tiles: list[str]) -> Atom:
    """Return a tile's index on each axis: `![tile0, 0]`."""
    return Atom(f"![{', '.join(tiles)}]")


def _doc_comment(text: str) -> str:
    """Lay out `text` as a Lean doc comment whose lines fit in the printer's width."""
    lines = textwrap.wrap(
        text, width=fx_emitter.WIDTH - 7, break_long_words=False, break_on_hyphens=False
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


class Emitter(fx_emitter.FxEmitter):
    """Check a captured Helion kernel and print its device IR as a Lean definition."""

    size_ops = (*fx_emitter.FxEmitter.size_ops, "_get_symnode")

    def __init__(self, kernel, example_args: tuple, bridge):
        """Capture device IR and validate the inputs and host-code pattern.

        Require tensor arguments with symbolic sizes, named by `fx_emitter.input_name` and `size_name`.
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
        self.lean_inputs = {name: fx_emitter.input_name(i) for i, name in enumerate(params)}
        # Name each size symbol by first appearance; sizes Helion shares between inputs share a name.
        dimension_names, self.input_shapes = {}, {}
        for name, fake in zip(params, bound_kernel.fake_args):
            if not isinstance(fake, torch.Tensor):
                refuse("only tensor parameters are supported")
            if not all(isinstance(size, torch.SymInt) for size in fake.shape):
                refuse("only symbolic sizes are supported")
            for size in fake.shape:
                dimension_names.setdefault(size.node.expr, fx_emitter.size_name(len(dimension_names)))
            self.input_shapes[self.lean_inputs[name]] = [dimension_names[size.node.expr] for size in fake.shape]
        self.dim_names = list(dimension_names.values())
        self.block_ids = {
            block.var.node.expr: block.block_id for block in self.helion_env.block_sizes
        }
        block_names = {symbol: f"B{block_id}" for symbol, block_id in self.block_ids.items()}
        super().__init__(bridge, dimension_names | block_names)
        self.output_name = self._validate_host_and_find_output()
        self.store_count = 0
        sizes = self.dim_names + [f"B{b}" for b in sorted(self.block_ids.values())]
        # Axis size ≤ block size, for every full slice (`:`).
        self.full_slices = Predicate(
            "FullSliceCovered", "Each full slice (`:`) fits in one block.", Role.CONDITION, sizes)
        # Loop bound ≤ tensor size, for every load and store; trivial ones omitted.
        self.in_bounds = Predicate(
            "InBounds", "Every lane the loads' and stores' masks enable lies inside its tensor.",
            Role.OBLIGATION, self.dim_names)
        # Loops that must run at least once, as `0 < extent` and `0 < block`.
        self.nonempty_loops = Predicate(
            "LoopsNonempty", "Each loop that turns a -inf start into a finite value runs at least once.",
            Role.OBLIGATION, sizes)
        self.predicates = [self.full_slices, self.in_bounds, self.nonempty_loops]
        self.carried_arguments = self._carried_arguments()

    def _loop_results(self, loop) -> list:
        """Return what a `_for_loop`'s body returns, one value per loop result, e.g. `[new_acc]`."""
        body_graph = self.device_ir.graphs[loop.args[0]].graph
        return body_graph.output_node().args[0]

    def _carried_arguments(self) -> dict:
        """Map each loop's result indices to the argument indices they update.

        Python:
            for tile_n in hl.tile(n):
                acc = x[tile_m, tile_n].sum(1) * scale + acc

        Captured graphs:
            body(scale, acc):
                new_acc = add(mul(sum(load(x, ...)), scale), acc)
                return [new_acc]

            outer:
                loop = _for_loop(body, [0], [n], [scale, acc])
                result = getitem(loop, 0)
                acc_after = _phi(acc, result)

        Returns `{loop: [1]}`, since `acc` is argument 1.
        """
        carries = {}
        # One slot per loop result.
        for info in self.device_ir.graphs:
            for loop in info.graph.nodes:
                if fx_emitter.op_name(loop) == "_for_loop":
                    carries[loop] = [None] * len(self._loop_results(loop))
        # Fill each slot from its phi.
        for info in self.device_ir.graphs:
            for phi in info.graph.nodes:
                if fx_emitter.op_name(phi) != "_phi":
                    continue
                acc, result = phi.args
                if fx_emitter.op_name(result) != "getitem" or fx_emitter.op_name(result.args[0]) != "_for_loop":
                    continue
                loop, index = result.args
                arguments = loop.args[3]
                if acc not in arguments or carries[loop][index] is not None:
                    invalid("expected one phi per loop result")
                carries[loop][index] = arguments.index(acc)
        # Every slot must be filled.
        if any(None in carried for carried in carries.values()):
            invalid("expected one phi per loop result")
        return carries

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
            if bound != size:
                self.in_bounds.add(f"{bound} ≤ {size}")

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
                self.full_slices.add(f"{bound} ≤ B{block}")
                tiles.append("0")
            else:
                tiles.append(f"tile{block}")
        return blocks, self._loop_bounds(blocks), tiles

    def _indexed_tile_layout(
        self, ir_node, indices, ctx: Context
    ) -> tuple[list[int], list[str], list[str]]:
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
            elif self._block_id_of(index.meta["val"]) != block or block not in ctx.nest:
                invalid("index is not the tile of an enclosing loop for that axis")
        return blocks, bounds, tiles

    def aten(self, ir_node, env: dict) -> LoweredValue:
        """Translate ops on extended reals and masked `amax`; send other ATen ops to the shared printer."""
        if any(env[operand].element is not Element.REAL for operand in ir_node.all_input_nodes):
            return self._ereal_op(ir_node, env)
        elif fx_emitter.op_name(ir_node) == "amax.default":
            return self._masked_amax(ir_node, env)
        else:
            return super().aten(ir_node, env)

    def _masked_amax(self, ir_node, env: dict) -> LoweredValue:
        """Print `amax` over a tile masked to -inf as `maskedMax`.

        Python:
            x[tile_m, :].amax(1)

        Captured graph:
            amax(_mask_to(load(x, [block_size_0, block_size_1]), -inf), [1])

        Lean:
            maskedMax [B0, B1] ![d0, d1] ![tile0, 0] 1 (loadTile …)
        """
        operand = ir_node.args[0]
        # Accept only an amax whose operand is `_mask_to(…, -inf)`: ℝ has no -inf, so the `_mask_to`
        # and the `amax` become one `maskedMax`.
        if (
            not isinstance(operand, torch.fx.Node)
            or fx_emitter.op_name(operand) != "_mask_to"
            or operand.args[1] != float("-inf")
        ):
            refuse("only amax over an input masked to -inf is supported")
        kind = fx_emitter.amax_kind(ir_node)
        axis, keepdim = kind["axes"][0], kind["keepdim"]
        blocks, bounds, tiles = self._tile_layout(operand)
        original = env[operand.args[0]].expr  # The tile before `_mask_to`; `maskedMax` skips its masked lanes.
        call = App("maskedMax", (*_tile_args(blocks, bounds), _tiles(tiles), Atom(str(axis)), original))
        # Restore the singleton axis when keepdim=True.
        return LoweredValue(self.reshape(call, self.shape(ir_node)) if keepdim else call, Kind.COMPUTED)

    def _ereal_op(self, ir_node, env: dict) -> LoweredValue:
        """Print an op on extended reals: `maximum` and subtraction stay in `EReal`, `exp` returns to `ℝ`.

        Python:
            mx = hl.full([tile_m, tile_n], float("-inf"), dtype=x.dtype)
            mx = torch.maximum(mx, x[tile_m, tile_n])
            out[tile_m, tile_n] = torch.exp(mx)

        Captured graph:
            mx = full([block_size_0, block_size_1], -inf)
            mx_1 = maximum(mx, load(x, [block_size_0, block_size_1]))
            exp = exp(mx_1)

        Lean:
            expE (Tensor.map2Spec max (Tensor.full [B0, B1] (⊥ : EReal)) (toEReal (loadTile …)))
        """
        op = fx_emitter.op_name(ir_node)
        # Promote real operands; the result stays extended.
        promoted = {
            arg: env[arg] if env[arg].element is not Element.REAL
            else LoweredValue(App("toEReal", (env[arg].expr,)), Kind.COMPUTED, element=Element.EREAL_FINITE)
            for arg in ir_node.args}
        if op == "exp.default":
            return LoweredValue(App("expE", (env[ir_node.args[0]].expr,)), Kind.COMPUTED)
        elif op == "maximum.default":
            operands = self.elementwise_operands(ir_node, ir_node.args, promoted)
            # The max is `⊥` only where every operand is.
            all_may_be_bottom = all(promoted[arg].element is Element.EREAL_MAY_BE_BOTTOM for arg in ir_node.args)
            element = Element.EREAL_MAY_BE_BOTTOM if all_may_be_bottom else Element.EREAL_FINITE
            return LoweredValue(App("Tensor.map2Spec", (Atom("max"), *operands)), Kind.COMPUTED, element=element)
        elif op == "sub.Tensor":
            minuend, subtrahend = ir_node.args
            # A finite value minus `⊥` is `⊤`, which `expE` gets wrong.
            if env[subtrahend].element is Element.EREAL_MAY_BE_BOTTOM:
                refuse("only subtracting values that cannot be -inf is supported")
            operands = self.elementwise_operands(ir_node, ir_node.args, promoted)
            return LoweredValue(App("Tensor.map2Spec", (Atom("(· - ·)"), *operands)), Kind.COMPUTED,
                                element=promoted[minuend].element)
        else:
            refuse("extended reals are supported only in maximum, subtraction and exp")

    def node(self, ir_node, env: dict, ctx: Context) -> LoweredValue | None:
        """Translate a Helion-specific IR node into Lean; return its value, or None for a store.

        Tile loops, carried values, loads, and stores are handled here. ATen
        tensor operations go through the shared FX walker and op printer.
        """
        args = ir_node.args
        op = fx_emitter.op_name(ir_node)
        if op == "_new_var":
            return env[args[0]]
        elif op == "_host_tensor":
            if args[0] not in self.lean_inputs and args[0] != self.output_name:
                refuse("only the arguments and the output are supported in device code")
            if args[0] in self.lean_inputs:
                return LoweredValue(Atom(self.lean_inputs[args[0]]), Kind.INPUT)
            else:
                return LoweredValue(None, Kind.OUTPUT)
        elif op == "full":
            shape = Atom(f"[{', '.join(self.shape(ir_node))}]")
            if args[1] == 0:
                return LoweredValue(App("Tensor.zeros", (shape,)), Kind.COMPUTED)
            elif args[1] == float("-inf"):
                # ℝ has no -inf: it is the extended real `⊥`.
                return LoweredValue(App("Tensor.full", (shape, Atom("(⊥ : EReal)"))), Kind.COMPUTED,
                                            element=Element.EREAL_MAY_BE_BOTTOM)
            else:
                refuse("only 0 and -inf fills are supported")
        elif op == "subscript":
            # `v[:, None]` adds a singleton axis: reshape to the captured shape.
            if not all(index == slice(None) or index is None for index in args[1]):
                refuse("only subscripts with `:` and `None` are supported")
            reshaped = self.reshape(env[args[0]].expr, self.shape(ir_node))
            return dataclasses.replace(env[args[0]], expr=reshaped)  # same entries, new shape
        elif op == "_for_loop":
            return self._emit_inner_loop(ir_node, env, ctx)
        elif op == "getitem":
            if fx_emitter.op_name(args[0]) != "_for_loop":
                refuse("only getitem of a loop's result is supported")
            loop, index = args
            return env[loop].parts[index]
        elif op == "_phi":
            before, after = args
            if fx_emitter.op_name(after) != "getitem" or fx_emitter.op_name(after.args[0]) != "_for_loop":
                refuse("only phis after loops are supported")
            return env[after]
        elif op == "_mask_to":
            if env[args[0]].element is not Element.REAL:
                refuse("only masking real values is supported")
            if args[1] == float("-inf"):
                users = list(ir_node.users)
                if len(users) != 1 or fx_emitter.op_name(users[0]) != "amax.default":
                    refuse("_mask_to(-inf) is supported only as the input of amax")
                return LoweredValue(None, Kind.FUSED)   # printed by its amax
            elif args[1] == 0:
                blocks, bounds, tiles = self._tile_layout(ir_node)
                return LoweredValue(
                    App("maskTo", (*_tile_args(blocks, bounds), _tiles(tiles), Atom("0"), env[args[0]].expr)),
                    Kind.COMPUTED)
            else:
                refuse("only _mask_to with fill 0 or -inf is supported")
        elif op == "load":
            source, indices = args[0], args[1]
            if env[source].kind != Kind.INPUT:
                refuse("only loads from an argument are supported")
            if len(args) > 2 and args[2] is not None:
                refuse("only loads without an extra mask are supported")
            blocks, bounds, tiles = self._indexed_tile_layout(ir_node, indices, ctx)
            self._record_in_bounds(bounds, self.shape(source))
            return LoweredValue(
                App("loadTile", (*_tile_args(blocks, bounds), Atom("0"), env[source].expr, _tiles(tiles))),
                Kind.COMPUTED)
        elif op == "store":
            destination, indices, value = args[0], args[1], args[2]
            if ctx.state is None:
                refuse("only stores outside loops that carry a value are supported")
            if env[destination].kind != Kind.OUTPUT:
                refuse("only stores into the output are supported")
            if len(args) > 3 and args[3] is not None:
                refuse("only stores without an extra mask are supported")
            if env[value].element is not Element.REAL:
                refuse("only storing real values is supported")
            blocks, bounds, tiles = self._indexed_tile_layout(value, indices, ctx)
            if [b for b, t in zip(blocks, tiles) if t != "0"] != ctx.nest:
                refuse("only stores at the enclosing loops' tiles, in order, are supported")
            self.output_shape = self.shape(destination)
            self._record_in_bounds(bounds, self.output_shape)
            self.store_count += 1
            ctx.state = (
                App("storeTile",
                    (*_tile_args(blocks, bounds), Atom(ctx.state), _tiles(tiles), env[value].expr)).show_as_argument()
            )
        else:
            refuse(f"device op {op} is not supported")

    def _emit_inner_loop(self, ir_node, env: dict, ctx: Context) -> LoweredValue | None:
        """Emit an inner `hl.tile` loop as `forTiles`; return a carrying loop's tuple, or None for a storing loop."""
        graph_id, begin, end, initial_values = ir_node.args[:4]
        loop_graph = self.device_ir.graphs[graph_id]
        if len(loop_graph.block_ids) != 1 or len(end) != 1:
            refuse("only inner loops over one axis are supported")
        if list(begin) != [0]:
            refuse("only inner loops from 0 are supported")
        (block_id,) = loop_graph.block_ids
        extent = self.sym(end[0].meta["val"])
        if extent not in self.dim_names:
            refuse("only inner loops up to an input size are supported")
        placeholders = loop_graph.graph.find_nodes(op="placeholder")
        if len(placeholders) != len(initial_values):
            invalid(f"inner loop body has {len(placeholders)} inputs but the loop passes {len(initial_values)}")
        nest = ctx.nest + [block_id]
        # The loop's arguments are the outside values its body reads; it returns the ones it updates.
        # The arguments the loop updates; the others are only read.
        carried = self.carried_arguments[ir_node]
        arguments = {p: env[v] for p, v in zip(placeholders, initial_values)}
        if len(carried) == 0:
            # A storing loop: each iteration writes into the output and only reads its arguments.
            if ctx.state is None:
                refuse("only storing loops outside loops that carry a value are supported")
            body = self.emit_graph(
                loop_graph.graph, {**env, **arguments}, Context(nest=nest, state="out")
            )
            ctx.state = (
                f"(forTiles {extent} B{block_id} {ctx.state} fun tile{block_id} out =>\n"
                f"{_indent(body.code, 2)})"
            )
        else:
            # A carrying loop, e.g. an accumulator.
            accumulator = f"acc{block_id}"
            count = len(carried)
            # In the body, each carried value has its start's entries.
            starts = [env[initial_values[a]].element for a in carried]
            for i, (argument, start) in enumerate(zip(carried, starts)):
                arguments[placeholders[argument]] = LoweredValue(
                    Atom(_component(accumulator, i, count)), Kind.COMPUTED, element=start)
            initial = Tup(tuple(env[initial_values[a]].expr for a in carried)).show()
            body = self.emit_graph(loop_graph.graph, {**env, **arguments}, Context(nest=nest))
            ends = [result.element for result in body.results]
            for start, end in zip(starts, ends):
                if (start is Element.REAL) != (end is Element.REAL):
                    refuse("only carried values that stay real or stay extended are supported")
                if start is Element.EREAL_FINITE and end is Element.EREAL_MAY_BE_BOTTOM:
                    refuse("only loops that keep a finite carried value finite are supported")
                # A `⊥` start ends finite only if the loop runs, e.g. a running max from -inf.
                if start is Element.EREAL_MAY_BE_BOTTOM and end is Element.EREAL_FINITE:
                    self.nonempty_loops.add(f"0 < {extent}")
                    self.nonempty_loops.add(f"0 < B{block_id}")
            ctx.lets.append(
                f"let {accumulator} := forTiles {extent} B{block_id} {initial} "
                f"fun tile{block_id} {accumulator} =>\n{_indent(body.code, 2)}"
            )
            # After the loop, each carried value has the entries its body produces.
            parts = tuple(LoweredValue(Atom(_component(accumulator, i, count)), Kind.COMPUTED, element=end)
                          for i, end in enumerate(ends))
            return LoweredValue(Atom(accumulator), Kind.TUPLE, parts)

    def emit(self, name: str, source: str) -> str:
        """Emit the complete Lean module for the outer `hl.tile` grid loop and its body."""
        if len(self.device_ir.grid_block_ids) != 1:
            refuse("only one grid loop is supported")
        self.grid_blocks = list(self.device_ir.grid_block_ids[0])
        if len(self.device_ir.root_ids) != 1:
            refuse("only one root graph is supported")
        root = self.device_ir.graphs[self.device_ir.root_ids[0]].graph
        body = self.emit_graph(root, {}, Context(nest=list(self.grid_blocks), state="out"))
        # Only the root graph, where each full slice is one block, is translated. Helion runs it
        # for configs whose `reduction_loops` entries are `None`; other configs run the
        # reduction-loop graphs, which split full slices into chunks.
        if (
            any(type(graph).__name__ == "ReductionLoopGraphInfo" for graph in self.device_ir.graphs)
            and not self.full_slices.facts
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
        signature = fx_emitter.print_def(
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
        scope = " This is Helion's root device IR, run by configs without reduction loops." if self.full_slices.facts else ""
        docstring = _doc_comment(
            f"Run the Helion `{self.fn_name}` kernel on {inputs}, writing into the output buffer `out0`.{scope}"
        )
        if any(len(line) > fx_emitter.WIDTH for line in docstring.splitlines()):
            docstring = _doc_comment(
                f"Run the Helion kernel on {inputs}, writing into the output buffer `out0`.{scope}"
            )
        return (
            f"-- Generated from {source} by NN/Runtime/Helion/helion_emitter.py; do not edit.\n"
            "import NN.Runtime.Helion.Tile\n\n"
            "open Spec TorchLean TorchLean.Helion\n\n"
            f"namespace {name}\n\n"
            f"{docstring}\n"
            f"{signature}\n"
            f"{grid_loops}"
            f"{fx_emitter.wrap_lines(_indent(body.code, 2 * len(self.grid_blocks) + 2))}\n\n"
            f"end {name}\n"
        )
