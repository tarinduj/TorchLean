"""Translate supported FX graphs into Lean calls over TorchLean tensors."""
from __future__ import annotations

import torch

SYM_OPS = ("sym_size.int", "sym_size")
WIDTH = 100


def input_name(i: int) -> str:
    """Name the i-th input: x0, x1, …"""
    return f"x{i}"


def size_name(i: int) -> str:
    """Name the i-th size: d0, d1, …"""
    return f"d{i}"


# The bridge targets TorchLean's graph IR, which has no division operator.
# Our direct translation supports tensor division through `Tensor.divSpec`.
_DIRECT_ONLY_KINDS = {"div.Tensor": {"kind": "div"}}
# Binary elementwise kinds and their TorchLean ops; operands are broadcast to the output shape.
_BINARY_ELEMENTWISE_KINDS = {
    "add": "addSpec", "sub": "subSpec", "mul_elem": "mulSpec", "div": "divSpec", "max_elem": "maxSpec"}


def refuse(msg: str):
    """Stop on input that is valid but not supported yet."""
    raise NotImplementedError(msg)


def invalid(msg: str):
    """Stop on input that is broken, or that breaks an assumption the translator relies on."""
    raise ValueError(msg)


def op_name(node) -> str:
    return getattr(node.target, "__name__", str(node.target))


def unparen(e: str) -> str:
    return e[1:-1] if e.startswith("(") and e.endswith(")") else e


def _split_top(expr: str) -> list[str]:
    """Split `expr` at spaces outside brackets."""
    parts, cur, depth = [], "", 0
    for ch in expr:
        depth += ch in "([{⟨"
        depth -= ch in ")]}⟩"
        if ch == " " and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return [p for p in parts + [cur] if p]


def _outer_parens(expr: str) -> bool:
    """Return whether `expr` is one parenthesized term."""
    if not (expr.startswith("(") and expr.endswith(")")):
        return False
    depth = 0
    for k, ch in enumerate(expr):
        depth += ch in "([{⟨"
        depth -= ch in ")]}⟩"
        if depth == 0 and k < len(expr) - 1:
            return False
    return True


def _tuple_components(expr: str) -> list[str]:
    """Split printed tuple text into its values, e.g. `"(a, b, c)"` → `["a", "b", "c"]`."""
    if not _outer_parens(expr):
        return [expr]
    parts, cur, depth = [], "", 0
    for ch in expr[1:-1]:
        depth += ch in "([{⟨"
        depth -= ch in ")]}⟩"
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    return parts + [cur.strip()]


def wrap_expr(expr: str, indent: int) -> list[str]:
    """Lay out an application in lines of at most WIDTH, putting arguments that do not fit below."""
    pad = " " * indent
    if indent + len(expr) <= WIDTH:
        return [pad + expr]
    components = _tuple_components(expr)
    if len(components) > 1:
        # A tuple: each component starts a line.
        lines = []
        for component in components:
            lines += wrap_expr(component, indent + 1)
            lines[-1] += ","
        lines[0] = pad + "(" + lines[0].lstrip()
        lines[-1] = lines[-1][:-1] + ")"
        return lines
    if _outer_parens(expr):
        lines = wrap_expr(expr[1:-1], indent + 1)
        lines = [pad + "(" + lines[0].lstrip()] + [ln[1:] for ln in lines[1:]]
        lines[-1] += ")"
        return lines
    head, *args = _split_top(expr)
    first = head
    while args and indent + len(first) + 1 + len(args[0]) <= WIDTH:
        first += " " + args.pop(0)
    return [pad + first] + [ln for a in args for ln in wrap_expr(a, indent + 2)]


def wrap_lines(text: str) -> str:
    """Wrap each over-long application line of `text`, and the value of an over-long `let`; leave
    other binding lines unchanged."""
    out = []
    for ln in text.splitlines():
        body = ln.lstrip()
        indent = len(ln) - len(body)
        head, sep, value = body.partition(" := ")
        if len(ln) > WIDTH and body.startswith("let ") and sep and " " not in head[4:]:
            out += [" " * indent + head + " :="] + wrap_expr(unparen(value), indent + 2)
            continue
        binding = any(t in ("let", "fun", ":=", "=>") for t in _split_top(body))
        out += [ln] if len(ln) <= WIDTH or binding else wrap_expr(body, indent)
    return "\n".join(out)


def print_def(head: str, binders: list[str], result: str) -> str:
    """Lay out `head binders : result :=`, continuing long signatures on lines indented four."""
    lines, cur = [], head
    for part in binders[:-1] + [f"{binders[-1]} : {result} :="]:
        if len(cur) + 1 + len(part) <= WIDTH:
            cur += " " + part
        else:
            lines.append(cur)
            cur = "    " + part
    return "\n".join(lines + [cur])


def print_op(kind: dict, args: list[str], shapes: list[list[str]]) -> tuple[str, set[str]]:
    """Print one normalized op and return the dimensions it requires to be positive."""
    if kind["kind"] in _BINARY_ELEMENTWISE_KINDS and len(kind) == 1:
        if len(args) != 2:
            refuse(f"{kind['kind']} with a scalar operand is not supported")
        return f"(Tensor.{_BINARY_ELEMENTWISE_KINDS[kind['kind']]} {args[0]} {args[1]})", set()
    if kind == {"kind": "relu"}:
        return f"(Activation.reluSpec {args[0]})", set()
    if kind == {"kind": "exp"}:
        return f"(Tensor.expSpec {args[0]})", set()
    if kind == {"kind": "softmax", "axis": 1} and len(shapes[0]) == 2:
        return f"(Activation.softmaxSpec 1 {args[0]})", set()
    if kind == {"kind": "reduce_sum", "axes": [1], "keepdim": False} and len(shapes[0]) == 2:
        return f"(Tensor.reduceSum 1 {args[0]} (.succ (Shape.hasNonemptyAxisZeroOfPos h{shapes[0][1]}).proof))", {shapes[0][1]}
    if kind == {"kind": "reduce_max", "axes": [1], "keepdim": False} and len(shapes[0]) == 2:
        return f"(Tensor.reduceMax 1 {args[0]} (.succ (Shape.hasNonemptyAxisZeroOfPos h{shapes[0][1]}).proof))", {shapes[0][1]}
    refuse(f"op {kind} is not supported")


def amax_kind(n) -> dict:
    """Read `amax` as a max over one axis.

    `x.amax(dim=-1, keepdim=True)` of a matrix → `{"kind": "reduce_max", "axes": [1], "keepdim": True}`
    """
    dims = n.args[1] if len(n.args) > 1 else n.kwargs.get("dim", [])
    keepdim = n.args[2] if len(n.args) > 2 else n.kwargs.get("keepdim", False)
    rank = len(n.args[0].meta["val"].shape)
    if len(dims) != 1:
        refuse("only amax over one axis is supported")
    if not -rank <= dims[0] < rank:
        invalid(f"expected an amax axis in [-{rank}, {rank})")
    return {"kind": "reduce_max", "axes": [dims[0] % rank], "keepdim": bool(keepdim)}


class FxEmitter:
    """Walk an FX graph, printing ATen nodes; subclasses handle other node kinds."""

    def __init__(self, bridge, names: dict):
        self.bridge, self.names, self.hyps = bridge, names, set()
        # _lower_kind uses this map when interpreting symbolic dimensions.
        bridge._DIM_NAMES = dict(names)

    def sym(self, v) -> str:
        """Resolve a captured size symbol to its Lean parameter name."""
        if isinstance(v, torch.SymInt) and v.node.expr in self.names:
            return self.names[v.node.expr]
        refuse(f"tensor size {v} is not supported")

    def shape(self, node) -> list[str]:
        """Read a node's tensor shape as Lean dimension names, or `1` for a singleton axis."""
        return ["1" if isinstance(s, int) and s == 1 else self.sym(s) for s in node.meta["val"].shape]

    def aten(self, n, env) -> str:
        """Lower an ATen node with the bridge, then print its TorchLean tensor call.

        Each value keeps its captured shape: an elementwise operand is broadcast to the node's
        shape, and a `keepdim` reduction is reshaped to it.
        """
        operands = self.bridge._node_refs((n.args, n.kwargs), {m: m for m in n.all_input_nodes})
        if op_name(n) == "amax.default":
            # The IR has no max over an axis either; it prints as `Tensor.reduceMax`.
            kind = amax_kind(n)
        elif op_name(n) in _DIRECT_ONLY_KINDS:
            kind = _DIRECT_ONLY_KINDS[op_name(n)]
        else:
            kind = self.bridge._lower_kind(n)
        if kind["kind"] in _BINARY_ELEMENTWISE_KINDS:
            args = self.elementwise_operands(n, operands, env)
            shapes = [self.shape(n)] * len(operands)
        else:
            args, shapes = [env[p] for p in operands], [self.shape(p) for p in operands]
        # TorchLean reductions drop the axis; reshape below restores it for keepdim=True.
        keepdim = bool(kind.get("keepdim"))
        if keepdim:
            kind = {**kind, "keepdim": False}
        call, positive = print_op(kind, args, shapes)
        self.hyps |= positive
        return self.reshape(call, self.shape(n)) if keepdim else call

    def elementwise_operands(self, n, operands, env: dict) -> list[str]:
        """Broadcast each operand of elementwise `n`, as printed in `env`, to `n`'s shape."""
        target = self.shape(n)
        return [self.broadcast(env[p], self.shape(p), target) for p in operands]

    def broadcast(self, value: str, shape: list[str], target: list[str]) -> str:
        """Broadcast `value` between equal-rank shapes; each axis of `shape` must match `target` or be `1`."""
        if shape == target:
            return value
        if len(shape) != len(target):
            refuse(f"broadcasting {shape} to {target} is not supported")
        if any(s not in (t, "1") for s, t in zip(shape, target)):
            invalid(f"cannot broadcast {shape} to {target}")
        return f"(Tensor.broadcastTo (s₂ := [{', '.join(target)}]) (by simp [Shape.CanBroadcastTo]) {value})"

    def reshape(self, value: str, target: list[str]) -> str:
        """Reshape `value` to `target`, e.g. a reduction's result to its `keepdim` shape."""
        return f"(Tensor.reshapeSpec (target := [{', '.join(target)}]) {value} (by simp [Shape.size]))"

    def node(self, n, env: dict, ctx: dict):
        """Refuse non-ATen nodes unless a subclass supplies their translation."""
        refuse(f"op {op_name(n)} is not supported")

    def emit_graph(self, graph, env: dict, ctx: dict) -> str:
        """Return Lean code for a graph; `env` holds node values and `ctx` collects lets and output state."""
        result = None
        for n in graph.nodes:
            if n.op == "placeholder":
                continue
            if n.op == "output":
                outs = n.args[0]
                if outs is not None and len(outs) > 0:
                    # Several results form a tuple, e.g. the values a loop carries.
                    result = unparen(env[outs[0]]) if len(outs) == 1 else (
                        f"({', '.join(unparen(env[out]) for out in outs)})")
            elif isinstance(n.target, torch._ops.OpOverload) and op_name(n) not in SYM_OPS:
                env[n] = self.aten(n, env)
                if len(n.users) > 1:
                    ctx["lets"].append(f"let v_{n.name} := {unparen(env[n])}")
                    env[n] = f"v_{n.name}"
            elif op_name(n) in SYM_OPS:
                env[n] = self.sym(n.meta["val"])
            else:
                self.node(n, env, ctx)
        return "\n".join(ctx["lets"] + [unparen(ctx["state"]) if result is None else result])
