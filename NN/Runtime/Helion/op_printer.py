"""Translate supported FX graphs into Lean calls over TorchLean tensors."""
from __future__ import annotations

import torch

SYM_OPS = ("sym_size.int", "sym_size")
WIDTH = 100


def refuse(msg: str):
    raise NotImplementedError(msg)


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


def wrap_expr(expr: str, indent: int) -> list[str]:
    """Lay out an application in lines of at most WIDTH, putting arguments that do not fit below."""
    pad = " " * indent
    if indent + len(expr) <= WIDTH:
        return [pad + expr]
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
    """Wrap each over-long plain application line of `text`; leave binding lines unchanged."""
    out = []
    for ln in text.splitlines():
        body = ln.lstrip()
        binding = any(t in f" {body} " for t in (" let ", " fun ", " := ", " => "))
        out += [ln] if len(ln) <= WIDTH or binding else wrap_expr(body, len(ln) - len(body))
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
    if kind == {"kind": "add"}:
        if shapes[0] != shapes[1]:
            refuse("not implemented: add with broadcasting")
        return f"(Tensor.addSpec {args[0]} {args[1]})", set()
    if kind == {"kind": "relu"}:
        return f"(Activation.reluSpec {args[0]})", set()
    if kind == {"kind": "reduce_sum", "axes": [1], "keepdim": False} and len(shapes[0]) == 2:
        return f"(Tensor.reduceSum 1 {args[0]} (.succ (Shape.hasNonemptyAxisZeroOfPos h{shapes[0][1]}).proof))", {shapes[0][1]}
    refuse(f"not implemented: {kind}")


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
        refuse(f"size {v} is not a named size")

    def shape(self, node) -> list[str]:
        """Read a node's tensor shape as Lean dimension names."""
        return [self.sym(s) for s in node.meta["val"].shape]

    def aten(self, n, env) -> str:
        """Lower an ATen node with the bridge, then print its TorchLean tensor call."""
        operands = self.bridge._node_refs((n.args, n.kwargs), {m: m for m in n.all_input_nodes})
        call, positive = print_op(self.bridge._lower_kind(n), [env[p] for p in operands], [self.shape(p) for p in operands])
        self.hyps |= positive
        return call

    def node(self, n, env: dict, ctx: dict):
        """Refuse non-ATen nodes unless a subclass supplies their translation."""
        refuse(f"unsupported op: {op_name(n)}")

    def emit_graph(self, graph, env: dict, ctx: dict) -> str:
        """Return Lean code for a graph; `env` holds node values and `ctx` collects lets and output state."""
        result = None
        for n in graph.nodes:
            if n.op == "placeholder":
                continue
            if n.op == "output":
                outs = n.args[0]
                if outs is not None:
                    if not (isinstance(outs, (list, tuple)) and len(outs) == 1):
                        refuse("graph must return one tensor")
                    result = env[outs[0]]
            elif isinstance(n.target, torch._ops.OpOverload) and op_name(n) not in SYM_OPS:
                env[n] = self.aten(n, env)
                if len(n.users) > 1:
                    ctx["lets"].append(f"let v_{n.name} := {unparen(env[n])}")
                    env[n] = f"v_{n.name}"
            elif op_name(n) in SYM_OPS:
                env[n] = self.sym(n.meta["val"])
            else:
                self.node(n, env, ctx)
        return "\n".join(ctx["lets"] + [unparen(ctx["state"] if result is None else result)])
