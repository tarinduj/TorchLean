"""Translate supported FX graphs into Lean calls over TorchLean tensors."""
from __future__ import annotations

import abc
import dataclasses
import enum
import fractions

import torch

WIDTH = 100


def input_name(i: int) -> str:
    """Name the i-th input: x0, x1, …"""
    return f"x{i}"


def size_name(i: int) -> str:
    """Name the i-th size: d0, d1, …"""
    return f"d{i}"


# The bridge targets TorchLean's graph IR, which has no division operator.
# Our direct translation supports tensor division through `Tensor.divSpec`.
# Binary elementwise ATen ops and their TorchLean ops; each operand, a node or a number, is brought to
# the output shape.
_ELEMENTWISE_OPS = {
    "add.Tensor": "Tensor.addSpec", "sub.Tensor": "Tensor.subSpec", "mul.Tensor": "Tensor.mulSpec",
    "div.Tensor": "Tensor.divSpec", "maximum.default": "Tensor.maxSpec"}


def refuse(msg: str):
    """Stop on input that is valid but not supported yet."""
    raise NotImplementedError(msg)


def invalid(msg: str):
    """Stop on input that is broken, or that breaks an assumption the translator relies on."""
    raise ValueError(msg)


def op_name(node) -> str:
    return getattr(node.target, "__name__", str(node.target))


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


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def real_literal(value: int | float) -> Atom:
    """Print a Python number as the real it is exactly: `0.125` → `(1 / 8 : ℝ)`."""
    exact = fractions.Fraction(value)
    if exact.denominator == 1:
        return Atom(f"({exact.numerator} : ℝ)")
    return Atom(f"({exact.numerator} / {exact.denominator} : ℝ)")


def _nonempty_axis(size: str) -> Atom:
    """Prove a reduced axis is nonempty from its size's positivity hypothesis."""
    return Atom(f"(.succ (Shape.hasNonemptyAxisZeroOfPos h{size}).proof)")


def lower_op(kind: dict, args: list[Term], shapes: list[list[str]]) -> tuple[App, set[str]]:
    """Build one normalized op's TorchLean term and the dims it requires to be nonempty, e.g. `d1` for
    a sum over axis 1."""
    if kind == {"kind": "matmul"} and len(shapes) == 2 and len(shapes[0]) == len(shapes[1]) == 2:
        return App("matMulSpec", (args[0], args[1])), set()
    if kind == {"kind": "relu"}:
        return App("Activation.reluSpec", (args[0],)), set()
    if kind == {"kind": "exp"}:
        return App("Tensor.expSpec", (args[0],)), set()
    if kind == {"kind": "softmax", "axis": 1} and len(shapes[0]) == 2:
        return App("Activation.softmaxSpec", (Atom("1"), args[0])), set()
    if kind == {"kind": "reduce_sum", "axes": [1], "keepdim": False} and len(shapes[0]) == 2:
        return App("Tensor.reduceSum", (Atom("1"), args[0], _nonempty_axis(shapes[0][1]))), {shapes[0][1]}
    if kind == {"kind": "reduce_max", "axes": [1], "keepdim": False} and len(shapes[0]) == 2:
        return App("Tensor.reduceMax", (Atom("1"), args[0], _nonempty_axis(shapes[0][1]))), {shapes[0][1]}
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


class Term(abc.ABC):
    """A Lean term."""

    @abc.abstractmethod
    def show(self) -> str:
        """Print the term on one line, apart from function bodies."""

    def show_as_argument(self) -> str:
        """Print the term as a function argument, in parentheses where it needs them."""
        return self.show()

    def layout(self, indent: int) -> list[str]:
        """Lay out the term from column `indent` in lines of at most WIDTH where it can."""
        return [" " * indent + self.show()]

    def layout_as_argument(self, indent: int) -> list[str]:
        """Lay out the term as a function argument from column `indent`."""
        return self.layout(indent)


@dataclasses.dataclass(frozen=True)
class Atom(Term):
    """Lean text printed as is: a name, a numeral, or a bracketed leaf, e.g. `x0`, `![tile0]`, `(⊥ : EReal)`."""
    text: str

    def show(self) -> str:
        return self.text


@dataclasses.dataclass(frozen=True)
class App(Term):
    """A function applied to terms: `App("Tensor.expSpec", (Atom("x0"),))` prints `Tensor.expSpec x0`."""
    head: str
    args: tuple[Term, ...]

    def show(self) -> str:
        return " ".join([self.head, *(arg.show_as_argument() for arg in self.args)])

    def show_as_argument(self) -> str:
        return f"({self.show()})"

    def layout(self, indent: int) -> list[str]:
        """Keep the head and the arguments that fit on the first line; put each other argument below,
        two deeper. A trailing `fun` stays on the first line, its body below."""
        pad = " " * indent
        if self.args and isinstance(self.args[-1], Fun):
            *args, fun = self.args
            first = " ".join([self.head, *(arg.show_as_argument() for arg in args), fun.header()])
            return [pad + first] + fun.body.layout(indent + 2)
        if indent + len(self.show()) <= WIDTH:
            return [pad + self.show()]
        first, rest = self.head, list(self.args)
        while rest and indent + len(first) + 1 + len(rest[0].show_as_argument()) <= WIDTH:
            first += " " + rest.pop(0).show_as_argument()
        return [pad + first] + [line for arg in rest for line in arg.layout_as_argument(indent + 2)]

    def layout_as_argument(self, indent: int) -> list[str]:
        if indent + len(self.show_as_argument()) <= WIDTH:
            return [" " * indent + self.show_as_argument()]
        # Lay out inside the parentheses, then open them where the term starts.
        lines = self.layout(indent + 1)
        lines = [" " * indent + "(" + lines[0].lstrip()] + [line[1:] for line in lines[1:]]
        lines[-1] += ")"
        return lines


@dataclasses.dataclass(frozen=True)
class Tup(Term):
    """A tuple of terms, e.g. the values a loop carries: `(acc1.1, v_mx)`."""
    items: tuple[Term, ...]

    def show(self) -> str:
        return f"({', '.join(item.show() for item in self.items)})"

    def layout(self, indent: int) -> list[str]:
        """Start each item on its own line when the tuple does not fit on one."""
        if indent + len(self.show()) <= WIDTH:
            return [" " * indent + self.show()]
        lines = []
        for item in self.items:
            lines += item.layout(indent + 1)
            lines[-1] += ","
        lines[0] = " " * indent + "(" + lines[0].lstrip()
        lines[-1] = lines[-1][:-1] + ")"
        return lines


@dataclasses.dataclass(frozen=True)
class Let(Term):
    """A binding and the term that uses it: `let v_mx := …` above `v_mx`."""
    name: str
    value: Term
    body: Term

    def show(self) -> str:
        return f"let {self.name} := {self.value.show()}\n{self.body.show()}"

    def layout(self, indent: int) -> list[str]:
        """Put the value after `:=` when it fits or ends in a `fun`; otherwise below, two deeper."""
        pad, head = " " * indent, f"let {self.name} :="
        if "\n" in self.value.show():
            # A value with a function body, e.g. a loop: its first line follows `:=`.
            value = self.value.layout(indent)
            binding = [pad + head + " " + value[0].lstrip()] + value[1:]
        elif indent + len(head) + 1 + len(self.value.show()) <= WIDTH:
            binding = [pad + head + " " + self.value.show()]
        else:
            binding = [pad + head] + self.value.layout(indent + 2)
        return binding + self.body.layout(indent)


@dataclasses.dataclass(frozen=True)
class Fun(Term):
    """A function whose body starts on the next line, two deeper: `fun tile1 acc1 =>` above the body."""
    params: tuple[str, ...]
    body: Term

    def header(self) -> str:
        return f"fun {' '.join(self.params)} =>"

    def show(self) -> str:
        body = "\n".join("  " + line for line in self.body.show().splitlines())
        return f"{self.header()}\n{body}"

    def layout(self, indent: int) -> list[str]:
        return [" " * indent + self.header()] + self.body.layout(indent + 2)


@dataclasses.dataclass
class Context:
    """Track the enclosing loops, current output, and generated let lines for one graph body.

    Grid body: `Context(nest=[0], state=Atom("out"), lets=[])`.
    Inner loop updating an accumulator: `Context(nest=[0, 1], state=None, lets=[])`.
    """
    nest: list[int] = dataclasses.field(default_factory=list)  # block ids of the enclosing tile loops
    state: Term | None = None  # the output as stored so far, in a body that stores
    lets: list[tuple[str, Term]] = dataclasses.field(default_factory=list)  # bindings before the body's result


class Kind(enum.Enum):
    """What a graph node's value represents."""
    COMPUTED = enum.auto()  # computed by the graph, e.g. a loaded tile or a sum
    INPUT = enum.auto()  # an argument tensor, e.g. `x0`; device code only loads from it
    OUTPUT = enum.auto()  # the output tensor; device code only stores into it
    SIZE = enum.auto()  # a symbolic size
    FUSED = enum.auto()  # printed by its user, e.g. `_mask_to(-inf)` inside its `amax`
    TUPLE = enum.auto()  # the values a loop carries, one part each


class Element(enum.Enum):
    """What a lowered tensor's entries are. Extended reals here are never `⊤`."""
    REAL = enum.auto()
    EREAL_FINITE = enum.auto()  # extended reals, each finite
    EREAL_MAY_BE_BOTTOM = enum.auto()  # extended reals, each finite or `⊥`, e.g. a running max from -inf


@dataclasses.dataclass(frozen=True)
class LoweredValue:
    """A graph node's Lean expression and what it represents.

    Kernel argument: `LoweredValue(Atom("x0"), Kind.INPUT)`.
    Computed sum: `LoweredValue(App("Tensor.addSpec", (Atom("x0"), Atom("x1"))), Kind.COMPUTED)`.
    Output buffer: `LoweredValue(None, Kind.OUTPUT)`.
    Two carried values: `LoweredValue(Atom("acc1"), Kind.TUPLE, parts)`, the parts lowered as `acc1.1`
    and `acc1.2`.
    """
    expr: Term | None  # None for output buffers and nodes printed by their users
    kind: Kind
    parts: tuple[LoweredValue, ...] = ()  # a tuple's values, in order; empty for other kinds
    element: Element = Element.REAL  # what a tensor's entries are


@dataclasses.dataclass(frozen=True)
class LoweredGraph:
    """A graph's Lean code and the values it returns.

    Loop body updating a running max: `LoweredGraph(Let("v_mx", …, Atom("v_mx")), (LoweredValue(Atom("v_mx"), …),))`.
    """
    code: Term
    results: tuple[LoweredValue, ...]  # the graph's outputs, in order; empty if it returns nothing


class FxEmitter:
    """Walk an FX graph, printing ATen nodes; subclasses handle other node kinds."""

    size_ops = ("sym_size.int", "sym_size")  # ops whose value is a symbolic size

    def __init__(self, bridge, names: dict):
        self.bridge, self.names, self.nonempty_dims = bridge, names, set()
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

    def aten(self, n, env) -> LoweredValue:
        """Lower an ATen node with the bridge, then print its TorchLean tensor call.

        Each value keeps its captured shape: an elementwise operand is broadcast to the node's
        shape, and a `keepdim` reduction is reshaped to it.
        """
        if op_name(n) in _ELEMENTWISE_OPS:
            if n.kwargs or len(n.args) != 2:
                refuse(f"only {op_name(n)} of two operands without options is supported")
            args = self.elementwise_operands(n, n.args, env)
            return LoweredValue(App(_ELEMENTWISE_OPS[op_name(n)], tuple(args)), Kind.COMPUTED)
        operands = self.bridge._node_refs((n.args, n.kwargs), {m: m for m in n.all_input_nodes})
        if op_name(n) == "amax.default":
            # The IR has no max over an axis either; it prints as `Tensor.reduceMax`.
            kind = amax_kind(n)
        else:
            kind = self.bridge._lower_kind(n)
        args, shapes = [env[p].expr for p in operands], [self.shape(p) for p in operands]
        # TorchLean reductions drop the axis; reshape below restores it for keepdim=True.
        keepdim = bool(kind.get("keepdim"))
        if keepdim:
            kind = {**kind, "keepdim": False}
        call, nonempty_dims = lower_op(kind, args, shapes)
        self.nonempty_dims |= nonempty_dims
        return LoweredValue(self.reshape(call, self.shape(n)) if keepdim else call, Kind.COMPUTED)

    def elementwise_operands(self, n, operands, env: dict) -> list[Term]:
        """Bring each operand of elementwise `n` to `n`'s shape: broadcast a node as printed in `env`,
        fill a constant tensor with a number."""
        target = self.shape(n)
        prepared = []
        for p in operands:
            if _is_number(p):
                prepared.append(App("Tensor.full", (Atom(f"[{', '.join(target)}]"), real_literal(p))))
            else:
                prepared.append(self.broadcast(env[p].expr, self.shape(p), target))
        return prepared

    def broadcast(self, value: Term, shape: list[str], target: list[str]) -> Term:
        """Broadcast `value` between equal-rank shapes; each axis of `shape` must match `target` or be `1`."""
        if shape == target:
            return value
        if len(shape) != len(target):
            refuse(f"broadcasting {shape} to {target} is not supported")
        if any(s not in (t, "1") for s, t in zip(shape, target)):
            invalid(f"cannot broadcast {shape} to {target}")
        return App("Tensor.broadcastTo",
                   (Atom(f"(s₂ := [{', '.join(target)}])"), Atom("(by simp [Shape.CanBroadcastTo])"), value))

    def reshape(self, value: Term, target: list[str]) -> App:
        """Reshape `value` to `target`, e.g. a reduction's result to its `keepdim` shape."""
        return App("Tensor.reshapeSpec",
                   (Atom(f"(target := [{', '.join(target)}])"), value, Atom("(by simp [Shape.size])")))

    def node(self, n, env: dict, ctx: Context) -> LoweredValue | None:
        """Refuse non-ATen nodes unless a subclass supplies their translation."""
        refuse(f"op {op_name(n)} is not supported")

    def emit_graph(self, graph, env: dict, ctx: Context) -> LoweredGraph:
        """Return Lean code for a graph; `env` holds node values and `ctx` collects lets and output state."""
        results = ()
        for n in graph.nodes:
            if n.op == "placeholder":
                continue
            if n.op == "output":
                results = tuple(env[out] for out in n.args[0] or ())
            elif isinstance(n.target, torch._ops.OpOverload) and op_name(n) not in self.size_ops:
                env[n] = self.aten(n, env)
                if len(n.users) > 1:
                    ctx.lets.append((f"v_{n.name}", env[n].expr))
                    env[n] = dataclasses.replace(env[n], expr=Atom(f"v_{n.name}"))
            elif op_name(n) in self.size_ops:
                env[n] = LoweredValue(Atom(self.sym(n.meta["val"])), Kind.SIZE)
            else:
                value = self.node(n, env, ctx)
                if value is not None:
                    env[n] = value
        if not results:
            code = ctx.state
        elif len(results) == 1:
            code = results[0].expr
        else:
            code = Tup(tuple(r.expr for r in results))
        for name, value in reversed(ctx.lets):
            code = Let(name, value, code)
        return LoweredGraph(code, results)
