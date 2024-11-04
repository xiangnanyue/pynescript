import itertools

from pynescript import ast
from pynescript.ast import NodeVisitor

from .opt import Operator
from .expression import Strategy, Series, Input, Const


class PineVisitor(NodeVisitor):
    def __init__(self, executor):
        self.executor = executor

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            if self.executor.scopes:
                for scope in self.executor.scopes:
                    if node.id in scope:
                        node_store = scope[node.id]
                        return self.executor.nodes[node_store]
            if node.id in self.executor.builtins:
                return self.executor.builtins[node.id]
            if node.id in self.executor.sources:
                return self.executor.sources[node.id]
        return node

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.ctx, ast.Load):
            value = self.visit(node.value)
            return getattr(value, node.attr)
        return node

    def visit_Constant(self, node: ast.Constant):
        return Const(node.value)

    def visit_Call(self, node: ast.Call):
        func = self.visit(node.func)

        args = []
        kwargs = {}

        found_has_name = False
        for arg in node.args:
            if arg.name:
                found_has_name = True
                kwargs[arg.name] = self.visit(arg.value)
            elif found_has_name:
                raise ValueError()
            else:
                args.append(self.visit(arg.value))
        # print(node.func, args, kwargs)
        result = func(*args, **kwargs)

        # 策略初始化
        if isinstance(result, Strategy) and self.executor.declaration is None:
            if result.default_qty_type is None:
                result.default_qty_type = Strategy.fixed
            if result.default_qty_value is None:
                result.default_qty_value = 1
            if result.initial_capital is None:
                result.initial_capital = 1000000

            self.executor.declaration = result
            self.executor.cash = result.initial_capital
            print(f"initial cash: {self.executor.cash}")

        # 策略执行逻辑
        if isinstance(result, Strategy.entry):
            price = (
                self.executor.sources["close"][0]
                if result.limit is None and result.stop is None else result.limit[0] or result.stop[0]
            )

            if self.executor.position_size != 0:
                if result.direction == Strategy.long and self.executor.position_size < 0:
                    self.executor.cash += 2 * self.executor.position_amount + self.executor.position_size * price
                    print(
                        f"{self.executor.current_date}: action=exit  "
                        f"direction=short price={price} "
                        f"quantity={-self.executor.position_size} cash={self.executor.cash} "
                    )
                    self.executor.position_size = 0
                    self.executor.position_amount = 0
                elif result.direction == Strategy.short and self.executor.position_size > 0:
                    self.executor.cash += self.executor.position_size * price
                    print(
                        f"{self.executor.current_date}: action=exit  "
                        f"direction=long  price={price} "
                        f"quantity={self.executor.position_size} cash={self.executor.cash}"
                    )
                    self.executor.position_size = 0
                    self.executor.position_amount = 0

            if result.qty is not None:
                quantity = result.qty[0]
            else:
                if self.executor.declaration.default_qty_type == Strategy.fixed:
                    quantity = self.executor.declaration.default_qty_value
                elif self.executor.declaration.default_qty_type == Strategy.cash:
                    cash = self.executor.declaration.default_qty_value
                    quantity = cash // price
                elif self.executor.declaration.default_qty_type == Strategy.percent_of_equity:
                    percent = self.executor.declaration.default_qty_value / 100
                    cash = self.executor.cash * percent
                    quantity = cash // price
                else:
                    print(self.executor.declaration.default_qty_type.data)
                    raise ValueError()

            cash_amount = price * quantity

            if self.executor.cash > cash_amount:
                if result.direction == Strategy.long and not self.executor.position_size > 0:
                    self.executor.cash -= cash_amount
                    print(
                        f"{self.executor.current_date}: action=enter direction=long  price={price} "
                        f"quantity={quantity} cash={self.executor.cash}"
                    )
                    self.executor.position_size = +quantity
                    self.executor.position_amount = cash_amount
                elif result.direction == Strategy.short and not self.executor.position_size < 0:
                    self.executor.cash -= cash_amount
                    print(
                        f"{self.executor.current_date}: action=enter direction=short price={price} "
                        f"quantity={quantity} cash={self.executor.cash}"
                    )
                    self.executor.position_size = -quantity
                    self.executor.position_amount = cash_amount

        return result

    def visit_Assign(self, node: ast.Assign):
        if node.target not in self.executor.nodes:
            self.executor.nodes[node.target] = Series([None])

        value = self.visit(node.value)
        if value is None:
            value = Series([None])

        if (
                isinstance(value, Input)
                and isinstance(node.target, ast.Name)
                and node.target.id in self.executor.inputs
        ):
            value.set(self.executor.inputs[node.target.id])

        self.executor.nodes[node.target].set(value[0])
        # print('node.target:', node.target, "node value:", value[0])

        if isinstance(node.target, ast.Name):
            self.executor.scopes[-1][node.target.id] = node.target

    def visit_Expr(self, node: ast.Expr):
        return self.visit(node.value)

    def visit_BoolOp(self, node: ast.BoolOp):
        if isinstance(node.op, ast.And):
            return [all(self.visit(value)[0] for value in node.values)]
        if isinstance(node.op, ast.Or):
            return [any(self.visit(value)[0] for value in node.values)]
        msg = f"unexpected node operator: {node.op}"
        raise ValueError(msg)

    def visit_BinOp(self, node: ast.BinOp):
        if isinstance(node.op, ast.Add):
            return [Operator.add(self.visit(node.left), self.visit(node.right))]
        if isinstance(node.op, ast.Sub):
            return [Operator.sub(self.visit(node.left), self.visit(node.right))]
        if isinstance(node.op, ast.Mult):
            return [Operator.mul(self.visit(node.left), self.visit(node.right))]
        if isinstance(node.op, ast.Div):
            return [Operator.truediv(self.visit(node.left), self.visit(node.right))]
        if isinstance(node.op, ast.Mod):
            return [Operator.mod(self.visit(node.left), self.visit(node.right))]
        msg = f"unexpected node operator: {node.op}"
        raise ValueError(msg)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return [not self.visit(node.operand)[0]]
        if isinstance(node.op, ast.UAdd):
            return [+self.visit(node.operand)[0]]
        if isinstance(node.op, ast.USub):
            return [-self.visit(node.operand)[0]]
        raise ValueError()

    def visit_If(self, node: ast.If):
        logic_value = self.visit(node.test)
        logic_value = Series([None]) if logic_value is None else logic_value
        if logic_value[0]:
            self.executor.scopes.append({})
            for stmt in node.body:
                self.visit(stmt)
            self.executor.scopes.pop()
        elif node.orelse:
            self.executor.scopes.append({})
            for stmt in node.orelse:
                self.visit(stmt)
            self.executor.scopes.pop()

    def visit_Compare(self, node: ast.Compare):  # noqa: C901, PLR0911, PLR0912
        left = self.visit(node.left)
        comparators = map(lambda x: self.visit(x), itertools.chain([node.left], node.comparators))
        comparator_pairs = list(itertools.pairwise(comparators))
        compare_ops = node.ops  # map(self.visit, node.ops)

        for op, (left, right) in zip(compare_ops, comparator_pairs, strict=True):
            if isinstance(op, ast.Eq):
                if not Operator.eq(left, right):
                    return [False]
            elif isinstance(op, ast.NotEq):
                if not Operator.ne(left, right):
                    return [False]
            elif isinstance(op, ast.Lt):
                if not Operator.lt(left, right):
                    return [False]
            elif isinstance(op, ast.LtE):
                if not Operator.le(left, right):
                    return [False]
            elif isinstance(op, ast.Gt):
                if not Operator.gt(left, right):
                    return [False]
            elif isinstance(op, ast.GtE):
                if not Operator.ge(left, right):
                    return [False]
            else:
                msg = f"unexpected node operator: {op}"
                raise ValueError(msg)
        return [True]

    def visit_Script(self, node: ast.Script):
        self.executor.scopes.append({})
        for stmt in node.body:
            self.visit(stmt)
        self.executor.scopes.pop()
