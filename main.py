import math
import os


# 変数クラス
class Var:
    def __init__(self, val):
        self.val = val  # 変数の値
        self.adj = 0  # 根変数をf,この変数をxとして$\frac{\partial f}{\partial x}$
        self.op_name = "var"  # 演算子名
        self.args = {}  # 引数のリスト
        self.chain_executed = False  # chainが実行済みかのフラグ
        self.graphviz_called = False  # graphviz用の出力をしたかのフラグ

    def chain(self):
        pass

    def grad(self):
        self.reset_adj_all()
        self.adj = 1  # 定義より
        queue = []
        self.chain_executed = True
        queue.append(self)
        while queue:
            node = queue.pop()
            node.chain()
            for arg in node.args.values():
                if hasattr(arg, 'chain_executed') and not arg.chain_executed:
                    arg.chain_executed = True
                    queue.append(arg)

    def reset_adj_all(self):
        self.adj = 0
        self.chain_executed = False
        for arg in self.args.values():
            if hasattr(arg, 'reset_adj_all'):
                arg.reset_adj_all()

    def graphviz(self, symbol_table={}, name=True, value=True, adj=True):
        self.reset_graphviz_called_all()
        s = "digraph {\n"
        queue = []
        self.graphviz_called = True
        queue.append(self)
        while queue:
            node = queue.pop()
            if hasattr(node, 'node_str'):
                s += node.node_str(symbol_table=symbol_table, name=name, value=value, adj=adj)
            for key, arg in node.args.items():
                s += str(id(node)) + "->" + str(id(arg)) + "[dir=back, label=\"" + str(key) + "\"];\n"
                if not isinstance(arg, Var):
                    s += str(id(arg)) + "[label=\"" + str(arg) + "\"];\n"
                elif not arg.graphviz_called:
                    arg.graphviz_called = True
                    queue.append(arg)
        s += "}"
        return s

    def reset_graphviz_called_all(self):
        self.graphviz_called = False
        for arg in self.args.values():
            if hasattr(arg, 'reset_graphviz_called_all'):
                arg.reset_graphviz_called_all()

    def node_str(self, symbol_table, name, value, adj):
        s = str(id(self)) + "[label=\"" + self.op_name + "\n"
        if name:
            for k, v in symbol_table.items():
                if id(v) == id(self):
                    s += "name = " + str(k) + "\n"
        if value:
            s += "val = " + str(self.val) + "\n"
        if adj:
            s += "adj = " + str(self.adj) + "\n"
        s += "\"];\n"
        return s

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        elif isinstance(self, OpMonoVar):
            return self.arg == self.arg
        elif isinstance(self, OpBinVar):
            return self.larg == self.larg and self.rarg == self.rarg
        elif isinstance(self, Var):
            return id(self) == id(other)


# 単項演算子の抽象クラス
class OpMonoVar(Var):
    def __init__(self, val: float, arg: Var):
        super().__init__(val)
        self.arg = arg
        self.args = {'arg': self.arg}


class OpLogVar(OpMonoVar):
    def __init__(self, arg: Var):
        super().__init__(math.log(arg.val), arg)
        self.op_name = "log"

    def chain(self):
        self.arg.adj += self.adj * 1.0 / self.arg.val


def log(x: Var):
    return OpLogVar(x)


# 二項演算子の抽象クラス
class OpBinVar(Var):
    def __init__(self, val: float, l, r):
        super().__init__(val)
        self.larg = l
        self.rarg = r
        self.args = {'left': self.larg, 'right': self.rarg}


class OpAddVar(OpBinVar):
    def __init__(self, larg: Var, rarg):
        if isinstance(larg, Var) and isinstance(rarg, Var):
            super().__init__(larg.val + rarg.val, larg, rarg)
            self.chain = self.chain_vv
        elif not isinstance(larg, Var) and isinstance(rarg, Var):
            super().__init__(larg + rarg.val, larg, rarg)
            self.chain = self.chain_fv
        elif isinstance(larg, Var) and not isinstance(rarg, Var):
            super().__init__(larg.val + rarg, larg, rarg)
            self.chain = self.chain_vf
        else:
            raise Exception("invalid type of argument")
        self.op_name = "+"

    def chain_vv(self):
        self.larg.adj += self.adj
        self.rarg.adj += self.adj

    def chain_vf(self):
        self.larg.adj += self.adj

    def chain_fv(self):
        self.rarg.adj += self.adj


def v_add(l, r):
    return OpAddVar(r, l)


def v_radd(r, l):
    return OpAddVar(r, l)


Var.__add__ = v_add
Var.__radd__ = v_radd


class OpMulVar(OpBinVar):
    def __init__(self, larg: Var, rarg):
        if isinstance(larg, Var) and isinstance(rarg, Var):
            super().__init__(larg.val * rarg.val, larg, rarg)
            self.chain = self.chain_vv
        elif not isinstance(larg, Var) and isinstance(rarg, Var):
            super().__init__(larg * rarg.val, larg, rarg)
            self.chain = self.chain_fv
        elif isinstance(larg, Var) and not isinstance(rarg, Var):
            super().__init__(larg.val * rarg, larg, rarg)
            self.chain = self.chain_vf
        else:
            raise Exception("invalid type of argument")
        self.op_name = "*"

    def chain_vv(self):
        self.larg.adj += self.adj * self.rarg.val
        self.rarg.adj += self.adj * self.larg.val

    def chain_fv(self):
        self.rarg.adj += self.adj * self.larg.val

    def chain_vf(self):
        self.larg.adj += self.adj * self.rarg.val


def v_mul(l, r):
    return OpMulVar(l, r)


def v_rmul(r, l):
    return OpMulVar(l, r)


Var.__mul__ = v_mul
Var.__rmul__ = v_rmul


def test():
    x = Var(2.0)
    y = Var(2.0)
    z = (x + y) + 2.0  # 式の構築
    z.grad()  # dz/dx,dz/dyを計算(x.adj,y.adj)で取得できる
    s = z.graphviz(symbol_table=locals())  # graphviz用のstringを取得
    print(s)
    file = open("graph.dot", "w")
    file.write(s)
    file.close()
    os.system("dot -Tpng graph.dot -o graph.png")  # graph.pngに計算グラフが出力される


if __name__ == "__main__":
    test()
