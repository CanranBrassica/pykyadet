import math
import os


# 変数クラス
class Var:
    def __init__(self, val):
        self.val = val  # 変数の値
        self.adj = 0  # 根変数をf,この変数をxとして$\frac{\partial f}{\partial x}$
        self.op_name = "var"  # 演算子名
        self.operands = []  # オペランドのリスト
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
            for operand in node.operands:
                operand.chain_calculated = True
                queue.append(operand)

    def reset_adj_all(self):
        self.adj = 0
        self.chain_executed = False

    def graphviz(self, symbol_table={}, name=True, value=True, adj=True):
        self.reset_graphviz_called_all()
        s = "digraph {\n"
        s += self.graphviz_impl(symbol_table, name, value, adj)
        s += "}"
        return s

    def reset_graphviz_called_all(self):
        self.graphviz_called = False

    def graphviz_impl(self, symbol_table, name, value, adj):
        if self.graphviz_called:
            return ""
        self.graphviz_called = True
        return self.node_str(symbol_table=symbol_table, name=name, value=value, adj=adj)

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


# 単項演算子の抽象クラス
class OpMonoVar(Var):
    def __init__(self, val: float, avi: Var):
        super().__init__(val)
        self.avi = avi
        self.operands = [self.avi]

    def reset_adj_all(self):
        self.adj = 0
        self.chain_executed = False
        self.avi.reset_adj_all()

    def graphviz_impl(self, symbol_table, name, value, adj):
        if self.graphviz_called:
            return ""
        self.graphviz_called = True
        s = self.node_str(symbol_table=symbol_table, name=name, value=value, adj=adj)
        s += str(id(self)) + "->" + str(id(self.avi)) + "[dir=back];\n"
        s += self.avi.graphviz_impl(symbol_table=symbol_table, name=name, value=value, adj=adj)
        return s

    def reset_graphviz_called_all(self):
        self.graphviz_called = False
        self.avi.reset_graphviz_called_all()


class OpLogVar(OpMonoVar):
    def __init__(self, avi: Var):
        super().__init__(math.log(avi.val), avi)
        self.op_name = "log"

    def chain(self):
        self.avi.adj += self.adj * 1.0 / self.avi.val


def log(x: Var):
    return OpLogVar(x)


# 二項演算子の抽象クラス
class OpBinVar(Var):
    def __init__(self, val: float, avi: Var, bvi):
        super().__init__(val)
        self.avi = avi
        self.bvi = bvi
        self.operands = [self.avi, self.bvi]

    def reset_adj_all(self):
        self.adj = 0
        self.chain_executed = False
        self.avi.reset_adj_all()
        if not isinstance(self.bvi, float):
            self.bvi.reset_adj_all()

    def graphviz_impl(self, symbol_table: dict, name: bool, value: bool, adj: bool):
        if self.graphviz_called:
            return ""
        self.graphviz_called = True
        s = self.node_str(symbol_table=symbol_table, name=name, value=value, adj=adj)
        s += str(id(self)) + "->" + str(id(self.avi)) + "[dir=back];\n"
        s += str(id(self)) + "->" + str(id(self.bvi)) + "[dir=back];\n"
        s += self.avi.graphviz_impl(symbol_table=symbol_table, name=name, value=value, adj=adj)
        if isinstance(self.bvi, float):
            s += str(id(self.bvi)) + "[label=\"" + str(self.bvi) + "\"];\n"
        else:
            s += self.bvi.graphviz_impl(symbol_table=symbol_table, name=name, value=value, adj=adj)
        return s

    def reset_graphviz_called_all(self):
        self.graphviz_called = False
        self.avi.reset_graphviz_called_all()
        self.bvi.reset_graphviz_called_all()


class OpAddVar(OpBinVar):
    def __init__(self, avi: Var, bvi):
        if isinstance(avi, Var) and isinstance(bvi, Var):
            super().__init__(avi.val + bvi.val, avi, bvi)
            self.chain = self.chain_vv
        elif isinstance(avi, Var) and (isinstance(bvi, float) or isinstance(bvi, int)):
            super().__init__(avi.val + bvi, avi, bvi)
            self.chain = self.chain_vf
        else:
            raise Exception("invalid type of argument")
        self.op_name = "+"

    def chain_vv(self):
        self.avi.adj += self.adj
        self.bvi.adj += self.adj

    def chain_vf(self):
        self.avi.adj += self.adj


def v_ladd(l: Var, r):
    return OpAddVar(l, r)


def v_radd(r: Var, l):
    return OpAddVar(r, l)


Var.__add__ = v_ladd
Var.__radd__ = v_radd


class OpMulVar(OpBinVar):
    def __init__(self, avi: Var, bvi):
        if isinstance(avi, Var) and isinstance(bvi, Var):
            super().__init__(avi.val * bvi.val, avi, bvi)
            self.chain = self.chain_vv
        elif isinstance(avi, Var) and (isinstance(bvi, float) or isinstance(bvi, int)):
            super().__init__(avi.val * bvi, avi, bvi)
            self.chain = self.chain_vf
        else:
            raise Exception("invalid type of argument")
        self.op_name = "*"

    def chain_vv(self):
        self.avi.adj += self.adj * self.bvi.val
        self.bvi.adj += self.adj * self.avi.val

    def chain_vf(self):
        self.avi.adj += self.adj * self.bvi.val


def v_lmul(l: Var, r):
    return OpMulVar(l, r)


def v_rmul(r: Var, l):
    return OpMulVar(r, l)


Var.__mul__ = v_lmul
Var.__rmul__ = v_rmul


def test():
    x = Var(2.0)
    y = Var(2.0)
    z = (x + y) * (x + y)  # 式の構築
    z.grad()  # dz/dx,dz/dyを計算(x.adj,y.adj)で取得できる
    s = z.graphviz(symbol_table=locals())  # graphviz用のstringを取得
    file = open("graph.dot", "w")
    file.write(s)
    file.close()
    os.system("dot -Tpng graph.dot -o graph.png")  # graph.pngに計算グラフが出力される


if __name__ == "__main__":
    test()
