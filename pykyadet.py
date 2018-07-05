# -*- coding:utf-8 -*-
r"""
pykyadet: reverse mode 自動微分モジュール
普通に式を構築すると,その式がグラフの形で保存される
このグラフのnodeは基本演算(加減乗除やlog,sinなど)を表す
基本演算の組み合わせで構築される式ならどんな式であってもグラフを構築できる
それをたどることで微分演算を行う
複雑な式であっても実装された演算の組み合わせなら微分が可能である

$z = f(x,y)$とする.ある中間変数(node)をwとする.
wが表す基本演算をgとし,中間変数$u_i$がwに陽に依存するとする.
$\bar{w}=\frac{\partial z}{\partial w}$とし,他の変数についても同様とすると
連鎖律を用いて
$$
\begin{align*}
\bar{w}&=\frac{\partial z}{\partial w}\\
&=\sum_i\frac{\partial z}{\partial u_i}\frac{\partial u_i}{\partial w}\\
&=\sum_i\bar{u_i}\frac{\partial u_i}{\partial w}
\end{align*}
$$
と表せる.
また定義より$\bar{z}=1$であり,
$u_i$が表す基本演算を用いれば$\frac{\partial u_i}{\partial w}$は計算できるので
zから辿っていくことで全ての中間変数,変数wについて$\bar{w}$が計算できる

参考:https://ja.wikipedia.org/wiki/%E8%87%AA%E5%8B%95%E5%BE%AE%E5%88%86
"""

import math


class Var:
    r"""
    変数を表すクラス
    このクラスのオブジェクトは計算グラフで葉となる
    他のクラスもこのクラスを継承している
    self.val: この(中間)変数の値
    self.adj: 根変数をf,この変数をxとして$\frac{\partial f}{\partial x}$
    self.op_name: 演算子名
    self.args: 引数の辞書 keyはその引数の属性(演算子の右か左かなど),valueは引数(への参照)
    self.chain_executed:  chainが実行済みかのフラグ
    self.graphviz_called: graphviz用の出力をしたかのフラグ
    """

    def __init__(self, val):
        r"""
        :param val: このオブジェクトの値
        """
        self.val = val
        self.adj = 0
        self.op_name = "var"
        self.args = {}
        self.chain_executed = False
        self.graphviz_called = False

    def chain(self):
        r"""
        連鎖律を計算する関数
        演算を表すクラスではこれをオーバーライドしなければならない
        現在の実装では初期化時にchain_vv,chain_vf,chain_fvで上書きするようになっている
        Varクラスでは何もしない
        :return: no return
        """
        pass

    def grad(self):
        r"""
        全ての変数についてadjメンバを計算する
        ある変数についてadjメンバを計算するにはその変数に依存している変数を計算し終わっている必要があるので,幅優先で計算している
        :return: no return
        """
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
        r"""
        再帰的に全ての中間変数についてadjメンバを0リセットする.
        gradメソッドを呼ぶと微分を実行する前に呼ばれる
        :return: no return
        """
        self.adj = 0
        self.chain_executed = False
        for arg in self.args.values():
            if hasattr(arg, 'reset_adj_all'):
                arg.reset_adj_all()

    def __str__(self):
        return str(self.val)

    def to_s(self, symbol_table={}):
        for k, v in symbol_table.items():
            if id(v) == id(self):
                return str(k)
        return self.op_name + "(" + str(self.val) + ")"

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        elif isinstance(self, OpMonoVar):
            return self.arg == other.arg
        elif isinstance(self, OpBinVar):
            return (self.larg == other.larg) and (self.rarg == other.rarg)
        elif isinstance(self, Var):
            return id(self) == id(other)

    def graphviz(self, symbol_table={}, name=True, value=True, adj=True):
        r"""
        graphviz用の出力を得る
        :param symbol_table: {変数名:オブジェクト}となるdict locals()などを渡すとよい
        :param name: (あれば)変数名を出力するか
        :param value: valを出力するか
        :param adj: adjを出力するか
        :return: string for graphviz
        """
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
        r"""
        graphvizメソッドの最初で呼んでリセットする
        :return:
        """
        self.graphviz_called = False
        for arg in self.args.values():
            if hasattr(arg, 'reset_graphviz_called_all'):
                arg.reset_graphviz_called_all()

    def node_str(self, symbol_table, name, value, adj):
        r"""
        各nodeに関する情報をstring for graphvizで返す
        :param symbol_table: {変数名:オブジェクト}となるdict locals()などを渡すとよい
        :param name: (あれば)変数名を出力するか
        :param value: valを出力するか
        :param adj: adjを出力するか
        :return: string for graphviz
        """
        s = str(id(self)) + "[label=\"" + self.op_name + "\n"
        if name:
            for k, v in symbol_table.items():
                if id(v) == id(self):
                    s += "name: " + str(k) + "\n"
        if value:
            s += "val: " + str(self.val) + "\n"
        if adj:
            s += "adj: " + str(self.adj) + "\n"
        s += "\"];\n"
        return s


class OpMonoVar(Var):
    r"""
    単項演算子を表すクラス
    単項演算子はこれを継承して実装される
    """

    def __init__(self, val: float, arg: Var):
        r"""
        :param val: オブジェクトの値
        :param arg: この演算子の引数への参照
        """
        super().__init__(val)
        self.arg = arg
        self.args = {'arg': self.arg}

    def __str__(self):
        return self.op_name + "(" + str(self.arg) + ")"

    def to_s(self, symbol_table={}):
        return self.op_name + self.arg.to_s(symbol_table=symbol_table)


class OpNegVar(OpMonoVar):
    r"""
    単項マイナスを表すクラス
    """

    def __init__(self, arg: Var):
        r"""
        :param arg: 引数への参照
        """
        super().__init__(-arg.val, arg)
        self.op_name = "-"

    def chain(self):
        r"""
        $$
        \frac{\partial -x}{\partial x} = -1
        $$
        :return: no return
        """
        self.arg.adj -= self.adj


def v_neg(x: Var):
    r"""
    :param x:
    :return: -x
    """
    return OpNegVar(x)


Var.__neg__ = v_neg


class OpLogVar(OpMonoVar):
    r"""
    logを表すクラス
    底は自然対数
    """

    def __init__(self, arg: Var):
        r"""
        :param arg: 引数への参照
        """
        super().__init__(math.log(arg.val), arg)
        self.op_name = "log"

    def chain(self):
        r"""
        $$
        \frac{\partial}{\partial x}\log x = \frac{1}{x}
        $$
        :return:
        """
        self.arg.adj += self.adj / self.arg.val


def log(x: Var):
    r"""
    :param x:
    :return: log(x)
    """
    return OpLogVar(x)


class OpSinVar(OpMonoVar):
    r"""
    sinを表すクラス
    """

    def __init__(self, arg: Var):
        r"""
        :param arg: 引数への参照
        """
        super().__init__(math.sin(arg.val), arg)
        self.op_name = "sin"

    def chain(self):
        r"""
        $$
        \frac{\partial}{\partial x}\sin x = \cos x
        $$
        :return:
        """
        self.arg.adj += self.adj * math.cos(self.arg.val)


def sin(x: Var):
    r"""
    :param x:
    :return: sin(x)
    """
    return OpSinVar(x)


class OpCosVar(OpMonoVar):
    r"""
    cosを表すクラス
    """

    def __init__(self, arg: Var):
        r"""
        :param arg: 引数への参照
        """
        super().__init__(math.cos(arg.val), arg)
        self.op_name = "cos"

    def chain(self):
        r"""
        $$
        \frac{\partial}{\partial x}\cos x = -\sin x
        $$
        :return:
        """
        self.arg.adj -= self.adj * math.sin(self.arg.val)


def cos(x: Var):
    r"""
    :param x:
    :return: cos(x)
    """
    return OpCosVar(x)


class OpBinVar(Var):
    r"""
    二項演算子を表すクラス
    二項演算子はこれを継承して実装される
    """

    def __init__(self, val: float, l, r):
        r"""
        :param val: このオブジェクトの値
        :param l: 第一引数への参照
        :param r: 第二引数への参照
        ex: l + r
        """
        super().__init__(val)
        self.larg = l
        self.rarg = r
        self.args = {'left': self.larg, 'right': self.rarg}

    def __str__(self):
        return "(" + str(self.larg) + self.op_name + str(self.rarg) + ")"

    def to_s(self, symbol_table={}):
        if isinstance(self.larg, Var) and isinstance(self.rarg, Var):
            return "(" + self.larg.to_s(symbol_table) + self.op_name + self.rarg.to_s(symbol_table) + ")"
        elif not isinstance(self.larg, Var) and isinstance(self.rarg, Var):
            return "(" + str(self.larg) + self.op_name + self.rarg.to_s(symbol_table=symbol_table) + ")"
        elif isinstance(self.larg, Var) and not isinstance(self.rarg, Var):
            return "(" + self.larg.to_s(symbol_table=symbol_table) + self.op_name + str(self.rarg) + ")"


class OpAddVar(OpBinVar):
    r"""
    加算を表すクラス
    """

    def __init__(self, larg, rarg):
        r"""
        :param larg:
        :param rarg:
        larg, rargがそれぞれVarのinstanceかどうかで場合分け
        """
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
        r"""
        l,rともにVarのinstanceの場合のchainメソッド
        :return:
        $$
        \frac{\parial l + r}{\partial l} = 1
        \frac{\parial l + r}{\partial r} = 1
        $$
        """
        self.larg.adj += self.adj
        self.rarg.adj += self.adj

    def chain_vf(self):
        r"""
        lがVarのinstance,rがfloatの場合のchainメソッド
        :return:
        """
        self.larg.adj += self.adj

    def chain_fv(self):
        r"""
        lがfloat,rがVarのinstanceの場合のchainメソッド
        :return:
        """
        self.rarg.adj += self.adj


def v_add(l, r):
    return OpAddVar(l, r)


def v_radd(r, l):
    return OpAddVar(l, r)


Var.__add__ = v_add
Var.__radd__ = v_radd


# def v_iadd(l, r):  # l += r


class OpSubVar(OpBinVar):
    r"""
    減算を表すクラス
    """

    def __init__(self, larg: Var, rarg):
        r"""
        :param larg:
        :param rarg:
        larg, rargがそれぞれVarのinstanceかどうかで場合分け
        """
        if isinstance(larg, Var) and isinstance(rarg, Var):
            super().__init__(larg.val - rarg.val, larg, rarg)
            self.chain = self.chain_vv
        elif not isinstance(larg, Var) and isinstance(rarg, Var):
            super().__init__(larg - rarg.val, larg, rarg)
            self.chain = self.chain_fv
        elif isinstance(larg, Var) and not isinstance(rarg, Var):
            super().__init__(larg.val - rarg, larg, rarg)
            self.chain = self.chain_vf
        else:
            raise Exception("invalid type of argument")
        self.op_name = "-"

    def chain_vv(self):
        r"""
        l,rともにVarのinstanceの場合のchainメソッド
        :return:
        $$
        \frac{\parial l - r}{\partial l} = 1
        \frac{\parial l - r}{\partial r} = -1
        $$
        """
        self.larg.adj += self.adj
        self.rarg.adj -= self.adj

    def chain_vf(self):
        r"""
        lがVarのinstance,rがfloatの場合のchainメソッド
        :return:
        """
        self.larg.adj += self.adj

    def chain_fv(self):
        r"""
        lがfloat,rがVarのinstanceの場合のchainメソッド
        :return:
        """
        self.rarg.adj -= self.adj


def v_sub(l, r):
    return OpSubVar(l, r)


def v_rsub(r, l):
    return OpSubVar(l, r)


Var.__sub__ = v_sub
Var.__rsub__ = v_rsub


class OpMulVar(OpBinVar):
    r"""
    乗算を表すクラス
    """

    def __init__(self, larg: Var, rarg):
        r"""
        :param larg:
        :param rarg:
        larg, rargがそれぞれVarのinstanceかどうかで場合分け
        """
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
        r"""
        l,rともにVarのinstanceの場合のchainメソッド
        :return:
        $$
        \frac{\parial lr}{\partial l} = r
        \frac{\parial lr}{\partial r} = l
        $$
        """
        self.larg.adj += self.adj * self.rarg.val
        self.rarg.adj += self.adj * self.larg.val

    def chain_fv(self):
        r"""
        lがfloat,rがVarのinstanceの場合のchainメソッド
        :return:
        """
        self.rarg.adj += self.adj * self.larg

    def chain_vf(self):
        r"""
        lがVarのinstance,rがfloatの場合のchainメソッド
        :return:
        """
        self.larg.adj += self.adj * self.rarg


def v_mul(l, r):
    return OpMulVar(l, r)


def v_rmul(r, l):
    return OpMulVar(l, r)


Var.__mul__ = v_mul
Var.__rmul__ = v_rmul


class OpDivVar(OpBinVar):
    r"""
    除算を表すクラス
    """

    def __init__(self, larg: Var, rarg):
        r"""
        :param larg:
        :param rarg:
        larg, rargがそれぞれVarのinstanceかどうかで場合分け
        """
        if isinstance(larg, Var) and isinstance(rarg, Var):
            super().__init__(larg.val / rarg.val, larg, rarg)
            self.chain = self.chain_vv
        elif not isinstance(larg, Var) and isinstance(rarg, Var):
            super().__init__(larg / rarg.val, larg, rarg)
            self.chain = self.chain_fv
        elif isinstance(larg, Var) and not isinstance(rarg, Var):
            super().__init__(larg.val / rarg, larg, rarg)
            self.chain = self.chain_vf
        else:
            raise Exception("invalid type of argument")
        self.op_name = "/"

    def chain_vv(self):
        r"""
        l,rともにVarのinstanceの場合のchainメソッド
        :return:
        $$
        \frac{\parial l / r}{\partial l} = 1 / r
        \frac{\parial l / r}{\partial r} = -l / r^2
        $$
        """
        self.larg.adj += self.adj / self.rarg.val
        self.rarg.adj += self.adj * -self.larg.val / self.rarg.val ** 2

    def chain_fv(self):
        r"""
        lがfloat,rがVarのinstanceの場合のchainメソッド
        :return:
        """
        self.rarg.adj += self.adj * -self.larg / self.rarg.val ** 2

    def chain_vf(self):
        r"""
        lがVarのinstance,rがfloatの場合のchainメソッド
        :return:
        """
        self.larg.adj += self.adj / self.rarg


def v_div(l, r):
    return OpDivVar(l, r)


def v_rdiv(r, l):
    return OpDivVar(l, r)


Var.__truediv__ = v_div
Var.__rtruediv__ = v_rdiv


class OpPowVar(OpBinVar):
    r"""
    冪乗を表すクラス
    """

    def __init__(self, larg, rarg):
        r"""
        :param larg: base
        :param rarg: exponent
        larg, rargがそれぞれVarのinstanceかどうかで場合分け
        """
        if isinstance(larg, Var) and isinstance(rarg, Var):
            super().__init__(larg.val ** rarg.val, larg, rarg)
            self.chain = self.chain_vv
        elif not isinstance(larg, Var) and isinstance(rarg, Var):
            super().__init__(larg ** rarg.val, larg, rarg)
            self.chain = self.chain_fv
        elif isinstance(larg, Var) and not isinstance(rarg, Var):
            super().__init__(larg.val ** rarg, larg, rarg)
            self.chain = self.chain_vf
        else:
            raise Exception("invalid type of argument")
        self.args = {'base': self.larg, 'exponent': self.rarg}
        self.op_name = "**"

    def chain_vv(self):
        r"""
        l,rともにVarのinstanceの場合のchainメソッド
        :return:
        $$
        \frac{\parial l^r}{\partial l} = r * l^(r - 1)
        \frac{\parial l^r}{\partial r} = l^r * \log(l)
        $$
        """
        self.larg.adj += self.adj * self.rarg.val * self.larg.val ** (self.rarg.val - 1)
        self.rarg.adj += self.adj * self.larg.val ** self.rarg.val * math.log(self.larg.val)

    def chain_fv(self):
        r"""
        lがfloat,rがVarのinstanceの場合のchainメソッド
        :return:
        """
        self.rarg.adj += self.adj * self.larg ** self.rarg.val * math.log(self.larg)

    def chain_vf(self):
        r"""
        lがVarのinstance,rがfloatの場合のchainメソッド
        :return:
        """
        self.larg.adj += self.adj * self.rarg * self.larg.val ** (self.rarg - 1)


def v_pow(l, r):
    return OpPowVar(l, r)


def v_rpow(r, l):
    return OpPowVar(l, r)


Var.__pow__ = v_pow
Var.__rpow__ = v_rpow


def test():
    x = Var(math.pi)
    y = Var(1.0)
    z = log(x) + y ** 2.0  # 式の構築
    print(z.to_s(symbol_table=locals()))
    z.grad()  # dz/dx,dz/dyを計算(x.adj,y.adj)で取得できる
    s = z.graphviz(symbol_table=locals())  # graphviz用のstringを取得
    # 画像生成
    import os
    file = open("graph.dot", "w")
    print(s, file=file)
    file.close()
    os.system("dot -Tpng graph.dot -o graph.png")  # graph.pngに計算グラフが出力される


if __name__ == "__main__":
    test()
