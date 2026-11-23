# 该脚本用于：演示 Python 继承与方法重写（多态）的基本用法。

class A():
    # 父类 A：提供 a1() 方法
    def a1(self):
        print('你好A')

    # b1() 中调用 a1()，实际执行哪个版本由运行时对象类型决定
    def b1(self):
        self.a1()

class B(A):
    # 子类 B：重写父类的 a1() 方法
    def a1(self,):
        print('你好B')

# 创建子类对象并调用继承自父类的 b1()，最终执行子类重写后的 a1()
b = B()
b.b1()

