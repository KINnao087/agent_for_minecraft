class SegmentTree:
    """
    线段树类，支持区间求和和区间更新
    
    功能特点：
    1. 支持区间查询（求和）
    2. 支持单点更新
    3. 支持区间更新（使用懒标记优化）
    4. 时间复杂度：查询和更新都是 O(log n)
    
    使用示例：
    >>> data = [1, 3, 5, 7, 9, 11]
    >>> seg_tree = SegmentTree(data)
    >>> seg_tree.query_range(0, 2)  # 查询区间[0,2]的和
    9
    >>> seg_tree.update_point(2, 10)  # 将索引2的值更新为10
    >>> seg_tree.update_range(1, 3, 5)  # 将区间[1,3]的值都增加5
    """
    
    def __init__(self, data):
        """
        初始化线段树
        Args:
            data: 原始数组
        """
        self.n = len(data)
        self.data = data
        # 线段树数组大小是原始数组的4倍
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)  # 懒标记数组
        self._build(0, 0, self.n - 1)
    
    def _build(self, node, start, end):
        """递归构建线段树"""
        if start == end:
            self.tree[node] = self.data[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            self._build(left_child, start, mid)
            self._build(right_child, mid + 1, end)
            self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def _push_down(self, node, start, end):
        """下推懒标记"""
        if self.lazy[node] != 0:
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            mid = (start + end) // 2
            
            # 更新子节点的值
            self.tree[left_child] += self.lazy[node] * (mid - start + 1)
            self.tree[right_child] += self.lazy[node] * (end - mid)
            
            # 下推懒标记到子节点
            self.lazy[left_child] += self.lazy[node]
            self.lazy[right_child] += self.lazy[node]
            
            # 清除当前节点的懒标记
            self.lazy[node] = 0
    
    def update_range(self, l, r, val):
        """
        区间更新
        Args:
            l: 区间左边界
            r: 区间右边界
            val: 要增加的值
        """
        if l > r:
            return  # 无效区间
        self._update_range(0, 0, self.n - 1, l, r, val)
    
    def _update_range(self, node, start, end, l, r, val):
        """递归实现区间更新"""
        if l > end or r < start:
            return
        
        if l <= start and end <= r:
            # 当前区间完全在更新区间内
            self.tree[node] += val * (end - start + 1)
            self.lazy[node] += val
            return
        
        # 下推懒标记
        self._push_down(node, start, end)
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        self._update_range(left_child, start, mid, l, r, val)
        self._update_range(right_child, mid + 1, end, l, r, val)
        
        # 更新当前节点
        self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def query_range(self, l, r):
        """
        区间查询
        Args:
            l: 区间左边界
            r: 区间右边界
        Returns:
            区间和
        """
        if l > r:
            return 0  # 无效区间
        return self._query_range(0, 0, self.n - 1, l, r)
    
    def _query_range(self, node, start, end, l, r):
        """递归实现区间查询"""
        if l > end or r < start:
            return 0
        
        if l <= start and end <= r:
            return self.tree[node]
        
        # 下推懒标记
        self._push_down(node, start, end)
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_sum = self._query_range(left_child, start, mid, l, r)
        right_sum = self._query_range(right_child, mid + 1, end, l, r)
        
        return left_sum + right_sum
    
    def update_point(self, index, val):
        """
        单点更新
        Args:
            index: 要更新的索引
            val: 新的值
        """
        if index < 0 or index >= self.n:
            return  # 索引越界
        self._update_point(0, 0, self.n - 1, index, val)
    
    def _update_point(self, node, start, end, index, val):
        """递归实现单点更新"""
        if start == end:
            self.tree[node] = val
            return
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        if index <= mid:
            self._update_point(left_child, start, mid, index, val)
        else:
            self._update_point(right_child, mid + 1, end, index, val)
        
        self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def get_array(self):
        """获取当前数组状态（用于调试）"""
        result = [0] * self.n
        self._get_array(0, 0, self.n - 1, result)
        return result
    
    def _get_array(self, node, start, end, result):
        """递归获取数组状态"""
        if start == end:
            result[start] = self.tree[node]
            return
        
        # 下推懒标记
        self._push_down(node, start, end)
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        self._get_array(left_child, start, mid, result)
        self._get_array(right_child, mid + 1, end, result)


# 演示代码
if __name__ == "__main__":
    print("线段树实现演示")
    print("=" * 50)
    
    # 示例1: 基本使用
    print("\n示例1: 基本使用")
    data = [1, 3, 5, 7, 9, 11]
    seg_tree = SegmentTree(data)
    
    print(f"原始数组: {data}")
    print(f"查询区间 [0, 2] 的和: {seg_tree.query_range(0, 2)}")
    print(f"查询区间 [2, 4] 的和: {seg_tree.query_range(2, 4)}")
    
    # 示例2: 单点更新
    print("\n示例2: 单点更新")
    seg_tree.update_point(2, 10)
    print(f"将索引2的值更新为10后，查询区间 [0, 2] 的和: {seg_tree.query_range(0, 2)}")
    
    # 示例3: 区间更新
    print("\n示例3: 区间更新")
    seg_tree.update_range(1, 3, 5)
    print(f"将区间[1,3]的值都增加5后，查询区间 [0, 3] 的和: {seg_tree.query_range(0, 3)}")
    
    # 示例4: 获取当前数组状态
    print(f"\n当前数组状态: {seg_tree.get_array()}")
    
    # 示例5: 边界情况处理
    print("\n示例5: 边界情况处理")
    print(f"无效区间查询 [3, 1]: {seg_tree.query_range(3, 1)}")
    print(f"单元素查询 [2, 2]: {seg_tree.query_range(2, 2)}")
    
    print("\n线段树实现完成！")