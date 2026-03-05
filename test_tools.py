#!/usr/bin/env python3
"""测试新添加的工具是否正常工作"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import copy_file, move_file, delete_file, mkdir_p

def test_mkdir_p():
    """测试mkdir_p工具"""
    print("测试mkdir_p工具...")
    result = mkdir_p("test_dir")
    if result["ok"]:
        print("✓ mkdir_p成功: 创建了test_dir目录")
        return True
    else:
        print(f"✗ mkdir_p失败: {result.get('error', '未知错误')}")
        return False

def test_copy_file():
    """测试copy_file工具"""
    print("\n测试copy_file工具...")
    
    # 创建源文件
    with open("test_source.txt", "w") as f:
        f.write("测试copy_file工具的内容")
    
    # 复制文件
    result = copy_file("test_source.txt", "test_dest.txt")
    if result["ok"]:
        print("✓ copy_file成功: 复制了test_source.txt到test_dest.txt")
        
        # 验证文件内容
        with open("test_dest.txt", "r") as f:
            content = f.read()
        if content == "测试copy_file工具的内容":
            print("✓ 文件内容验证成功")
            return True
        else:
            print("✗ 文件内容验证失败")
            return False
    else:
        print(f"✗ copy_file失败: {result.get('error', '未知错误')}")
        return False

def test_move_file():
    """测试move_file工具"""
    print("\n测试move_file工具...")
    
    # 创建要移动的文件
    with open("to_move.txt", "w") as f:
        f.write("测试move_file工具的内容")
    
    # 移动文件
    result = move_file("to_move.txt", "moved.txt")
    if result["ok"]:
        print("✓ move_file成功: 移动了to_move.txt到moved.txt")
        
        # 验证源文件不存在，目标文件存在
        if not os.path.exists("to_move.txt") and os.path.exists("moved.txt"):
            print("✓ 文件移动验证成功")
            return True
        else:
            print("✗ 文件移动验证失败")
            return False
    else:
        print(f"✗ move_file失败: {result.get('error', '未知错误')}")
        return False

def test_delete_file():
    """测试delete_file工具"""
    print("\n测试delete_file工具...")
    
    # 创建要删除的文件
    with open("to_delete.txt", "w") as f:
        f.write("测试delete_file工具的内容")
    
    # 删除文件
    result = delete_file("to_delete.txt")
    if result["ok"]:
        print("✓ delete_file成功: 删除了to_delete.txt")
        
        # 验证文件不存在
        if not os.path.exists("to_delete.txt"):
            print("✓ 文件删除验证成功")
            return True
        else:
            print("✗ 文件删除验证失败")
            return False
    else:
        print(f"✗ delete_file失败: {result.get('error', '未知错误')}")
        return False

def cleanup():
    """清理测试文件"""
    files_to_clean = [
        "test_source.txt", "test_dest.txt", "to_move.txt", 
        "moved.txt", "to_delete.txt"
    ]
    dirs_to_clean = ["test_dir"]
    
    for file in files_to_clean:
        if os.path.exists(file):
            os.remove(file)
    
    for dir in dirs_to_clean:
        if os.path.exists(dir):
            import shutil
            shutil.rmtree(dir)
    
    print("\n清理完成")

def main():
    print("开始测试新添加的工具...")
    
    tests = [
        test_mkdir_p,
        test_copy_file,
        test_move_file,
        test_delete_file,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    cleanup()
    
    if passed == total:
        print("所有测试通过！")
        return 0
    else:
        print("部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())