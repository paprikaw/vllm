#!/usr/bin/env python3
"""
测试基准测试配置功能的脚本
"""

import json
import os

def test_config_loading():
    """测试配置文件加载功能"""
    print("测试基准测试配置功能...")
    
    # 测试多阶段配置文件
    config_file = "multi_stage_config_example.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print(f"✓ 成功加载配置文件: {config_file}")
            print(f"  配置类型: {config.get('benchmark_type', 'unknown')}")
            print(f"  描述: {config.get('description', 'N/A')}")
            
            request_rates = config.get("request_rates")
            num_requests = config.get("num_requests")
            
            if request_rates and num_requests:
                print(f"  阶段数: {len(request_rates)}")
                print("  阶段配置:")
                for i, (rate, num_req) in enumerate(zip(request_rates, num_requests)):
                    print(f"    阶段 {i+1}: {num_req} 个请求，速率 {rate} req/s")
                
                total_requests = sum(num_requests)
                print(f"  总请求数: {total_requests}")
                
                # 验证配置
                if len(request_rates) == len(num_requests):
                    print("  ✓ 配置验证通过")
                else:
                    print("  ✗ 配置验证失败: 数组长度不匹配")
            else:
                print("  ✗ 缺少必需的配置字段")
                
        except json.JSONDecodeError as e:
            print(f"✗ JSON解析错误: {e}")
        except Exception as e:
            print(f"✗ 文件读取错误: {e}")
    else:
        print(f"✗ 配置文件不存在: {config_file}")
    
    # 测试通用配置文件
    general_config_file = "benchmark_config_example.json"
    if os.path.exists(general_config_file):
        try:
            with open(general_config_file, 'r') as f:
                config = json.load(f)
            
            print(f"\n✓ 成功加载通用配置文件: {general_config_file}")
            print(f"  配置类型: {config.get('benchmark_type', 'unknown')}")
            
            # 检查多阶段配置
            multi_stage_config = config.get("multi_stage_config")
            if multi_stage_config:
                request_rates = multi_stage_config.get("request_rates")
                num_requests = multi_stage_config.get("num_requests")
                
                if request_rates and num_requests:
                    print(f"  多阶段配置: {len(request_rates)} 个阶段")
                    print("  阶段详情:")
                    for i, (rate, num_req) in enumerate(zip(request_rates, num_requests)):
                        print(f"    阶段 {i+1}: {num_req} 个请求，速率 {rate} req/s")
            
            # 检查其他配置
            if "dataset_config" in config:
                print("  ✓ 包含数据集配置")
            if "model_config" in config:
                print("  ✓ 包含模型配置")
            if "sampling_config" in config:
                print("  ✓ 包含采样配置")
            if "monitoring_config" in config:
                print("  ✓ 包含监控配置")
                
        except json.JSONDecodeError as e:
            print(f"✗ JSON解析错误: {e}")
        except Exception as e:
            print(f"✗ 文件读取错误: {e}")
    else:
        print(f"✗ 通用配置文件不存在: {general_config_file}")

def create_sample_configs():
    """创建示例配置文件"""
    print("\n创建示例配置文件...")
    
    # 创建简单的多阶段配置
    simple_config = {
        "benchmark_type": "multi_stage",
        "description": "简单两阶段测试",
        "request_rates": [10.0, 30.0],
        "num_requests": [100, 200],
        "stage_descriptions": [
            "阶段1: 低负载测试 (10 req/s)",
            "阶段2: 高负载测试 (30 req/s)"
        ]
    }
    
    try:
        with open("simple_test_config.json", "w") as f:
            json.dump(simple_config, f, indent=2, ensure_ascii=False)
        print("✓ 创建 simple_test_config.json")
    except Exception as e:
        print(f"✗ 创建配置文件失败: {e}")
    
    # 创建压力测试配置
    stress_config = {
        "benchmark_type": "multi_stage",
        "description": "压力测试配置",
        "request_rates": [1.0, 5.0, 20.0, 50.0],
        "num_requests": [50, 100, 150, 200],
        "stage_descriptions": [
            "阶段1: 预热 (1 req/s)",
            "阶段2: 低负载 (5 req/s)",
            "阶段3: 中负载 (20 req/s)",
            "阶段4: 高负载 (50 req/s)"
        ],
        "total_requests": 500,
        "expected_duration": "约30秒"
    }
    
    try:
        with open("stress_test_config.json", "w") as f:
            json.dump(stress_config, f, indent=2, ensure_ascii=False)
        print("✓ 创建 stress_test_config.json")
    except Exception as e:
        print(f"✗ 创建配置文件失败: {e}")

def main():
    """主函数"""
    print("基准测试配置功能测试")
    print("=" * 50)
    
    # 测试现有配置文件
    test_config_loading()
    
    # 创建示例配置文件
    create_sample_configs()
    
    print("\n测试完成！")
    print("\n使用说明:")
    print("1. 使用 --benchmark-config <config_file.json> 参数指定配置文件")
    print("2. 配置文件必须包含 'request_rates' 和 'num_requests' 数组")
    print("3. 两个数组的长度必须相同")
    print("4. 支持扩展配置字段，如描述、监控等")

if __name__ == "__main__":
    main()
