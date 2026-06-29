#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bootstrap.py - 最小闭环验证脚本 (Minimal Closed-Loop Verification Script)

功能：
  一条命令验证整个系统配置。检查：
  1. 配置文件加载
  2. 交叉字段验证
  3. 依赖库可用性
  4. 输出目录可写性
  5. 平台和 TensorFlow 配置
  6. 配置和环境快照保存

使用方式：
  python verify_bootstrap.py [--config CONFIG_FILE] [--quiet]

示例：
  python verify_bootstrap.py
print("  后续步骤: python main.py --config config.yaml")  python verify_bootstrap.py --quiet  # 仅输出最终状态

输出：
  - 清晰的 ✓/✗ 检查列表
  - 快照文件路径
  - 最终状态: PASS / WARN / FAIL + actionable 建议

依赖：
  - config.py (Config 类)
  - environment.py (setup_logging, perform_dry_run, save_environment_snapshot 函数)
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import importlib.util

# 确保当前目录在 Python 路径中，用于直接运行脚本
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入配置和环境模块
try:
    # 尝试直接导入
    from src.utils.config import Config
    from src.utils.environment import (
        setup_logging,
        perform_dry_run,
        save_environment_snapshot,
    )
except ImportError as e:
    # 如果直接导入失败，尝试用 importlib 从文件路径加载
    try:
        config_path = os.path.join(current_dir, 'config.py')
        env_path = os.path.join(current_dir, 'environment.py')
        
        config_spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_module)
        Config = config_module.Config
        
        env_spec = importlib.util.spec_from_file_location("environment", env_path)
        env_module = importlib.util.module_from_spec(env_spec)
        env_spec.loader.exec_module(env_module)
        setup_logging = env_module.setup_logging
        perform_dry_run = env_module.perform_dry_run
        save_environment_snapshot = env_module.save_environment_snapshot
    except Exception as loader_error:
        print(f"✗ 导入失败: {e}")
        print(f"  备选方案也失败: {loader_error}")
        print("  请确保 config.py 和 environment.py 在当前目录中")
        sys.exit(1)


class BootstrapVerifier:
    """系统引导验证系统 - Performs comprehensive system bootstrap verification"""

    def __init__(self, config_file: str = "./config.yaml", supply_chain_file: str = "./supply_chain.yaml", quiet: bool = False):
        """
        初始化验证器
        
        Args:
            config_file: 配置文件路径 (默认: ./config.yaml)
            supply_chain_file: 供应链配置文件路径 (默认: ./supply_chain.yaml)
            quiet: 是否只输出最终状态 (不显示详细日志)
        """
        self.config_file = config_file
        self.supply_chain_file = supply_chain_file
        self.quiet = quiet
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "config_file": config_file,
            "checks": {},
            "warnings": [],
            "errors": [],
            "snapshots": {},
            "final_status": "unknown",
        }
        self.config: Optional[Config] = None
        self.dry_run_result: Optional[Dict[str, Any]] = None

    def _log(self, message: str, level: str = "INFO"):
        """输出日志（除非 quiet 模式）"""
        if not self.quiet:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level:8} - {message}")

    def _print_check(self, name: str, status: str, details: str = ""):
        """打印单个检查结果"""
        symbol = "✓" if status == "pass" else "✗" if status == "fail" else "⚠"
        print(f"  {symbol} {name:40} {status.upper():6} {details}")

    def step_1_load_config(self) -> bool:
        """步骤1: 加载和解析配置文件"""
        self._log("=" * 70)
        self._log("步骤 1/5: 加载配置文件")
        self._log("=" * 70)

        try:
            # 检查配置文件存在
            config_path = Path(self.config_file)
            if not config_path.exists():
                self._log(f"配置文件不存在: {self.config_file}", "ERROR")
                self.results["errors"].append(f"配置文件不存在: {self.config_file}")
                self._print_check("配置文件", "fail", f"未找到 {self.config_file}")
                return False

            self._log(f"找到配置文件: {config_path.absolute()}")

            # 加载配置
            self.config = Config()
            self._log(f"✓ 配置加载成功")
            self._print_check("配置文件加载", "pass", self.config_file)

            return True

        except Exception as e:
            self._log(f"配置加载失败: {e}", "ERROR")
            self.results["errors"].append(f"配置加载失败: {str(e)}")
            self._print_check("配置文件加载", "fail", str(e)[:40])
            return False

    def step_2_validate_config(self) -> bool:
        """步骤2: 验证配置交叉字段和完整性"""
        self._log("\n" + "=" * 70)
        self._log("步骤 2/5: 验证配置交叉字段和完整性")
        self._log("=" * 70)

        if not self.config:
            self._log("配置未加载，跳过验证", "ERROR")
            return False

        try:
            # 执行验证
            validation_errors = self.config._validate_cross_fields()

            if validation_errors:
                self._log(f"发现 {len(validation_errors)} 个验证错误:")
                for error in validation_errors:
                    self._log(f"  - {error}", "WARN")
                    self.results["warnings"].append(error)
                self._print_check("配置验证", "warn", f"{len(validation_errors)} 个警告")
                return True  # 有警告但继续
            else:
                self._log("✓ 所有交叉字段验证通过")
                self._print_check("配置验证", "pass", "无错误")
                return True

        except Exception as e:
            self._log(f"验证异常: {e}", "ERROR")
            self.results["errors"].append(f"配置验证异常: {str(e)}")
            self._print_check("配置验证", "fail", str(e)[:40])
            return False

    def step_3_system_checks(self) -> bool:
        """步骤3: 执行系统级检查 (使用 perform_dry_run)"""
        self._log("\n" + "=" * 70)
        self._log("步骤 3/5: 执行系统级检查")
        self._log("=" * 70)

        if not self.config:
            self._log("配置未加载，跳过系统检查", "ERROR")
            return False

        try:
            # 调用 perform_dry_run
            self.dry_run_result = perform_dry_run(
                config=self.config,
                config_file=self.config_file,
                supply_chain_file=self.supply_chain_file,
            )

            status = self.dry_run_result.get("status", "unknown")
            checks = self.dry_run_result.get("checks", {})
            warnings = self.dry_run_result.get("warnings", [])
            errors = self.dry_run_result.get("errors", [])

            # 记录详细检查结果
            for check_name, check_result in checks.items():
                check_status = check_result.get("status", "unknown")
                self._print_check(f"检查: {check_name}", check_status)

            # 记录警告和错误
            self.results["warnings"].extend(warnings)
            self.results["errors"].extend(errors)

            self._log(f"\n干运行状态: {status.upper()}")
            self._log(f"  警告数: {len(warnings)}")
            self._log(f"  错误数: {len(errors)}")

            return status in ["pass", "warn"]

        except Exception as e:
            self._log(f"系统检查异常: {e}", "ERROR")
            self.results["errors"].append(f"系统检查异常: {str(e)}")
            return False

    def step_4_save_snapshots(self) -> bool:
        """步骤4: 保存配置和环境快照"""
        self._log("\n" + "=" * 70)
        self._log("步骤 4/5: 保存配置和环境快照")
        self._log("=" * 70)

        if not self.config:
            self._log("配置未加载，无法保存快照", "ERROR")
            return False

        try:
            # 确保输出目录存在
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            # 保存配置快照
            config_snapshot_path = self.config.save_config_snapshot()
            self._log(f"✓ 配置快照: {config_snapshot_path}")
            self._print_check("配置快照", "pass", str(config_snapshot_path))
            self.results["snapshots"]["config"] = str(config_snapshot_path)

            # 保存环境快照
            try:
                env_snapshot_path = save_environment_snapshot(
                    config=self.config,
                    env_info=self.dry_run_result.get("checks", {}) if self.dry_run_result else {},
                    snapshot_dir=str(output_dir / "env_snapshots"),
                )
                self._log(f"✓ 环境快照: {env_snapshot_path}")
                self._print_check("环境快照", "pass", str(env_snapshot_path))
                self.results["snapshots"]["environment"] = str(env_snapshot_path)
            except Exception as e:
                self._log(f"环境快照保存失败: {e}", "WARN")
                self.results["warnings"].append(f"环境快照失败: {str(e)}")

            return True

        except Exception as e:
            self._log(f"快照保存异常: {e}", "ERROR")
            self.results["errors"].append(f"快照保存异常: {str(e)}")
            return False

    def step_5_final_report(self):
        """步骤5: 最终报告和建议"""
        self._log("\n" + "=" * 70)
        self._log("步骤 5/5: 最终报告")
        self._log("=" * 70)

        # 确定最终状态
        if self.results["errors"]:
            final_status = "FAIL"
        elif self.results["warnings"]:
            final_status = "WARN"
        else:
            final_status = "PASS"

        self.results["final_status"] = final_status

        # 打印最终摘要
        print("\n" + "=" * 70)
        print(f"最终状态: {final_status}")
        print("=" * 70)

        if self.config:
            print(f"\n📊 配置摘要:")
            print(f"  数据集: {self.config.dataset.name if self.config.dataset else 'N/A'}")
            if self.config.model:
                print(f"  模型: {self.config.model.architecture if self.config.model else 'N/A'}")
                print(f"  学习率: {self.config.training.learning_rate if self.config.training else 'N/A'}")
            if self.config.training:
                print(f"  轮数: {self.config.training.epochs}")
                print(f"  批大小: {self.config.training.batch_size}")

        if self.dry_run_result:
            checks = self.dry_run_result.get("checks", {})
            print(f"\n🔍 关键检查结果:")
            print(f"  平台检查: {checks.get('platform', {}).get('status', 'N/A')}")
            if "tensorflow" in checks:
                tf_info = checks["tensorflow"]
                print(
                    f"  TensorFlow: {tf_info.get('status', 'N/A')} "
                    f"(可用: {tf_info.get('available', False)})"
                )
            print(f"  输出目录: {checks.get('output_directories', {}).get('status', 'N/A')}")
            print(f"  依赖库: {checks.get('dependencies', {}).get('status', 'N/A')}")

        if self.results["snapshots"]:
            print(f"\n💾 快照文件:")
            for snapshot_type, path in self.results["snapshots"].items():
                print(f"  {snapshot_type}: {Path(path).name}")

        if self.results["warnings"]:
            print(f"\n⚠️  警告数: {len(self.results['warnings'])}")
            for i, warning in enumerate(self.results["warnings"][:5], 1):
                print(f"  {i}. {warning}")
            if len(self.results["warnings"]) > 5:
                print(f"  ... 还有 {len(self.results['warnings']) - 5} 个警告")

        if self.results["errors"]:
            print(f"\n❌ 错误数: {len(self.results['errors'])}")
            for i, error in enumerate(self.results["errors"][:5], 1):
                print(f"  {i}. {error}")
            if len(self.results["errors"]) > 5:
                print(f"  ... 还有 {len(self.results['errors']) - 5} 个错误")

        # 打印建议
        print(f"\n💡 建议:")
        if final_status == "PASS":
            print("  ✓ 系统配置完整，可以开始运行实验！")
            print("  后续步骤: python main.py --config config.yaml")
        elif final_status == "WARN":
            print("  ⚠️  存在一些警告，但系统可运行")
            print("  请查看上方的警告列表，并根据需要调整配置")
            print("  后续步骤: 修正警告后运行 python main.py")
        else:  # FAIL
            print("  ❌ 存在致命错误，无法继续")
            print("  请查看上方的错误列表进行修复")
            print("  常见问题:")
            print("    - 检查配置文件路径是否正确")
            print("    - 确保 Python 版本满足要求 (3.8+)")
            print("    - 验证所有依赖库已安装: pip install -r requirements.txt")
            print("    - Mac M1 用户: 使用 tensorflow-macos 而不是 tensorflow")

        print("\n" + "=" * 70)
        print(f"验证完成 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print("=" * 70 + "\n")

        return final_status == "PASS"

    def run(self) -> int:
        """运行完整验证流程"""
        print("\n" + "=" * 70)
        print("🚀 实验配置系统引导验证")
        print("=" * 70)
        print(f"配置文件: {self.config_file}")
        print("=" * 70 + "\n")

        # 执行每个步骤
        results = {
            "step1_load": self.step_1_load_config(),
            "step2_validate": self.step_2_validate_config(),
            "step3_system": self.step_3_system_checks(),
            "step4_snapshots": self.step_4_save_snapshots(),
        }

        # 最终报告
        self.step_5_final_report()

        # 保存结果到文件
        result_file = Path("outputs/verify_bootstrap_result.json")
        result_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            self._log(f"验证结果已保存: {result_file}")
        except Exception as e:
            self._log(f"无法保存结果文件: {e}", "WARN")

        # 返回出口代码
        if self.results["final_status"] == "PASS":
            return 0
        elif self.results["final_status"] == "WARN":
            return 1
        else:
            return 2


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="实验配置系统引导验证脚本\n\n"
        "一条命令检查系统配置、依赖库、输出目录和环境。"
        "生成配置和环境快照供后续参考。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python verify_bootstrap.py
print("  后续步骤: python main.py --config config.yaml")
  python verify_bootstrap.py --quiet

输出:
  - 清晰的检查列表和最终状态
  - 配置快照 (JSON)
  - 环境快照 (JSON)
  - 结果日志 (JSON)
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="配置文件路径 (默认: ./config.yaml)",
    )

    parser.add_argument(
        "--supply-chain",
        type=str,
        default="./supply_chain.yaml",
        help="供应链配置文件路径 (默认: ./supply_chain.yaml)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="仅输出最终状态，不显示详细日志",
    )

    args = parser.parse_args()

    # 创建验证器并运行
    verifier = BootstrapVerifier(config_file=args.config, supply_chain_file=args.supply_chain, quiet=args.quiet)
    exit_code = verifier.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
